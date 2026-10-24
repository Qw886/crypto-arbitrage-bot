"""Task failure text is bounded without losing source diagnostics or evidence."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from decimal import Decimal

import pytest
from test_forward_validation import AT, OPENED, later_snapshot
from test_validation_scheduler import setup_scheduler

from crypto_arbitrage_bot.exchanges.funding_history import (
    FundingFailure,
    FundingPage,
    FundingWindow,
)
from crypto_arbitrage_bot.exchanges.http_client import RequestPhase
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.market import OrderBookLevel
from crypto_arbitrage_bot.models.snapshot import SnapshotSources
from crypto_arbitrage_bot.models.spot import SpotErrorKind, SpotFetchError
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind, UsdtmFetchError
from crypto_arbitrage_bot.models.validation_task import TaskFailure, TaskStatus
from crypto_arbitrage_bot.utils.failure_summary import (
    MAX_FAILURE_REASON_CHARS,
    TRUNCATED_MARKER,
    collection_failure_summary,
    limit_failure_reason,
)


def source_rows(repository):
    with closing(sqlite3.connect(repository.config.database_path)) as db:
        return db.execute("SELECT * FROM opportunity_sources ORDER BY opportunity_id,position").fetchall()


def test_huge_collection_failure_saves_only_diagnostics_and_preserves_original_sources(tmp_path):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    before = source_rows(scheduler.opportunities)
    original = later_snapshot()
    sources = original.sources
    huge_body = b"UNIQUE_RESPONSE_PAYLOAD" * 100000
    big_book = replace(sources.spot_depth.order_book, bids=tuple(
        OrderBookLevel(Decimal("110") - Decimal(i) / Decimal("100000"), Decimal("1"))
        for i in range(2000)))
    sources = replace(
        sources,
        spot_depth=replace(sources.spot_depth, order_book=big_book, raw_response=huge_body),
        perp_depth=replace(sources.perp_depth, data=None, raw_response=huge_body, elapsed_ms=3000,
                           error=UsdtmFetchError(UsdtmErrorKind.TIMEOUT, "timeout detail " * 10000,
                                                phase=RequestPhase.REQUEST)),
        premium_index=replace(sources.premium_index, data=None, raw_response=huge_body, status_code=429,
                              error=UsdtmFetchError(UsdtmErrorKind.RATE_LIMITED, "rate limit", -1003,
                                                   RequestPhase.RESPONSE_BODY)),
        open_interest=replace(sources.open_interest, data=None, raw_response=huge_body, status_code=503,
                              error=UsdtmFetchError(UsdtmErrorKind.HTTP_ERROR, "service unavailable",
                                                   phase=RequestPhase.RESPONSE_HEADERS)),
    )
    failed = assemble_snapshot("BTCUSDT", sources, assembled_at=AT)
    scheduler.collector.collect.side_effect = lambda symbol: failed
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.status is TaskStatus.FAILED and task.failure_code is TaskFailure.COLLECTION_FAILED
    assert len(task.failure_reason) <= MAX_FAILURE_REASON_CHARS
    payload = json.loads(task.failure_reason)
    assert payload["failure_code"] == "COLLECTION_FAILED" and payload["truncated"] is True
    details = {item["source"]: item for item in payload["sources"]}
    assert set(details) == {"PERP_DEPTH", "PREMIUM_INDEX", "OPEN_INTEREST"}
    assert details["PERP_DEPTH"]["kind"] == "TIMEOUT"
    assert details["PERP_DEPTH"]["phase"] == "REQUEST"
    assert details["PERP_DEPTH"]["elapsed_ms"] == 3000
    assert details["PREMIUM_INDEX"]["api_code"] == -1003
    assert details["PREMIUM_INDEX"]["http_status"] == 429
    assert details["PREMIUM_INDEX"]["endpoint"] == "PERP_PREMIUM_INDEX"
    assert details["OPEN_INTEREST"]["kind"] == "HTTP_ERROR"
    assert details["OPEN_INTEREST"]["http_status"] == 503
    assert details["OPEN_INTEREST"]["phase"] == "RESPONSE_HEADERS"
    for forbidden in ("raw_response", "base64", "bids", "asks", "order_book", "UNIQUE_RESPONSE_PAYLOAD"):
        assert forbidden not in task.failure_reason
    assert source_rows(scheduler.opportunities) == before
    assert scheduler.opportunities.get_opportunity_by_id(record.id) == record
    assert sources.spot_depth.raw_response is huge_body and sources.spot_depth.order_book is big_book
    assert not scheduler.validations.list_validations_for_opportunity(record.id)
    scheduler.funding.fetch_window.assert_not_called()


def test_all_four_sources_keep_codes_even_when_message_json_escaping_expands():
    snapshot = later_snapshot()
    values = {}
    for name in ("spot_depth", "perp_depth", "premium_index", "open_interest"):
        source = getattr(snapshot.sources, name)
        if name == "spot_depth":
            error = SpotFetchError(SpotErrorKind.API_ERROR, "\x00" * 50000, -1121, RequestPhase.PARSE)
            values[name] = replace(source, order_book=None, error=error)
        else:
            error = UsdtmFetchError(UsdtmErrorKind.API_ERROR, "\x00" * 50000, -1121, RequestPhase.PARSE)
            values[name] = replace(source, data=None, error=error)
    failed = assemble_snapshot("BTCUSDT", replace(snapshot.sources, **values), assembled_at=AT)
    text = collection_failure_summary(failed)
    assert len(text) <= MAX_FAILURE_REASON_CHARS
    payload = json.loads(text)
    assert payload["truncated"] is True and len(payload["sources"]) == 4
    assert all(s["api_code"] == -1121 and s["kind"] == "API_ERROR" and s["phase"] == "PARSE"
               and s["endpoint"] for s in payload["sources"])
    assert text == collection_failure_summary(failed)


def test_missing_sources_remain_explicit_without_fake_error_codes():
    failed = assemble_snapshot("ETHUSDT", SnapshotSources(), assembled_at=AT)
    payload = json.loads(collection_failure_summary(failed))
    assert len(payload["sources"]) == 4 and not payload["truncated"]
    assert all(s["issues"] == ["MISSING"] and s["kind"] is None and s["endpoint"] is None
               for s in payload["sources"])


def test_funding_failure_does_not_embed_raw_pages(tmp_path):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    body = b"FUNDING_BODY_MUST_NOT_BE_STORED" * 10000
    window = FundingWindow(record.symbol, OPENED, AT,
                           pages=(FundingPage(AT, AT, 429, body),),
                           failure=FundingFailure.RATE_LIMIT, message="rate limit " * 10000)
    scheduler.funding.fetch_window.side_effect = None
    scheduler.funding.fetch_window.return_value = window
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.status is TaskStatus.FAILED and task.failure_code is TaskFailure.FUNDING_UNAVAILABLE
    assert len(task.failure_reason) <= MAX_FAILURE_REASON_CHARS
    summary = json.loads(task.failure_reason)
    assert summary["sources"][0]["kind"] == "RATE_LIMIT"
    assert summary["sources"][0]["http_status"] == 429
    assert summary["sources"][0]["endpoint"] == "PERP_FUNDING_RATE"
    assert summary["truncated"] is True
    assert "raw_response" not in task.failure_reason and "FUNDING_BODY" not in task.failure_reason
    assert window.pages[0].raw_response is body
    assert not scheduler.validations.list_validations_for_opportunity(record.id)


@pytest.mark.parametrize("size", [2047, 2048, 2049, 100000])
def test_repository_applies_limit_even_for_direct_writer(tmp_path, size):
    scheduler, clock, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    pending, = scheduler.tasks.list_for_opportunity(record.id)
    task = scheduler.tasks.mark_running(pending.id, now=clock[0], lease_seconds=60)
    reason = "诊" * size
    assert scheduler.tasks.finish(task, now=clock[0], failure=TaskFailure.PERSISTENCE_FAILED, reason=reason)
    saved = scheduler.tasks.get(task.id)
    assert saved.status is TaskStatus.FAILED and saved.failure_code is TaskFailure.PERSISTENCE_FAILED
    assert len(saved.failure_reason) <= MAX_FAILURE_REASON_CHARS
    if size > MAX_FAILURE_REASON_CHARS:
        assert saved.failure_reason.endswith(TRUNCATED_MARKER)
    else:
        assert saved.failure_reason == reason


def test_unexpected_collection_exception_is_bounded_before_task_log(tmp_path):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    scheduler.collector.collect.side_effect = TimeoutError("read timeout " * 10000)
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.status is TaskStatus.FAILED and task.failure_code is TaskFailure.COLLECTION_FAILED
    assert task.failure_reason.startswith("TimeoutError: read timeout")
    assert task.failure_reason.endswith(TRUNCATED_MARKER)
    logged = json.loads(scheduler.logger.info.call_args.args[0])
    assert logged["failure_reason"] == task.failure_reason


@pytest.mark.parametrize("reason", [None, "", "ValueError: [\"STALE_SPOT_DATA\"]"])
def test_short_and_absent_reasons_are_unchanged(reason):
    assert limit_failure_reason(reason) == reason
