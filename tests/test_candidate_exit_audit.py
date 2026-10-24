"""Original public parsers/validator, realistic offline bytes and damaged evidence."""

import base64
import gzip
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

import pytest
import test_candidate_market as market_fixture
from test_candidate_catalog import NOW
from test_candidate_market import offline as offline
from test_candidate_study import study as study

from crypto_arbitrage_bot import candidate_exit_audit as module
from crypto_arbitrage_bot import candidate_study_control as cli
from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.candidate_study import write_public_archive
from crypto_arbitrage_bot.exchanges.funding_history import FundingPage, FundingWindow, _epoch_ms
from crypto_arbitrage_bot.market_data.candidate_catalog import build_candidate_catalog
from crypto_arbitrage_bot.market_data.candidate_market import CandidateMarketCollector
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

AT = NOW + timedelta(seconds=3630)


@pytest.fixture
def audited(study, monkeypatch):
    settings, catalog, runner, _, directory = study
    runner.run_once()
    record = next(v for v in runner.repository.list_recent_opportunities() if v.symbol == "THEUSDT")
    monkeypatch.setattr(market_fixture, "NOW", AT)
    monkeypatch.setattr(market_fixture, "MS", _epoch_ms(AT))
    sources = [
        replace(v, requested_at=v.requested_at + (AT - NOW), received_at=AT, finished_at=AT)
        for v in catalog.sources
    ]
    current = build_candidate_catalog(*sources, settings.research.candidate_catalog, now=AT)
    feed = CandidateMarketCollector(
        current,
        settings.research.candidate_catalog,
        settings.research.data_quality,
        timeout_seconds=3,
        funding_page_limit=1000,
        funding_max_pages=2,
    )
    observed = feed.collect("THEUSDT")
    settled = NOW + timedelta(hours=1)
    event = FundingEvent(
        f"THEUSDT:{_epoch_ms(settled)}", settled, Decimal("100"), Decimal(".001"), "THEUSDT"
    )
    page = FundingPage(
        AT + timedelta(milliseconds=10),
        AT + timedelta(milliseconds=100),
        200,
        json.dumps(
            [
                {
                    "symbol": "THEUSDT",
                    "fundingTime": _epoch_ms(settled),
                    "fundingRate": "0.001",
                    "markPrice": "100",
                }
            ]
        ).encode(),
    )
    window = FundingWindow("THEUSDT", NOW, AT, (event,), (page,))
    result = validate_forward_outcome(
        record, observed.assembly, validation_time=AT, funding_events=(event,)
    )
    assert result.status == "VALIDATED", result.reason_codes
    (task,) = runner.scheduler.tasks.list_for_opportunity(record.id)
    task = runner.scheduler.tasks.mark_running(task.id, now=AT, lease_seconds=120)
    runner.scheduler.validations.save_forward_validation(
        result, validation_id=task.validation_id, created_at=AT + timedelta(seconds=2)
    )
    assert runner.scheduler.tasks.finish(task, now=AT + timedelta(seconds=3))
    common = {
        "task_id": task.id,
        "opportunity_id": record.id,
        "validation_id": task.validation_id,
        "symbol": "THEUSDT",
        "registration_sha256": runner.studies["THEUSDT"].sha256,
        "recorded_at": AT,
    }
    packs = {
        "EXIT_MARKET": {
            **common,
            "phase": "EXIT_MARKET",
            "catalogs": [current],
            "captures": [{"phase": "EXIT_MARKET", "symbol": "THEUSDT", "observation": observed}],
        },
        "ACTUAL_FUNDING": {
            **common,
            "recorded_at": AT + timedelta(seconds=1),
            "phase": "ACTUAL_FUNDING",
            "catalogs": [],
            "captures": [{"phase": "ACTUAL_FUNDING", "symbol": "THEUSDT", "window": window}],
        },
    }
    paths = {}
    target = directory / "registry/exit-evidence"
    target.mkdir()
    for phase, pack in packs.items():
        path = target / f"{uuid4().hex}.json"
        write_public_archive(path, pack)
        paths[phase] = path.with_suffix(".json.gz")
    return settings, record, task, result, paths


def read_pack(audited, phase):
    return json.loads(gzip.decompress(audited[-1][phase].read_bytes()))


def rewrite(audited, phase, pack):
    audited[-1][phase].write_bytes(gzip.compress(exact_json(pack).encode()))


def audit(audited, **kwargs):
    return module.audit_candidate_exits(audited[0], evaluated_at=AT + timedelta(hours=1), **kwargs)


def assert_unavailable(report):
    assert report["status"] == "INCOMPLETE"
    assert report["all_audited_profits_negative"] is None
    item = report["items"][0]
    assert item["audit_status"] == "UNAVAILABLE"
    assert item["recomputed_net_profit"] is None and item["recomputed_net_return"] is None
    assert not report["trade_allowed"] and not report["qualification_issued"]


def test_full_original_projection_recomputed_from_four_sources_and_all_funding_rows(audited):
    settings, record, task, result, _ = audited
    before = hashlib.sha256(settings.research.sqlite_path.read_bytes()).hexdigest()
    report = audit(audited)
    assert report["status"] == "AUDITED_VALIDATED_EXITS" and report["verified_exits"] == 1
    item = report["items"][0]
    assert item["recomputed_net_profit"] == result.validated_simulated_net_profit
    assert item["recomputed_net_return"] == result.validated_simulated_net_return
    assert item["funding_event_count"] == 1 and item["original_decision"] == record.decision
    assert not report["realized_trade_profit"] and not report["external_authentication_proved"]
    assert before == hashlib.sha256(settings.research.sqlite_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        "depth_body",
        "premium_body",
        "interest_body",
        "request_symbol",
        "duplicate_params",
        "truncated",
        "http",
        "typed_book",
        "typed_snapshot",
        "quality",
        "candidate",
        "metadata_body",
    ],
)
def test_raw_or_typed_market_mismatch_never_reuses_saved_profit(audited, change):
    pack = read_pack(audited, "EXIT_MARKET")
    observation = pack["captures"][0]["observation"]
    assembly = observation["assembly"]
    key = {
        "depth_body": "perp_depth",
        "premium_body": "premium_index",
        "interest_body": "open_interest",
    }.get(change, "spot_depth")
    source = assembly["sources"][key]
    if change in ("depth_body", "premium_body", "interest_body"):
        raw = json.loads(base64.b64decode(source["raw_response"]["base64"]))
        field = {"depth_body": "E", "premium_body": "markPrice", "interest_body": "openInterest"}[
            change
        ]
        raw[field] = raw[field] + 1 if isinstance(raw[field], int) else "999"
        source["raw_response"] = {"base64": base64.b64encode(json.dumps(raw).encode()).decode()}
    elif change == "request_symbol":
        source["request"]["params"] = [["limit", 100], ["symbol", "BTCUSDT"]]
    elif change == "duplicate_params":
        source["request"]["params"].append(["symbol", "THEUSDT"])
    elif change == "truncated":
        source["body_truncated"] = True
    elif change == "http":
        source["status_code"] = 429
    elif change == "typed_book":
        source["order_book"]["bids"][0]["price"] = "999"
    elif change == "typed_snapshot":
        assembly["snapshot"]["mark_price"] = "999"
    elif change == "quality":
        observation["data_quality"]["issues"] = [{"code": "SAMPLING_SKEW_TOO_LARGE"}]
    elif change == "candidate":
        observation["candidate"]["spot_tick_size"] = ".1"
    else:
        raw = json.loads(
            base64.b64decode(pack["catalogs"][0]["sources"][0]["raw_response"]["base64"])
        )
        raw["symbols"][0]["filters"][0]["stepSize"] = "1"
        pack["catalogs"][0]["sources"][0]["raw_response"] = {
            "base64": base64.b64encode(json.dumps(raw).encode()).decode()
        }
    rewrite(audited, "EXIT_MARKET", pack)
    assert_unavailable(audit(audited))


@pytest.mark.parametrize(
    "change",
    [
        "omit_declared_event",
        "add_raw_event",
        "duplicate_raw_event",
        "wrong_symbol",
        "wrong_time",
        "wrong_mark",
        "wrong_event_id",
        "failed_window",
        "wrong_open",
        "noninteger_time",
        "page_limit_incomplete",
        "extra_page",
    ],
)
def test_funding_coverage_and_all_official_fields_must_match(audited, change):
    pack = read_pack(audited, "ACTUAL_FUNDING")
    window = pack["captures"][0]["window"]
    page = window["pages"][0]
    raw = json.loads(base64.b64decode(page["raw_response"]["base64"]))
    if change == "omit_declared_event":
        window["events"] = []
    elif change == "add_raw_event":
        raw.append({**raw[0], "fundingTime": raw[0]["fundingTime"] + 1000})
    elif change == "duplicate_raw_event":
        raw.append(raw[0])
    elif change == "wrong_symbol":
        raw[0]["symbol"] = "BTCUSDT"
    elif change == "wrong_time":
        raw[0]["fundingTime"] += 1000
    elif change == "wrong_mark":
        raw[0]["markPrice"] = "999"
    elif change == "wrong_event_id":
        window["events"][0]["event_id"] = "FORECAST_ONLY:fake"
    elif change == "failed_window":
        window["failure"] = "TIMEOUT"
    elif change == "wrong_open":
        window["opened_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif change == "noninteger_time":
        raw[0]["fundingTime"] = float(raw[0]["fundingTime"])
    elif change == "page_limit_incomplete":
        settings = audited[0]
        scheduler = settings.research.validation_scheduler.model_copy(
            update={"funding_page_limit": 1}
        )
        # Exercise the pure page-coverage decoder without changing registered policy.
        from crypto_arbitrage_bot.analytics.public_exit_evidence import decode_funding

        with pytest.raises(ValueError, match="RAW_FUNDING_COVERAGE"):
            decode_funding(
                window, symbol="THEUSDT", opened_at=NOW, closed_at=AT, criteria=scheduler
            )
        return
    else:
        window["pages"].append(page.copy())
    page["raw_response"] = {"base64": base64.b64encode(json.dumps(raw).encode()).decode()}
    rewrite(audited, "ACTUAL_FUNDING", pack)
    assert_unavailable(audit(audited))


@pytest.mark.parametrize(
    "column",
    [
        "funding_total",
        "spot_exit_fee",
        "capital_occupied",
        "validated_simulated_net_profit",
        "details_json",
    ],
)
def test_saved_projection_changes_cannot_be_confirmed_without_original_recomputation(
    audited, column
):
    settings, _, task, _, _ = audited
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        value = "{}" if column == "details_json" else "999"
        db.execute(
            f"UPDATE forward_validations SET {column}=? WHERE id=?", (value, task.validation_id)
        )
    assert_unavailable(audit(audited))


def test_missing_or_multiple_phase_archives_are_not_chosen_by_best_profit(audited):
    path = audited[-1]["EXIT_MARKET"]
    duplicate = path.with_name(f"{uuid4().hex}.json.gz")
    duplicate.write_bytes(path.read_bytes())
    assert_unavailable(audit(audited))
    duplicate.unlink()
    path.unlink()
    assert_unavailable(audit(audited))


def test_resource_limits_and_empty_results_do_not_emit_vacuous_negative_success(audited):
    assert_unavailable(audit(audited, limits=module.ExitAuditLimits(max_total_bytes=10)))
    settings = audited[0]
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("DELETE FROM forward_validations")
        db.execute("UPDATE validation_tasks SET status='FAILED',failure_code='COLLECTION_FAILED'")
    report = audit(audited)
    assert report["status"] == "NO_VALIDATED_EXITS" and report["verified_exits"] == 0
    assert report["all_audited_profits_negative"] is None


def test_cli_refuses_existing_report_before_scanning_and_keeps_new_report_read_only(
    audited, monkeypatch, capsys
):
    output = audited[-1]["EXIT_MARKET"].parent.parent / "audit.json"
    monkeypatch.setattr(cli, "load_settings", lambda _: audited[0])
    monkeypatch.setattr(cli, "utc_now", lambda: AT + timedelta(hours=1))
    argv = ["audit-exits", "--config", "offline.toml", "--output", str(output)]
    assert (
        cli.main(argv) == 0
        and json.loads(output.read_text(encoding="utf-8"))["verified_exits"] == 1
    )
    before = output.read_bytes()
    monkeypatch.setattr(cli, "audit_candidate_exits", Mock(side_effect=AssertionError("no rescan")))
    assert cli.main(argv) == 1 and output.read_bytes() == before
    assert "raw_response" not in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["missing", "wrong_hash", "late", "duplicate_field"])
def test_original_entry_receipt_must_still_bind_record_policy_and_time(audited, damage):
    settings, record, _, _, _ = audited
    _, studies = module.load_index(settings)
    p = studies[record.symbol].receipt_directory / (
        hashlib.sha256(record.id.encode()).hexdigest() + ".json"
    )
    if damage == "missing":
        p.unlink()
    else:
        value = json.loads(p.read_text(encoding="utf-8"))
        if damage == "wrong_hash":
            value["record_sha256"] = "0" * 64
        elif damage == "late":
            value["captured_at"] = (NOW + timedelta(hours=1)).isoformat()
        else:
            p.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            assert_unavailable(audit(audited))
            return
        p.write_text(exact_json(value), encoding="utf-8")
    assert_unavailable(audit(audited))


@pytest.mark.parametrize("bound", ["task", "validation", "publication"])
def test_snapshot_task_and_archive_original_timing_is_checked(audited, bound):
    settings, _, task, _, _ = audited
    if bound == "publication":
        pack = read_pack(audited, "EXIT_MARKET")
        pack["recorded_at"] = (AT + timedelta(seconds=1)).isoformat()
        rewrite(audited, "EXIT_MARKET", pack)
    else:
        with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
            if bound == "task":
                db.execute(
                    "UPDATE validation_tasks SET due_at=? WHERE id=?",
                    ((AT + timedelta(seconds=1)).isoformat(), task.id),
                )
            else:
                db.execute(
                    "UPDATE forward_validations SET validation_time=? WHERE id=?",
                    ((AT + timedelta(seconds=301)).isoformat(), task.validation_id),
                )
    assert_unavailable(audit(audited))


def test_file_and_record_limits_are_reported_without_partial_complete_claim(audited):
    assert audit(audited, limits=module.ExitAuditLimits(max_files=1))["status"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="EXIT_AUDIT_RECORD_LIMIT"):
        audit(audited, limits=module.ExitAuditLimits(max_records=1))


def test_expired_read_deadline_does_not_publish_success(audited, monkeypatch):
    monkeypatch.setattr(module, "monotonic", Mock(side_effect=[0, 100]))
    with pytest.raises((ValueError, sqlite3.OperationalError)):
        audit(audited, limits=module.ExitAuditLimits(max_seconds=1))


@pytest.mark.parametrize("value", [True, 0, -1, 121])
def test_invalid_resource_limit_is_rejected(value):
    with pytest.raises(ValueError, match="BOUNDED_EXIT_AUDIT_LIMITS_REQUIRED"):
        module.ExitAuditLimits(max_seconds=value)


def test_changed_archive_between_scan_and_recompute_is_not_trusted(audited, monkeypatch):
    path = audited[-1]["EXIT_MARKET"]
    original = gzip.open
    reads = [0]

    def opened(file, *args, **kwargs):
        if file == path:
            reads[0] += 1
            if reads[0] == 2:
                value = read_pack(audited, "EXIT_MARKET")
                value["recorded_at"] = (AT - timedelta(milliseconds=1)).isoformat()
                rewrite(audited, "EXIT_MARKET", value)
        return original(file, *args, **kwargs)

    monkeypatch.setattr(module.gzip, "open", opened)
    assert_unavailable(audit(audited))


def test_cli_incomplete_audit_is_saved_with_nonzero_status(audited, monkeypatch):
    audited[-1]["EXIT_MARKET"].unlink()
    output = audited[-1]["ACTUAL_FUNDING"].parent.parent / "incomplete-audit.json"
    monkeypatch.setattr(cli, "load_settings", lambda _: audited[0])
    monkeypatch.setattr(cli, "utc_now", lambda: AT + timedelta(hours=1))
    assert cli.main(["audit-exits", "--config", "offline.toml", "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "INCOMPLETE"


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_original_reference_catalog_hash_remains_required(audited, damage):
    reference = audited[0].research.study_registration.path.parent / "reference-catalog.json"
    if damage == "missing":
        reference.unlink()
    else:
        reference.write_bytes(b"{}")
    report = audit(audited)
    assert_unavailable(report)
    assert report["original_catalog_sha256_verified"] is False
