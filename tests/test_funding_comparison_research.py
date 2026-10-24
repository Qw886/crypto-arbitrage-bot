"""Offline HTTP + real SQLite comparison entry point; no source writes."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from test_forward_validation import research_record
from test_funding_comparison import NOW

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges import binance_usdtm, funding_history
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repository = SQLiteOpportunityRepository(SQLiteConfig(tmp_path / "sources.sqlite"))
    repository.initialize()
    records = tuple(replace(research_record(symbol), id=f"source-{symbol}") for symbol in ("BTCUSDT", "ETHUSDT"))
    for record in records:
        repository.save_opportunity(record)
    settings = load_settings(Path("docs/research-multisettlement.toml"))
    settings = settings.model_copy(update={"research": settings.research.model_copy(
        update={"sqlite_path": repository.config.database_path})})
    monkeypatch.setenv("HTTPS_PROXY", "")
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setattr(binance_usdtm, "utc_now", lambda: NOW)
    monkeypatch.setattr(funding_history, "utc_now", lambda: NOW)
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline HTTP only")))
    requests = []
    failures = {}

    def connection_factory(*args, **kwargs):
        connection = Mock()
        reply = Mock()

        def request(method, route, *, headers):
            assert method == "GET" and "X-MBX-APIKEY" not in headers
            requests.append(route)
            parts = urlsplit(route)
            query = parse_qs(parts.query)
            symbol = query.get("symbol", [None])[0]
            error = failures.get(symbol)
            if error is not None:
                raise error
            if parts.path == "/fapi/v1/fundingRate":
                delta = NOW - datetime(1970, 1, 1, tzinfo=UTC)
                data = [{"symbol": symbol, "fundingTime": (delta.days * 86400 + delta.seconds) * 1000,
                         "fundingRate": "0.001", "markPrice": "100"}]
            else:
                assert parts.path == "/fapi/v1/fundingInfo"
                data = [{"symbol": "BTCUSDT", "adjustedFundingRateCap": "0.003",
                         "adjustedFundingRateFloor": "-0.003", "fundingIntervalHours": 8}]
            body = json.dumps(data).encode()
            reply.status = 200
            reply.getheaders.return_value = [("Content-Length", str(len(body)))]
            reply.read.return_value = body

        connection.request.side_effect = request
        connection.getresponse.return_value = reply
        return connection

    monkeypatch.setattr(binance_usdtm, "HTTPSConnection", connection_factory)
    monkeypatch.setattr(funding_history, "HTTPSConnection", connection_factory)
    return repository, settings, records, requests, failures


def test_same_cutoff_collection_preserves_source_records_and_uses_actual_history(setup, tmp_path):
    from crypto_arbitrage_bot.funding_comparison_research import research_funding_comparison

    repository, settings, records, requests, _ = setup
    before = exact_json(repository.list_recent_opportunities(10))
    summary = research_funding_comparison(settings, tuple(r.id for r in records),
        closed_at=NOW, lookback_seconds=(28800, 86400), output=tmp_path / "comparison.json", clock=lambda: NOW)
    assert summary["status"] == "COMPLETE" and summary["orders_submitted"] is False
    report = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert len(report["history_sources"]) == 2
    assert report["comparison"]["rows"][0]["study"]["slices"][0]["historical_short_funding_at_reference_quantity"] == "0.2000"
    queries = [parse_qs(urlsplit(r).query) for r in requests if "/fundingRate?" in r]
    assert len(queries) == 2 and queries[0]["startTime"] == queries[1]["startTime"]
    assert queries[0]["endTime"] == queries[1]["endTime"] and len(requests) == 3
    assert exact_json(repository.list_recent_opportunities(10)) == before
    assert all("net_profit" not in row for row in summary["horizons"][0]["rows"])


def test_btc_timeout_does_not_block_eth_or_modify_original_database(setup, tmp_path):
    from crypto_arbitrage_bot.funding_comparison_research import research_funding_comparison

    repository, settings, records, requests, failures = setup
    failures["BTCUSDT"] = TimeoutError("offline timeout")
    before = repository.config.database_path.read_bytes()
    summary = research_funding_comparison(settings, tuple(r.id for r in records),
        closed_at=NOW, lookback_seconds=(28800,), output=tmp_path / "partial.json", clock=lambda: NOW)
    assert summary["status"] == "PARTIAL" and len(requests) == 3
    btc, eth = summary["horizons"][0]["rows"]
    assert btc["observed_funding"] is None and btc["cost_coverage_ratio"] is None
    assert eth["observed_funding"] == Decimal(".2")
    assert summary["horizons"][0]["unranked_symbols"] == ("BTCUSDT",)
    assert repository.config.database_path.read_bytes() == before


def test_known_archived_settlement_missing_from_history_is_unavailable_not_zero(setup, tmp_path):
    from crypto_arbitrage_bot.funding_comparison_research import research_funding_comparison

    repository, settings, records, _, _ = setup
    # Synthetic archived quote declared a due event which the mock history lacks.
    later = replace(records[0], id="known-due", next_funding_time=NOW.replace(hour=11))
    repository.save_opportunity(later)
    summary = research_funding_comparison(settings, (later.id,), closed_at=NOW,
        lookback_seconds=(28800,), output=tmp_path / "missing.json", clock=lambda: NOW)
    assert summary["status"] == "UNAVAILABLE"
    assert summary["horizons"][0]["rows"][0]["observed_funding"] is None
    assert summary["history_failures"][0]["failure"] == "INVALID_RESPONSE"


@pytest.mark.parametrize("ids,seconds", [((), (28800,)), (("not-in-db",), (28800,)),
                                      (("source-BTCUSDT",), (0,)), (("source-BTCUSDT",), (28800,28800))])
def test_invalid_configuration_or_missing_reference_fails_before_network(setup, tmp_path, ids, seconds):
    from crypto_arbitrage_bot.funding_comparison_research import research_funding_comparison

    _, settings, _, requests, _ = setup
    with pytest.raises(ValueError):
        research_funding_comparison(settings, ids, closed_at=NOW, lookback_seconds=seconds,
                                   output=tmp_path / "bad.json", clock=lambda: NOW)
    assert requests == [] and not (tmp_path / "bad.json").exists()


def test_existing_artifact_is_not_overwritten_or_requeried(setup, tmp_path):
    from crypto_arbitrage_bot.funding_comparison_research import research_funding_comparison

    _, settings, records, requests, _ = setup
    output = tmp_path / "existing.json"
    output.write_bytes(b"existing immutable evidence")
    with pytest.raises(FileExistsError):
        research_funding_comparison(settings, (records[0].id,), closed_at=NOW, lookback_seconds=(28800,),
                                   output=output, clock=lambda: NOW)
    assert requests == [] and output.read_bytes() == b"existing immutable evidence"


def test_cli_runs_one_read_only_comparison_and_preserves_database(setup, tmp_path, monkeypatch, capsys):
    from crypto_arbitrage_bot.funding_comparison_research import main

    repository, settings, records, requests, _ = setup
    text = Path("docs/research-multisettlement.toml").read_text(encoding="utf-8")
    text = text.replace('data/research/multisettlement.sqlite', repository.config.database_path.as_posix())
    config = tmp_path / "config.toml"
    config.write_text(text, encoding="utf-8")
    monkeypatch.setattr(binance_usdtm, "utc_now", lambda: datetime.now(UTC))
    monkeypatch.setattr(funding_history, "utc_now", lambda: datetime.now(UTC))
    before = repository.config.database_path.read_bytes()
    code = main(["--config", str(config), "--opportunity-ids", *(r.id for r in records),
                 "--closed-at", NOW.isoformat(), "--lookback-seconds", "28800", "86400",
                 "--output", str(tmp_path / "cli.json")])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE" and result["orders_submitted"] is False
    assert len(requests) == 3 and repository.config.database_path.read_bytes() == before
