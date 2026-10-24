"""Acceptance failures retain safe causes without submitting an order."""

import json
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.types import ExecutionMarket


def ready(fixture, tmp_path):
    bot, transport, state = fixture
    acceptance = FundingAcceptance(bot, tmp_path / "acceptance.json")
    assert acceptance.tick() == "WAITING"
    state["now"] = state["settlement"] - timedelta(seconds=180)
    return bot, transport, state, acceptance


@pytest.mark.parametrize("code", ["ACCOUNT_NOT_READY", "INSUFFICIENT_TESTNET_BALANCE",
                                 "MARKET_EXPIRED_DURING_PREFLIGHT"])
def test_preflight_guard_preserves_reason_stage_and_research(live_clock_bot, tmp_path, monkeypatch, code):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=ExecutionGuardError(code)))
    assert acceptance.run() == "ABORTED"
    report = json.loads(acceptance.report_path.read_text())
    assert report["reason"] == code
    assert report["failure_stage"] == "ENTRY_PREFLIGHT"
    assert report["error_type"] == "ExecutionGuardError"
    assert bot.repository.get_opportunity_by_id(report["opportunity_id"]) is not None
    assert transport.orders == {} and bot.journal.plans() == ()


def test_api_error_preserves_code_without_signed_url(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    error = GatewayError(ExecutionErrorKind.HTTP_REJECTED, "private-body-not-for-report",
                                 market=ExecutionMarket.USDTM, http_status=400, api_code=-1021)
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=error))
    assert acceptance.run() == "ABORTED"
    report = json.loads(acceptance.report_path.read_text())
    assert report["reason"] == "HTTP_REJECTED"
    assert report["diagnostics"]["api_code"] == -1021
    assert report["diagnostics"]["http_status"] == 400
    assert "private-body-not-for-report" not in acceptance.report_path.read_text()
    assert transport.orders == {}


def test_unknown_error_is_redacted(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=RuntimeError("signed?signature=SECRET")))
    assert acceptance.run() == "ABORTED"
    assert acceptance.state["reason"] == "RuntimeError"
    assert "SECRET" not in acceptance.report_path.read_text()
    assert transport.orders == {}


def test_invalid_quote_reports_compact_quality_provenance(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = ready(live_clock_bot, tmp_path)
    assembly = bot.collector.collect("ETHUSDT")
    source = assembly.sources.spot_depth
    stale = replace(source, started_at=state["now"] - timedelta(seconds=21),
                    finished_at=state["now"] - timedelta(seconds=20),
                    received_at=state["now"] - timedelta(seconds=20),
                    raw_response=b"secret-raw-orderbook" * 10000)
    invalid = replace(assembly, sources=replace(assembly.sources, spot_depth=stale))
    # Research has already saved a genuine offline record; fail only the fresh preflight.
    cycle = bot.research_runner._run_symbol("ETHUSDT")
    monkeypatch.setattr(bot.research_runner, "_run_symbol", Mock(return_value=cycle))
    monkeypatch.setattr(bot.collector, "collect", Mock(return_value=invalid))
    assert acceptance.run() == "ABORTED"
    report = json.loads(acceptance.report_path.read_text())
    assert report["reason"] == "INVALID_TESTNET_MARKET_DATA"
    assert "STALE_SPOT_DATA" in report["diagnostics"]["quality_reason_codes"]
    assert "spot_depth" in report["diagnostics"]["sources"]
    text = acceptance.report_path.read_text()
    assert "raw_response" not in text and "secret-raw-orderbook" not in text
    assert len(text) < 6000
    assert transport.orders == {} and bot.journal.plans() == ()


def test_valid_quality_remains_unchanged(live_clock_bot):
    bot, transport, _ = live_clock_bot
    assembly = bot.collector.collect("ETHUSDT")
    before = repr(assembly)
    bot._require_quality(assembly, "INVALID_TESTNET_MARKET_DATA")
    assert repr(assembly) == before and transport.orders == {}


def test_guard_reason_is_bounded():
    assert len(str(ExecutionGuardError("X" * 1000))) == 240
