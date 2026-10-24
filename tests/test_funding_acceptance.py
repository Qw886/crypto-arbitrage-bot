"""Offline controlled settlement acceptance, including crash and null evidence."""
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.types import FundingPayment


def arm(fixture, tmp_path):
    bot, transport, state = fixture
    state["rate"] = "0"  # Explicit acceptance bypass does not rewrite REJECT.
    acceptance = FundingAcceptance(bot, tmp_path / "acceptance.json")
    assert acceptance.tick() == "WAITING"
    return bot, transport, state, acceptance


def enter(fixture, tmp_path):
    bot, transport, state, acceptance = arm(fixture, tmp_path)
    state["now"] = state["settlement"] - timedelta(seconds=180)
    assert acceptance.tick() == "HOLDING"
    return bot, transport, state, acceptance


def close(fixture, tmp_path):
    bot, transport, state, acceptance = enter(fixture, tmp_path)
    state["now"] = state["settlement"] + timedelta(seconds=61)
    bot._managed_exits()
    with bot.journal._connect() as db:
        db.execute("UPDATE testnet_pair_attempts SET updated_at=? WHERE state='CLOSED'", (state["now"].isoformat(),))
    return bot, transport, state, acceptance


def test_wait_until_entry_has_no_orders_or_network_poll(live_clock_bot, tmp_path):
    bot, transport, state, acceptance = arm(live_clock_bot, tmp_path)
    calls = len(transport.requests)
    for _ in range(20):
        assert acceptance.tick() == "WAITING"
    assert transport.orders == {} and len(transport.requests) == calls
    assert acceptance.state["sample_type"] == "TESTNET_CONNECTIVITY_ACCEPTANCE"


def test_controlled_entry_retains_actual_reject_decision(live_clock_bot, tmp_path):
    bot, transport, state, acceptance = enter(live_clock_bot, tmp_path)
    assert len(transport.orders) == 2
    record = bot.repository.get_opportunity_by_id(acceptance.state["opportunity_id"])
    assert record.decision == "REJECT" and record.net_profit < 0
    plan = bot.journal.plans()[0][0]
    evidence = json.loads(plan.projection_json)
    assert evidence["acceptance_only"] and evidence["decision_bypassed"]
    assert plan.scheduled_exit_at == state["settlement"] + timedelta(seconds=60)


def test_missed_entry_window_no_journal_no_orders(live_clock_bot, tmp_path):
    bot, transport, state, acceptance = arm(live_clock_bot, tmp_path)
    state["now"] = state["settlement"] + timedelta(seconds=1)
    assert acceptance.tick() == "ABORTED"
    assert transport.orders == {} and bot.journal.plans() == ()


def test_restart_after_entry_never_submits_again(live_clock_bot, tmp_path):
    bot, transport, state, acceptance = enter(live_clock_bot, tmp_path)
    # Simulate a crash after POST before updating WAITING -> HOLDING report.
    acceptance._save("WAITING")
    state["now"] = state["settlement"] + timedelta(seconds=1)
    restarted = FundingAcceptance(bot, acceptance.report_path)
    assert restarted.tick() == "HOLDING"
    assert restarted.tick() == "HOLDING"
    assert len(transport.orders) == 2


@pytest.mark.parametrize("amount", ["0.12345678901234567890123456789", "0", "-0.01"])
def test_actual_payment_and_flat_accounts_complete_acceptance(live_clock_bot, tmp_path, monkeypatch, amount):
    bot, transport, state, acceptance = close(live_clock_bot, tmp_path)
    query = Mock(return_value=(FundingPayment("ledger-id", "ETHUSDT", state["settlement"], Decimal(amount)),))
    monkeypatch.setattr(bot.perp, "get_funding_payments", query)
    assert acceptance.tick() == "VERIFIED"
    assert Decimal(acceptance.state["actual_funding"]) == Decimal(amount)
    assert transport.positions["ETHUSDT"] == transport.base["ETHUSDT"] == 0
    assert len(transport.orders) == 4
    assert acceptance.tick() == "VERIFIED" and query.call_count == 1


def test_no_payment_remains_null_and_ends_unavailable(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = close(live_clock_bot, tmp_path)
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=()))
    for _ in range(5):
        result = acceptance.tick()
        state["now"] += timedelta(seconds=60)
    assert result == "UNAVAILABLE" and acceptance.state["actual_funding"] is None
    assert len(transport.orders) == 4


def test_external_position_blocks_arming(live_clock_bot, tmp_path):
    bot, transport, _ = live_clock_bot
    transport.positions["BTCUSDT"] = Decimal("-.001")
    acceptance = FundingAcceptance(bot, tmp_path / "acceptance.json")
    with pytest.raises(ExecutionGuardError, match="FLAT_DEMO"):
        acceptance.tick()
    assert transport.orders == {}


def test_interruption_before_entry_cancels_without_orders(live_clock_bot, tmp_path):
    bot, transport, _, acceptance = arm(live_clock_bot, tmp_path)
    bot.stop.set()
    assert acceptance.tick() == "CANCELLED"
    assert transport.orders == {}


def test_shutdown_run_waits_instead_of_busy_loop(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, acceptance = arm(live_clock_bot, tmp_path)
    def wait(seconds):
        assert seconds == bot.settings.poll_interval_seconds > 0
        bot.stop.set()
        return True
    monkeypatch.setattr(bot.stop, "wait", wait)
    assert acceptance.run() == "WAITING"
    assert transport.orders == {}


def test_wrong_symbol_report_cannot_resume(live_clock_bot, tmp_path):
    bot, transport, _, acceptance = arm(live_clock_bot, tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        FundingAcceptance(bot, acceptance.report_path, symbol="BTCUSDT")
    assert transport.orders == {}


def test_acceptance_config_changes_only_execution_limit_and_evidence_version():
    original = load_settings(Path("docs/testnet-funding.toml"))
    acceptance = load_settings(Path("docs/testnet-funding-acceptance.toml"))
    assert original.fees == acceptance.fees
    assert original.research.model_copy(update={"config_version": acceptance.research.config_version}) == acceptance.research
    assert original.testnet.model_copy(update={"max_price_deviation": Decimal(".001")}) == acceptance.testnet


def test_time_budget_stops_waiting_when_wall_clock_does_not_advance(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, acceptance = arm(live_clock_bot, tmp_path)
    ticks = iter((0, 99999999999999999))
    monkeypatch.setattr("crypto_arbitrage_bot.execution.funding_acceptance.monotonic_ns", lambda: next(ticks))
    assert acceptance.run() == "ABORTED"
    assert acceptance.state["reason"] == "ACCEPTANCE_TIME_BUDGET_EXHAUSTED"
    assert transport.orders == {}
