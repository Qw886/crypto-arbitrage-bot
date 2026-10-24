"""Offline bounded loop acceptance: real bot, fake exchange, no network."""

import importlib.util
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationCode,
    ReconciliationIssue,
    ReconciliationStatus,
    TradeReconciliation,
)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("work").resolve()))
    spec = importlib.util.spec_from_file_location("continuous_acceptance", Path("work/testnet_continuous_acceptance.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(harness, live_clock_bot, monkeypatch):
    bot, transport, state = live_clock_bot
    state["rate"] = "0"  # Ordinary research remains REJECT; no threshold change.
    evidence = {}
    class FakeTimer:
        def __init__(self, seconds, stop):
            evidence["seconds"] = seconds
            self.stop = stop
        def start(self):
            evidence["timer_started"] = True
        def cancel(self):
            evidence["timer_cancelled"] = True
    monkeypatch.setattr(harness, "Timer", FakeTimer)
    def loop():
        bot.run_once()
        state["now"] += timedelta(seconds=65)
        bot.run_once()
        bot.stop.set()
    monkeypatch.setattr(bot, "run_forever", loop)
    def reconcile(*args):
        return TradeReconciliation(args[1], "ETHUSDT", state["now"],
            status=ReconciliationStatus.INCOMPLETE,
            issues=(ReconciliationIssue(ReconciliationCode.FUNDING_NOT_CONFIRMED),),
            price_pnl=Decimal("-.02"), fees=FourLegFees(*([Decimal("0")] * 4)),
            exit_time=state["now"])
    monkeypatch.setattr(harness, "reconcile_closed_attempt", reconcile)
    return bot, transport, state, evidence


def test_bounded_normal_runner_exits_one_pair_and_restores_configuration(harness, live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, audit = prepare(harness, live_clock_bot, monkeypatch)
    original = bot.config
    path = tmp_path / "continuous.json"
    assert harness.run_acceptance(bot, path, duration_seconds=600) == 0
    report = json.loads(path.read_text())
    assert report["status"] == "PASS" and report["rounds"] == 2
    assert report["new_entry_limit"] == 1 and report["no_new_entry"]
    assert report["no_open_orders"] and report["inventory_restored"]
    assert report["funding_total"] is report["net_profit"] is None
    assert report["graceful_loop_stop"] and report["exit_timeliness"] == "PASS"
    assert bot.config is original and bot.stop.is_set()
    assert len(transport.orders) == 4 and len(bot.journal.plans()) == 1
    assert audit == {"seconds": 600, "timer_started": True, "timer_cancelled": True}


def test_pass_research_does_not_create_additional_future_holding(harness, live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, _ = prepare(harness, live_clock_bot, monkeypatch)
    state["rate"] = ".1"
    assert harness.run_acceptance(bot, tmp_path / "continuous.json") == 0
    assert len(bot.journal.plans()) == 1 and len(transport.orders) == 4
    records = bot.repository.list_recent_opportunities()
    assert any(r.decision == "PASS" for r in records)  # Research output is unchanged.


@pytest.mark.parametrize("duration", [0, 119, 1801, True])
def test_duration_bounds_block_entry(harness, live_clock_bot, tmp_path, duration):
    bot, transport, _ = live_clock_bot
    with pytest.raises(ValueError):
        harness.run_acceptance(bot, tmp_path / "continuous.json", duration_seconds=duration)
    assert transport.orders == {}


def test_existing_report_prevents_duplicate_entry(harness, live_clock_bot, tmp_path):
    bot, transport, _ = live_clock_bot
    path = tmp_path / "continuous.json"
    path.write_text("{}")
    with pytest.raises(ExecutionGuardError):
        harness.run_acceptance(bot, path)
    assert transport.orders == {}


def test_late_exit_cannot_pass_continuous_acceptance(harness, live_clock_bot, tmp_path, monkeypatch):
    bot, _, _, _ = prepare(harness, live_clock_bot, monkeypatch)
    original = harness.reconcile_closed_attempt
    monkeypatch.setattr(harness, "reconcile_closed_attempt", lambda *args:
        replace(original(*args), exit_time=bot.journal.plans()[0][1] + timedelta(seconds=95)))
    path = tmp_path / "continuous.json"
    assert harness.run_acceptance(bot, path) == 1
    report = json.loads(path.read_text())
    assert report["execution_status"] == "PASS" and report["exit_timeliness"] == "FAIL"
    assert report["status"] == "NEEDS_ATTENTION" and report["exit_lateness_seconds"] == "35"


def test_accounting_failure_preserves_closed_pair_and_stops(harness, live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, audit = prepare(harness, live_clock_bot, monkeypatch)
    monkeypatch.setattr(harness, "reconcile_closed_attempt", Mock(side_effect=TimeoutError("private details")))
    path = tmp_path / "continuous.json"
    assert harness.run_acceptance(bot, path) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "NEEDS_ATTENTION" and report["error_type"] == "TimeoutError"
    assert "private details" not in path.read_text()
    assert len(transport.orders) == 4 and bot.journal.get(report["attempt_id"]).state.value == "CLOSED"
    assert bot.stop.is_set() and audit["timer_cancelled"]


def test_bounded_drain_manages_position_after_loop_stop(harness, live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, _ = prepare(harness, live_clock_bot, monkeypatch)
    def stop_before_exit():
        state["now"] += timedelta(seconds=65)
        bot.stop.set()
    monkeypatch.setattr(bot, "run_forever", stop_before_exit)
    assert harness.run_acceptance(bot, tmp_path / "continuous.json") == 0
    assert len(transport.orders) == 4


def test_main_order_gate_blocks_bot_creation(harness, tmp_path, monkeypatch):
    create = Mock()
    monkeypatch.setattr(harness, "create_testnet_bot", create)
    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    monkeypatch.setattr(harness, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr("sys.argv", ["tool", "--report", str(tmp_path / "disabled.json")])
    assert harness.main() == 1
    create.assert_not_called()
