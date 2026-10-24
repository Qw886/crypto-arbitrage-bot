"""Offline safety and process-handoff tests for the immediate testnet test."""

import importlib.util
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationCode,
    ReconciliationIssue,
    ReconciliationStatus,
    TradeReconciliation,
)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("work").resolve()))
    spec = importlib.util.spec_from_file_location("short_restart_acceptance", Path("work/short_restart_acceptance.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_immediate_entry_and_fresh_bot_exit_keep_same_two_entry_orders(harness, live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state = live_clock_bot
    path = tmp_path / "short.json"
    assert harness.first_process(bot, path) == 0
    data = json.loads(path.read_text())
    assert data["status"] == "HEDGED_PROCESS_EXITING"
    assert data["funding_settlement_test"] is False
    assert len(transport.orders) == 2
    plan, _ = bot.journal.plans()[0]
    assert plan.funding_settlement_at is plan.scheduled_exit_at is None
    first_ids = list(transport.orders)
    new_bot = Bot(bot.settings, bot.spot, bot.perp, clock=lambda: state["now"])
    new_bot.collector.clock = transport.clock
    state["now"] += timedelta(seconds=65)
    monkeypatch.setattr(harness, "os", SimpleNamespace(getpid=lambda: data["first_pid"] + 1))
    accounting = TradeReconciliation(plan.attempt_id, plan.symbol, state["now"],
        price_pnl=Decimal("-.02"), fees=FourLegFees(*([Decimal("0")] * 4)),
        status=ReconciliationStatus.INCOMPLETE,
        issues=(ReconciliationIssue(ReconciliationCode.FUNDING_NOT_CONFIRMED),),
        calculation=None, funding_total=None, exit_time=state["now"])
    monkeypatch.setattr(harness, "reconcile_closed_attempt", lambda *args: accounting)
    assert harness.resumed_process(new_bot, path) == 0
    final = json.loads(path.read_text())
    assert final["status"] == "PASS"
    assert len(transport.orders) == 4 and list(transport.orders)[:2] == first_ids
    assert transport.positions["ETHUSDT"] == transport.base["ETHUSDT"] == 0
    assert final["net_profit"] is final["funding_total"] is None
    assert final["accounting_status"] == "INCOMPLETE"


def test_existing_report_cannot_create_duplicate_entry(harness, live_clock_bot, tmp_path):
    bot, transport, _ = live_clock_bot
    path = tmp_path / "short.json"
    path.write_text("{}")
    with pytest.raises(ExecutionGuardError, match="SHORT_ACCEPTANCE_REPORT_ALREADY_EXISTS"):
        harness.first_process(bot, path)
    assert transport.orders == {}


def test_existing_active_pair_cannot_start_another_sample(harness, live_clock_bot, tmp_path):
    bot, transport, _ = live_clock_bot
    assert harness.first_process(bot, tmp_path / "one.json") == 0
    with pytest.raises(ExecutionGuardError, match="SHORT_ACCEPTANCE_REQUIRES_NO_MANAGED_POSITION"):
        harness.first_process(bot, tmp_path / "two.json")
    assert len(transport.orders) == 2


def test_unmanaged_position_prevents_any_order(harness, live_clock_bot, tmp_path):
    bot, transport, _ = live_clock_bot
    transport.positions["BTCUSDT"] = Decimal("-.001")
    with pytest.raises(ExecutionGuardError, match="SHORT_ACCEPTANCE_REQUIRES_FLAT_DEMO_ACCOUNT"):
        harness.first_process(bot, tmp_path / "short.json")
    assert transport.orders == {}


@pytest.mark.parametrize("change", ["same_process", "symbol", "journal_path", "entry_ids"])
def test_bad_restart_identity_cannot_manage_or_submit(harness, live_clock_bot, tmp_path, change):
    bot, transport, _ = live_clock_bot
    path = tmp_path / "short.json"
    assert harness.first_process(bot, path) == 0
    data = json.loads(path.read_text())
    if change == "symbol":
        data["symbol"] = "BTCUSDT"
    elif change == "journal_path":
        data["journal_path"] = "other.sqlite"
    elif change == "entry_ids":
        data["entry_ids"] = ["other", "orders"]
        data["first_pid"] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ExecutionGuardError, match="SHORT_RESTART"):
        harness.resumed_process(bot, path)
    assert len(transport.orders) == 2


def test_failed_first_child_is_not_restarted(harness, tmp_path, monkeypatch):
    child = Mock(return_value=SimpleNamespace(returncode=1))
    monkeypatch.setattr(harness.subprocess, "run", child)
    args = SimpleNamespace(report=tmp_path / "short.json", config=Path("docs/testnet-funding-acceptance.toml"),
                           max_exit_lateness_seconds=Decimal("30"))
    assert harness.supervise(args) == 1
    assert child.call_count == 1


def test_unchanged_acceptance_config_uses_existing_sixty_second_exit():
    from crypto_arbitrage_bot.config.settings import load_settings
    config = load_settings(Path("docs/testnet-funding-acceptance.toml"))
    assert config.testnet.holding_seconds == 60
    assert config.research.sizing["ETHUSDT"].target_quantity == Decimal("0.01")


@pytest.mark.parametrize("delay,expected,passed", [
    (-1, "0", True), (0, "0", True), (30, "30", True),
    (31, "31", False), (180, "180", False),
])
def test_exit_lateness_threshold_does_not_call_late_close_success(harness, delay, expected, passed):
    from datetime import UTC, datetime
    due = datetime(2026, 10, 4, 12, tzinfo=UTC)
    late, on_time = harness.exit_timing(due, due + timedelta(seconds=delay), Decimal("30"))
    assert late == Decimal(expected) and isinstance(late, Decimal)
    assert on_time is passed


def test_missing_official_exit_timestamp_is_not_timely_success(harness):
    from datetime import UTC, datetime
    assert harness.exit_timing(datetime(2026, 10, 4, tzinfo=UTC), None, Decimal("30")) == (None, False)


def test_subsecond_exit_timing_keeps_exact_decimal(harness):
    from datetime import UTC, datetime
    due = datetime(2026, 10, 4, tzinfo=UTC)
    late, on_time = harness.exit_timing(due, due + timedelta(microseconds=30000001), Decimal("30"))
    assert late == Decimal("30.000001") and not on_time


@pytest.mark.parametrize("tolerance", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_tolerance_blocks_entry_before_any_orders(harness, live_clock_bot, tmp_path, tolerance):
    bot, transport, _ = live_clock_bot
    with pytest.raises(ValueError, match="nonnegative Decimal"):
        harness.first_process(bot, tmp_path / "short.json", max_exit_lateness_seconds=tolerance)
    assert transport.orders == {}
