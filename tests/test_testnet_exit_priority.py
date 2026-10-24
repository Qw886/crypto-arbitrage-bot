"""Overdue exits have priority; optional scanning must not extend failures."""

from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState


def enter(fixture, symbol="ETHUSDT"):
    bot, transport, state = fixture
    quantity = Decimal(".01") if symbol == "ETHUSDT" else Decimal(".001")
    plan = bot._plan(symbol, quantity, "priority-" + symbol)
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    return bot, transport, state, plan


def test_failed_due_exit_skips_ledger_research_and_entry(live_clock_bot, monkeypatch):
    bot, transport, state, plan = enter(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    close = Mock(side_effect=TimeoutError("temporary preflight error"))
    ledger = Mock()
    scan = Mock()
    monkeypatch.setattr(bot, "_close", close)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.reconcile_funding_ledger", ledger)
    monkeypatch.setattr(bot.research_runner, "run_once", scan)
    cycles = bot.run_once()
    assert [cycle.status for cycle in cycles] == ["EXIT_PRIORITY", "EXIT_PRIORITY"]
    assert cycles[1].pair.attempt_id == plan.attempt_id and cycles[1].planned_exit_at is not None
    close.assert_called_once()
    ledger.assert_not_called()
    scan.assert_not_called()
    assert len(transport.orders) == 2


def test_recovery_pending_skips_optional_network_work(live_clock_bot, monkeypatch):
    bot, transport, _, plan = enter(live_clock_bot)
    current = bot.journal.get(plan.attempt_id)
    bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.recover_perp_exit", Mock(return_value=current))
    scan = Mock()
    monkeypatch.setattr(bot.research_runner, "run_once", scan)
    assert all(cycle.status == "EXIT_PRIORITY" for cycle in bot.run_once())
    scan.assert_not_called()
    assert len(transport.orders) == 2


def test_network_recovery_resumes_exit_then_normal_scan(live_clock_bot, monkeypatch):
    bot, transport, state, plan = enter(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = bot._close
    calls = 0
    def close(selected):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary preflight")
        return original(selected)
    scan = Mock(return_value=())
    monkeypatch.setattr(bot, "_close", close)
    monkeypatch.setattr(bot.research_runner, "run_once", scan)
    assert all(c.status == "EXIT_PRIORITY" for c in bot.run_once())
    assert bot.run_once() == ()
    assert calls == 2 and scan.call_count == 1
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
    assert len(transport.orders) == 4


def test_deadline_crossed_during_ledger_does_not_start_scan(live_clock_bot, monkeypatch):
    bot, transport, state, _ = enter(live_clock_bot)
    def ledger(instance):
        state["now"] += timedelta(seconds=65)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.reconcile_funding_ledger", ledger)
    scan = Mock()
    monkeypatch.setattr(bot.research_runner, "run_once", scan)
    assert all(c.status == "EXIT_PRIORITY" for c in bot.run_once())
    scan.assert_not_called()
    assert len(transport.orders) == 2


def test_deadline_crossed_during_scan_exits_before_next_poll(live_clock_bot, monkeypatch):
    bot, transport, state, plan = enter(live_clock_bot)
    def scan():
        state["now"] += timedelta(seconds=65)
        return ()
    monkeypatch.setattr(bot.research_runner, "run_once", scan)
    assert bot.run_once() == ()
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
    assert len(transport.orders) == 4


def test_btc_exit_failure_does_not_prevent_due_eth_exit(live_clock_bot, monkeypatch):
    bot, transport, state, btc = enter(live_clock_bot, "BTCUSDT")
    _, _, _, eth = enter(live_clock_bot, "ETHUSDT")
    state["now"] += timedelta(seconds=65)
    original = bot._close
    def close(plan):
        if plan.symbol == "BTCUSDT":
            raise TimeoutError("BTC temporarily unreachable")
        return original(plan)
    monkeypatch.setattr(bot, "_close", close)
    assert all(c.status == "EXIT_PRIORITY" for c in bot.run_once())
    assert bot.journal.get(btc.attempt_id).state is PairExecutionState.HEDGED
    assert bot.journal.get(eth.attempt_id).state is PairExecutionState.CLOSED
    assert len(transport.orders) == 6


def test_failure_still_waits_once_per_poll_without_busy_loop(live_clock_bot, monkeypatch):
    bot, _, state, _ = enter(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    close = Mock(side_effect=TimeoutError("preflight down"))
    waits = []
    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            bot.stop.set()
            return True
        return False
    monkeypatch.setattr(bot, "_close", close)
    monkeypatch.setattr(bot.stop, "wait", wait)
    monkeypatch.setattr(bot.research_runner, "run_once", Mock(side_effect=AssertionError("must not scan")))
    bot.run_forever()
    assert close.call_count == 3
    assert waits == [bot.settings.poll_interval_seconds] * 3


def test_priority_summary_reaches_status_callback(live_clock_bot, monkeypatch):
    bot, _, state, _ = enter(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    monkeypatch.setattr(bot, "_close", Mock(side_effect=TimeoutError("preflight down")))
    callback = Mock()
    bot.status_callback = callback
    cycles = bot.run_once()
    assert [call.args[0] for call in callback.call_args_list] == list(cycles)
