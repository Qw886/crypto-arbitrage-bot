"""Offline regression: original order GET resolves lost ACKs; no blind POST."""

from datetime import timedelta

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot


def restart(bot, transport, state):
    fresh = Bot(bot.settings, bot.spot, bot.perp, clock=lambda: state["now"])
    fresh.collector.clock = transport.clock
    return fresh


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("fresh_process", [False, True])
def test_lost_perp_exit_ack_reconciles_closed_without_new_order(live_clock_bot, monkeypatch, symbol, fresh_process):
    bot, transport, state = live_clock_bot
    plan = bot._plan(symbol, bot.research.sizing[symbol].target_quantity, "unknown-exit")
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    original = bot.perp.place_ioc_limit_order
    def lost(order):
        original(order)
        raise TimeoutError("mock ACK lost after execution")
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", lost)
    assert bot._close(plan).state is State.RECOVERY_REQUIRED
    assert bot.journal.get(plan.attempt_id).perp_exit is None
    assert len(transport.orders) == 4 and transport.positions[symbol] == 0
    if fresh_process:
        bot = restart(bot, transport, state)
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    bot._managed_exits()
    assert len(transport.orders) == 4


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_lost_perp_entry_ack_recovers_known_hedge_then_due_exit(live_clock_bot, monkeypatch, symbol):
    bot, transport, state = live_clock_bot
    plan = bot._plan(symbol, bot.research.sizing[symbol].target_quantity, "unknown-entry")
    original = bot.perp.place_ioc_limit_order
    def lost(order):
        original(order)
        raise TimeoutError("mock ACK lost after execution")
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", lost)
    assert bot.coordinator.open_pair(plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 2
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", original)
    bot = restart(bot, transport, state)
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.HEDGED
    assert len(transport.orders) == 2
    state["now"] += timedelta(seconds=65)
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    assert len(transport.orders) == 4 and transport.positions[symbol] == 0


def test_lost_spot_entry_ack_queries_original_without_guessing_second_leg(live_clock_bot, monkeypatch):
    bot, transport, _ = live_clock_bot
    symbol = "ETHUSDT"
    plan = bot._plan(symbol, bot.research.sizing[symbol].target_quantity, "unknown-spot")
    original = bot.spot.place_ioc_limit_order
    def lost(order):
        original(order)
        raise TimeoutError("mock ACK lost after execution")
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", lost)
    assert bot.coordinator.open_pair(plan).state is State.RECOVERY_REQUIRED
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).spot_entry is not None
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 1 and transport.positions[symbol] == 0
