"""Offline price-ceiling regressions; synthetic PASS is not a validated strategy."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.demo import DemoResponse
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.models.market import OrderBookLevel
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

D = Decimal


def research_and_orders(fixture, symbol="ETHUSDT"):
    bot, transport, _ = fixture
    cycle = bot.research_runner._run_symbol(symbol)
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record is not None and record.decision == "PASS"
    # Obtain the ordinary bounded plan without using its Opportunity bypass as
    # evidence of strategy approval. No order is submitted by planning.
    plan = bot._plan(symbol, record.matched_quantity, "cap-plan")
    assembly = bot.collector.collect(symbol)
    orders = (plan.spot_entry, plan.perp_entry, plan.spot_exit, plan.perp_exit)
    rules = (bot.spot.get_symbol_rules(symbol), bot.perp.get_symbol_rules(symbol))
    assert transport.orders == {}
    return bot, transport, record, assembly, orders, rules


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_nonzero_tolerance_normal_pass_opens_without_spending_research_buffer(live_clock_bot, symbol):
    bot, transport, _ = live_clock_bot
    bot.config = bot.config.model_copy(update={"max_price_deviation": D(".001")})
    bot.settings = bot.settings.model_copy(update={"symbols": (symbol,)})
    bot.research_runner.settings = bot.settings
    cycles = bot.run_once()
    assert len(cycles) == 1 and cycles[0].status == "HEDGED"
    record = bot.repository.get_opportunity_by_id(cycles[0].opportunity_id)
    plan, _ = bot.journal.plans()[0]
    assert plan.spot_entry.limit_price <= record.spot_ask
    assert plan.perp_entry.limit_price >= record.perp_bid
    assert len(transport.orders) == 2
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.HEDGED
    assert plan.scheduled_exit_at is not None


@pytest.mark.parametrize("deviation", ["0", ".001"])
def test_caps_preserve_original_record_snapshot_exit_orders_and_quantity(live_clock_bot, deviation):
    bot, _, _ = live_clock_bot
    bot.config = bot.config.model_copy(update={"max_price_deviation": D(deviation)})
    bot, transport, record, assembly, orders, rules = research_and_orders(live_clock_bot)
    evidence = exact_json((record, assembly))
    capped = bot._cap_research_entry_prices(orders, assembly, record, *rules)
    assert exact_json((record, assembly)) == evidence
    assert capped[0].quantity == capped[1].quantity == record.matched_quantity
    assert capped[0].client_order_id == orders[0].client_order_id
    assert capped[1].client_order_id == orders[1].client_order_id
    assert capped[2:] == orders[2:]
    assert capped[0].limit_price <= record.spot_ask
    assert capped[1].limit_price >= record.perp_bid
    assert all(isinstance(order.limit_price, Decimal) for order in capped)
    assert transport.orders == {}
    assert bot.repository.get_opportunity_by_id(record.id) == record


@pytest.mark.parametrize("market", ["spot", "perp"])
def test_depth_outside_cap_cannot_be_counted_as_full_entry(live_clock_bot, market):
    bot, transport, record, assembly, orders, rules = research_and_orders(live_clock_bot)
    snapshot = assembly.snapshot
    quantity = record.matched_quantity
    if market == "spot":
        book = snapshot.spot_book
        bounded = replace(book, asks=(OrderBookLevel(record.spot_ask, quantity / 2),
                                     OrderBookLevel(record.spot_ask + D("1"), quantity * 10)))
        snapshot = replace(snapshot, spot_book=bounded)
    else:
        book = snapshot.perp_book
        bounded = replace(book, bids=(OrderBookLevel(record.perp_bid, quantity / 2),
                                     OrderBookLevel(record.perp_bid - D("1"), quantity * 10)))
        snapshot = replace(snapshot, perp_book=bounded)
    current = replace(assembly, snapshot=snapshot)
    with pytest.raises(ExecutionGuardError, match="INSUFFICIENT_DEPTH_WITHIN_RESEARCH_PRICE_CAP"):
        bot._cap_research_entry_prices(orders, current, record, *rules)
    assert transport.orders == {}


@pytest.mark.parametrize("market", ["spot", "perp"])
def test_price_moves_outside_recorded_bound_blocks_before_orders(live_clock_bot, market):
    bot, transport, record, assembly, orders, rules = research_and_orders(live_clock_bot)
    snapshot = assembly.snapshot
    if market == "spot":
        book = replace(snapshot.spot_book, asks=(OrderBookLevel(record.spot_ask + D("1"), D("1")),))
        snapshot = replace(snapshot, spot_book=book)
    else:
        book = replace(snapshot.perp_book, bids=(OrderBookLevel(record.perp_bid - D("1"), D("1")),))
        snapshot = replace(snapshot, perp_book=book)
    with pytest.raises(ExecutionGuardError, match="INSUFFICIENT_DEPTH_WITHIN_RESEARCH_PRICE_CAP"):
        bot._cap_research_entry_prices(orders, replace(assembly, snapshot=snapshot), record, *rules)
    assert transport.orders == {}


def test_reference_tick_rounding_never_expands_economic_bound(live_clock_bot):
    bot, transport, record, assembly, orders, rules = research_and_orders(live_clock_bot)
    tick = D(".01")
    rules = tuple(replace(rule, tick_size=tick) for rule in rules)
    record = replace(record, spot_ask=record.spot_ask + D(".005"), perp_bid=record.perp_bid + D(".005"))
    book = assembly.snapshot.perp_book
    book = replace(book, bids=tuple(replace(level, price=level.price + D(".1")) for level in book.bids),
                   asks=tuple(replace(level, price=level.price + D(".1")) for level in book.asks))
    assembly = replace(assembly, snapshot=replace(assembly.snapshot, perp_book=book))
    orders = (replace(orders[0], limit_price=orders[0].limit_price + D("1")), *orders[1:])
    capped = bot._cap_research_entry_prices(orders, assembly, record, *rules)
    assert capped[0].limit_price <= record.spot_ask
    assert capped[1].limit_price >= record.perp_bid
    assert capped[0].limit_price % tick == capped[1].limit_price % tick == 0
    assert transport.orders == {}


@pytest.mark.parametrize("changes", [
    {"symbol": "BTCUSDT"}, {"matched_quantity": D(".02")},
    {"decision": "REJECT"}, {"data_quality_status": "INVALID"},
    {"net_profit": D("0")}, {"net_return": D("0")},
    {"spot_ask": None}, {"perp_bid": None},
])
def test_invalid_or_mismatched_research_cannot_authorize_capped_orders(live_clock_bot, changes):
    bot, transport, record, assembly, orders, rules = research_and_orders(live_clock_bot)
    with pytest.raises(ExecutionGuardError):
        bot._cap_research_entry_prices(orders, assembly, replace(record, **changes), *rules)
    assert transport.orders == {}


def test_capped_depth_failure_in_normal_cycle_submits_neither_leg(live_clock_bot, monkeypatch):
    bot, transport, _ = live_clock_bot
    bot.config = bot.config.model_copy(update={"max_price_deviation": D(".001")})
    bot.settings = bot.settings.model_copy(update={"symbols": ("ETHUSDT",)})
    bot.research_runner.settings = bot.settings
    original = transport.open
    spot_reads = 0

    def changed_depth(request, **kwargs):
        nonlocal spot_reads
        response = original(request, **kwargs)
        if "/depth?" in request.full_url and "testnet.binance.vision" in request.full_url:
            spot_reads += 1
            if spot_reads >= 2:
                body = json.loads(response.body)
                price = D(body["asks"][0][0])
                body["asks"] = [[str(price), "0.001"], [str(price + 1), "1"]]
                return DemoResponse(body)
        return response

    monkeypatch.setattr(transport, "open", changed_depth)
    cycles = bot.run_once()
    assert cycles[0].status == "ENTRY_BLOCKED"
    assert cycles[0].reason == "INSUFFICIENT_DEPTH_WITHIN_RESEARCH_PRICE_CAP"
    assert transport.orders == {}
    assert bot.journal.plans() == ()


def test_explicit_connectivity_plan_without_opportunity_keeps_its_configured_limits(live_clock_bot):
    bot, transport, _ = live_clock_bot
    bot.config = bot.config.model_copy(update={"max_price_deviation": D(".001")})
    assembly = bot.collector.collect("ETHUSDT")
    plan = bot._plan("ETHUSDT", D(".01"), "explicit-connectivity")
    assert plan.spot_entry.limit_price > assembly.snapshot.spot_book.best_ask
    assert plan.perp_entry.limit_price < assembly.snapshot.perp_book.best_bid
    assert transport.orders == {}


@pytest.mark.parametrize("rate", ["0", "-.1", ".0001"])
def test_nonzero_tolerance_does_not_turn_reject_into_entry(live_clock_bot, rate):
    bot, transport, state = live_clock_bot
    bot.config = bot.config.model_copy(update={"max_price_deviation": D(".001")})
    state["rate"] = rate
    cycles = bot.run_once()
    assert all(cycle.status == "NO_ENTRY" and cycle.reason == "REJECT" for cycle in cycles)
    assert transport.orders == {}


def test_capped_entry_restarts_and_exits_once_without_mutating_research(live_clock_bot):
    bot, transport, state = live_clock_bot
    bot.config = bot.config.model_copy(update={"max_price_deviation": D(".001")})
    bot.settings = bot.settings.model_copy(update={"symbols": ("ETHUSDT",), "testnet": bot.config})
    bot.research_runner.settings = bot.settings
    cycle = bot.run_once()[0]
    assert cycle.status == "HEDGED"
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    original = exact_json(record)
    plan, _ = bot.journal.plans()[0]
    assert len(transport.orders) == 2
    state["now"] = plan.scheduled_exit_at + timedelta(seconds=1)
    restarted = Bot(bot.settings, bot.spot, bot.perp, clock=lambda: state["now"])
    restarted.collector.clock = transport.clock
    restarted._managed_exits()
    assert restarted.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
    assert len(transport.orders) == 4
    assert transport.base["ETHUSDT"] == transport.positions["ETHUSDT"] == 0
    restarted._managed_exits()
    assert len(transport.orders) == 4
    assert exact_json(restarted.repository.get_opportunity_by_id(record.id)) == original
