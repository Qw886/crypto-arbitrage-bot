"""Offline fault injection: reduce known testnet risk without auxiliary data.

No real accounts, keys, network, thresholds, PnL or funding assumptions change.
Existing fixtures deliberately use synthetic virtual executions.
"""

import copy
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot
from test_testnet_exit_recovery import failed_exit as failed_exit
from test_unhedged_entry_recovery import unhedged as unhedged

from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.execution.exit_market_data import validate_exit_depth
from crypto_arbitrage_bot.execution.exit_recovery import recover_perp_exit
from crypto_arbitrage_bot.execution.market_data import TestnetGetRequest as GetRequest
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind as Kind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.data_quality import DataQualityReason as R
from crypto_arbitrage_bot.models.market import Market

D = Decimal
BOTH = (Market.SPOT, Market.USDTM)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline only")))


def observe(fixture, markets=BOTH):
    bot, _, state = fixture
    data = bot.collector.collect_exit_depth("ETHUSDT", markets=markets)
    return data, validate_exit_depth(data, bot.research.data_quality,
                                     evaluated_at=state["now"], markets=markets)


def opened(fixture, symbol="ETHUSDT"):
    bot, transport, state = fixture
    quantity = D(".01") if symbol == "ETHUSDT" else D(".001")
    plan = bot._plan(symbol, quantity, "depth-only-" + symbol)
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    return bot, transport, state, plan


def broken_auxiliary(bot, monkeypatch, action):
    original = bot.perp.public_payload
    seen = []
    def read(name, symbol, **kwargs):
        seen.append(name)
        if name == action:
            raise GatewayError(Kind.NETWORK_ERROR, "synthetic unavailable auxiliary")
        return original(name, symbol, **kwargs)
    monkeypatch.setattr(bot.perp, "public_payload", read)
    return seen


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("action", ["premium", "interest"])
def test_auxiliary_outage_does_not_block_known_hedged_exit(live_clock_bot, monkeypatch, symbol, action):
    bot, transport, _, plan = opened(live_clock_bot, symbol)
    seen = broken_auxiliary(bot, monkeypatch, action)
    monkeypatch.setattr(bot.collector, "collect", Mock(side_effect=AssertionError("no research snapshot on exit")))
    result = bot._close(plan)
    assert result.state is State.CLOSED and len(transport.orders) == 4
    assert seen == ["depth"] and result.perp_exit is not None
    assert transport.positions[symbol] == transport.base[symbol] == 0
    assert bot.journal.intents(plan.attempt_id)[-1].reduce_only


@pytest.mark.parametrize("action", ["premium", "interest"])
def test_entry_still_requires_complete_research_sources(live_clock_bot, monkeypatch, action):
    bot, transport, _ = live_clock_bot
    broken_auxiliary(bot, monkeypatch, action)
    with pytest.raises(ExecutionGuardError, match="INVALID_TESTNET_MARKET_DATA"):
        bot._plan("ETHUSDT", D(".01"), "auxiliary-entry-blocked")
    assert not transport.orders


def test_sources_and_decimal_are_retained_without_fabricating_snapshot(live_clock_bot):
    data, quality = observe(live_clock_bot)
    assert quality.valid and data.spot_depth.raw_response and data.perp_depth.raw_response
    assert data.spot_depth.order_book.event_time is None
    assert data.perp_depth.data.order_book.event_time is not None
    assert data.perp_depth.data.transaction_time is not None
    assert isinstance(data.spot_depth.order_book.best_bid, D)
    assert isinstance(data.perp_depth.data.order_book.best_ask, D)
    assert not hasattr(data, "snapshot") and not hasattr(data, "funding_rate")
    assert data.spot_depth.request.url.startswith("https://testnet.binance.vision/")
    assert data.perp_depth.request.url.startswith("https://demo-fapi.binance.com/")


@pytest.mark.parametrize("market", list(Market))
def test_single_market_scope_does_not_collect_the_other_market(live_clock_bot, market):
    bot, transport, _ = live_clock_bot
    start = len(transport.requests)
    data, quality = observe(live_clock_bot, (market,))
    assert quality.valid and quality.sampling_skew is None
    assert len(transport.requests[start:]) == 1
    assert (data.spot_depth is None) == (market is Market.USDTM)
    assert (data.perp_depth is None) == (market is Market.SPOT)
    assert data.book(market).market is market
    with pytest.raises(ValueError, match="not requested"):
        data.book(Market.SPOT if market is Market.USDTM else Market.USDTM)


@pytest.mark.parametrize("field,reason", [("spot_depth", R.MISSING_SPOT_DEPTH),
                                         ("perp_depth", R.MISSING_PERP_DEPTH)])
def test_missing_required_depth_is_invalid(live_clock_bot, field, reason):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    quality = validate_exit_depth(replace(data, **{field: None}), bot.research.data_quality,
                                  evaluated_at=state["now"])
    assert not quality.valid and reason in quality.reason_codes


@pytest.mark.parametrize("field", ["spot_depth", "perp_depth"])
@pytest.mark.parametrize("seconds", [10, 11])
def test_age_boundary_reuses_original_config(live_clock_bot, field, seconds):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    source = getattr(data, field)
    source = replace(source, started_at=source.started_at-timedelta(seconds=seconds))
    quality = validate_exit_depth(replace(data, **{field: source}), bot.research.data_quality,
                                  evaluated_at=state["now"])
    assert quality.valid == (seconds == 10)


@pytest.mark.parametrize("seconds", [2, 3])
def test_two_leg_skew_boundary_not_relaxed(live_clock_bot, seconds):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    source = data.spot_depth
    book = replace(source.order_book, observed_at=state["now"]-timedelta(seconds=seconds))
    source = replace(source, order_book=book, started_at=book.observed_at,
                     received_at=book.observed_at, finished_at=book.observed_at)
    quality = validate_exit_depth(replace(data, spot_depth=source), bot.research.data_quality,
                                  evaluated_at=state["now"])
    assert quality.valid == (seconds == 2)
    assert quality.sampling_skew == timedelta(seconds=seconds)


@pytest.mark.parametrize("field", ["spot_depth", "perp_depth"])
@pytest.mark.parametrize("timestamp", ["started_at", "received_at", "finished_at"])
def test_future_missing_or_naive_request_time_invalid(live_clock_bot, field, timestamp):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    for value in (None, state["now"].replace(tzinfo=None), state["now"]+timedelta(seconds=1)):
        bad = replace(data, **{field: replace(getattr(data, field), **{timestamp: value})})
        assert not validate_exit_depth(bad, bot.research.data_quality, evaluated_at=state["now"]).valid


@pytest.mark.parametrize("field", ["spot_depth", "perp_depth"])
@pytest.mark.parametrize("fault", ["production_origin", "symbol", "truncated", "http_error", "observation_time"])
def test_invalid_provenance_is_rejected(live_clock_bot, field, fault):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    source = getattr(data, field)
    endpoint = PublicEndpoint.SPOT_DEPTH if field == "spot_depth" else PublicEndpoint.PERP_DEPTH
    if fault == "production_origin":
        source = replace(source, request=prepare_get(endpoint, {"symbol": "ETHUSDT", "limit": 100}))
    elif fault == "symbol":
        source = replace(source, request=GetRequest(endpoint, (("symbol", "BTCUSDT"), ("limit", 100))))
    elif fault == "truncated":
        source = replace(source, body_truncated=True)
    elif fault == "http_error":
        source = replace(source, status_code=503)
    else:
        book = data.book(Market.SPOT if field == "spot_depth" else Market.USDTM)
        book = replace(book, observed_at=state["now"]-timedelta(seconds=1))
        source = replace(source, order_book=book) if field == "spot_depth" else replace(
            source, data=replace(source.data, order_book=book))
    assert not validate_exit_depth(replace(data, **{field: source}), bot.research.data_quality,
                                   evaluated_at=state["now"]).valid


@pytest.mark.parametrize("market", list(Market))
@pytest.mark.parametrize("fault,reason", [("price", R.INVALID_PRICE), ("quantity", R.INVALID_QUANTITY),
                                         ("empty", R.MISSING_SPOT_DEPTH), ("crossed", R.CROSSED_SPOT_BOOK)])
def test_corrupt_depth_fails_without_filling_values(live_clock_bot, market, fault, reason):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    book = data.book(market)
    level = copy.copy(book.bids[0])
    if fault in ("price", "quantity"):
        object.__setattr__(level, fault, D("0"))  # Simulate corrupted in-memory evidence, not constructor behavior.
        book = replace(book, bids=(level,))
    elif fault == "empty":
        book = replace(book, bids=())
        reason = R.MISSING_SPOT_DEPTH if market is Market.SPOT else R.MISSING_PERP_DEPTH
    else:
        object.__setattr__(level, "price", book.best_ask+D("1"))
        book = replace(book, bids=(level,))
        reason = R.CROSSED_SPOT_BOOK if market is Market.SPOT else R.CROSSED_PERP_BOOK
    bad = replace(data, spot_depth=replace(data.spot_depth, order_book=book)) if market is Market.SPOT else replace(
        data, perp_depth=replace(data.perp_depth, data=replace(data.perp_depth.data, order_book=book)))
    quality = validate_exit_depth(bad, bot.research.data_quality, evaluated_at=state["now"])
    assert reason in quality.reason_codes and not quality.valid


@pytest.mark.parametrize("market", list(Market))
def test_required_book_failure_blocks_both_orders_before_first_exit(live_clock_bot, monkeypatch, market):
    bot, transport, _, plan = opened(live_clock_bot)
    client = bot.spot if market is Market.SPOT else bot.perp
    monkeypatch.setattr(client, "public_payload", Mock(side_effect=GatewayError(Kind.NETWORK_ERROR, "offline")))
    with pytest.raises(ExecutionGuardError, match="INVALID_TESTNET_MARKET_DATA"):
        bot._close(plan)
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).state is State.HEDGED


@pytest.mark.parametrize("market", list(Market))
def test_insufficient_exit_depth_does_not_extrapolate(live_clock_bot, monkeypatch, market):
    bot, transport, _, plan = opened(live_clock_bot)
    data = bot.collector.collect_exit_depth(plan.symbol)
    book = data.book(market)
    levels = tuple(replace(level, quantity=D(".000001")) for level in (
        book.bids if market is Market.SPOT else book.asks))
    book = replace(book, bids=levels) if market is Market.SPOT else replace(book, asks=levels)
    bad = replace(data, spot_depth=replace(data.spot_depth, order_book=book)) if market is Market.SPOT else replace(
        data, perp_depth=replace(data.perp_depth, data=replace(data.perp_depth.data, order_book=book)))
    monkeypatch.setattr(bot.collector, "collect_exit_depth", lambda *args, **kwargs: bad)
    with pytest.raises(ExecutionGuardError, match="INSUFFICIENT_TESTNET_DEPTH"):
        bot._close(plan)
    assert len(transport.orders) == 2


def test_scope_cannot_hide_missing_second_leg(live_clock_bot):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot, (Market.SPOT,))
    quality = validate_exit_depth(data, bot.research.data_quality, evaluated_at=state["now"])
    assert R.SOURCE_MISMATCH in quality.reason_codes and R.MISSING_PERP_DEPTH in quality.reason_codes


def test_unhedged_spot_abort_does_not_depend_on_any_perp_public_data(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    monkeypatch.setattr(bot.perp, "public_payload", Mock(side_effect=AssertionError("no Perp quote required")))
    for _ in range(3):
        bot._managed_exits()
    result = bot.journal.get(plan.attempt_id)
    assert result.state is State.FLATTENED and transport.positions[plan.symbol] == 0
    assert transport.base[plan.symbol] == D("1") and len(transport.orders) == 2
    assert not any(i.market.value == "USDTM" for i in bot.journal.intents(plan.attempt_id))


def test_confirmed_perp_residual_does_not_depend_on_spot_depth_or_auxiliary(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    monkeypatch.setattr(bot.spot, "public_payload", Mock(side_effect=AssertionError("no Spot quote required")))
    original = bot.perp.public_payload
    def only_depth(action, symbol, **kwargs):
        assert action == "depth"
        return original(action, symbol, **kwargs)
    monkeypatch.setattr(bot.perp, "public_payload", only_depth)
    result = recover_perp_exit(bot, plan)
    assert result.state is State.CLOSED and transport.positions[plan.symbol] == 0
    assert len(transport.orders) == 5


def test_invalid_exit_diagnostics_remain_small(live_clock_bot, monkeypatch):
    bot, transport, _, plan = opened(live_clock_bot)
    monkeypatch.setattr(bot.perp, "public_payload", Mock(side_effect=GatewayError(Kind.NETWORK_ERROR, "offline")))
    with pytest.raises(ExecutionGuardError) as caught:
        bot._close(plan)
    diagnostics = json.dumps(caught.value.diagnostics, default=str)
    assert "TESTNET_EXIT_DEPTH_ONLY" in diagnostics and "NETWORK_ERROR" in diagnostics
    assert "raw_response" not in diagnostics and "bids" not in diagnostics and len(diagnostics) < 4000
    assert len(transport.orders) == 2


def test_freshness_recheck_after_price_preflight_blocks_both_exits(live_clock_bot, monkeypatch):
    bot, transport, state, plan = opened(live_clock_bot)
    original = bot._exit_limit
    def slow_quote(*args):
        price = original(*args)
        state["now"] += timedelta(seconds=11)
        return price
    monkeypatch.setattr(bot, "_exit_limit", slow_quote)
    with pytest.raises(ExecutionGuardError, match="EXIT_MARKET_EXPIRED_DURING_PREFLIGHT"):
        bot._close(plan)
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).state is State.HEDGED


@pytest.mark.parametrize("leg", [TradeLeg.SPOT_EXIT, TradeLeg.PERP_EXIT])
def test_original_vwap_limit_and_tick_math_is_unchanged(live_clock_bot, leg):
    bot, _, _ = live_clock_bot
    assembly = bot.collector.collect("ETHUSDT")
    data = bot.collector.collect_exit_depth("ETHUSDT")
    client = bot.spot if leg is TradeLeg.SPOT_EXIT else bot.perp
    rules = client.get_symbol_rules("ETHUSDT")
    assert bot._exit_limit(data, leg, D(".01"), rules) == bot._limit(assembly, leg, D(".01"), rules)


@pytest.mark.parametrize("leg", [TradeLeg.SPOT_ENTRY, TradeLeg.PERP_ENTRY])
def test_exit_only_quote_is_never_an_entry_authorization(live_clock_bot, leg):
    bot, transport, _ = live_clock_bot
    data = bot.collector.collect_exit_depth("ETHUSDT")
    client = bot.spot if leg is TradeLeg.SPOT_ENTRY else bot.perp
    with pytest.raises(ExecutionGuardError, match="CANNOT_AUTHORIZE_ENTRY"):
        bot._exit_limit(data, leg, D(".01"), client.get_symbol_rules("ETHUSDT"))
    assert not transport.orders


@pytest.mark.parametrize("field,value", [("open_order_count", 1), ("can_trade", False),
    ("one_way", False), ("single_asset", False), ("leverage", D("2")),
    ("position_quantity", D("-.02"))])
def test_original_account_exposure_guards_still_block_exits(live_clock_bot, monkeypatch, field, value):
    bot, transport, _, plan = opened(live_clock_bot)
    account = bot.perp.get_account_state(plan.symbol)
    monkeypatch.setattr(bot.perp, "get_account_state", lambda _: replace(account, **{field: value}))
    with pytest.raises(ExecutionGuardError):
        bot._close(plan)
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).state is State.HEDGED


@pytest.mark.parametrize("official", ["event_time", "transaction_time"])
def test_missing_official_perp_time_not_fabricated(live_clock_bot, official):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    depth = copy.copy(data.perp_depth.data)
    if official == "event_time":
        book = copy.copy(depth.order_book)
        object.__setattr__(book, "event_time", None)
        object.__setattr__(depth, "order_book", book)
    else:
        object.__setattr__(depth, "transaction_time", None)
    quality = validate_exit_depth(replace(data, perp_depth=replace(data.perp_depth, data=depth)),
                                  bot.research.data_quality, evaluated_at=state["now"])
    assert not quality.valid and R.MISSING_TIMESTAMP in quality.reason_codes


def test_multiple_depth_errors_are_accumulated(live_clock_bot):
    bot, _, state = live_clock_bot
    data, _ = observe(live_clock_bot)
    bad = replace(data, spot_depth=None, perp_depth=replace(
        data.perp_depth, received_at=state["now"]+timedelta(seconds=1), status_code=429))
    quality = validate_exit_depth(bad, bot.research.data_quality, evaluated_at=state["now"])
    assert {R.MISSING_SPOT_DEPTH, R.SOURCE_ERROR, R.FUTURE_TIMESTAMP}.issubset(quality.reason_codes)
