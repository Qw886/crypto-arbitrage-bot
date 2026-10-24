"""Single offline estimates using synthetic source observations and exit books."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.arbitrage import estimate as engine
from crypto_arbitrage_bot.arbitrage.basis import basis_pct
from crypto_arbitrage_bot.arbitrage.fees import FeeRates, FourLegFees
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.profit import (
    decomposed_net_profit,
    net_return,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.arbitrage.sizing import QuantityRules
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.estimate import (
    EstimateConfig,
    EstimateStatus,
    ExitFundingScenario,
)
from crypto_arbitrage_bot.models.market import Market, OrderBook, OrderBookLevel
from crypto_arbitrage_bot.models.snapshot import SnapshotSources
from crypto_arbitrage_bot.models.spot import SpotDepthResult
from crypto_arbitrage_bot.models.usdtm import (
    OpenInterest,
    PerpDepth,
    PremiumIndex,
    UsdtmFetchResult,
)

D = Decimal
OPENED = datetime(2026, 1, 1, 7, tzinfo=UTC)
SETTLED = datetime(2026, 1, 1, 8, tzinfo=UTC)
CLOSED = datetime(2026, 1, 1, 9, tzinfo=UTC)
RATES = FeeRates(D("0.001"), D("0.002"), D("0.0005"), D("0.0008"))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("offline estimate must not use network")),
    )


def book(market, bids, asks, symbol="BTCUSDT", observed_at=OPENED):
    return OrderBook(
        symbol, market,
        tuple(OrderBookLevel(D(price), D(quantity)) for price, quantity in bids),
        tuple(OrderBookLevel(D(price), D(quantity)) for price, quantity in asks),
        observed_at, event_time=observed_at if market is Market.USDTM else None,
    )


def books(symbol="BTCUSDT", recurring=False):
    spot = book(Market.SPOT, (("99", "3"),), (("101", "1"), ("102" if recurring else "103", "2")), symbol)
    perp = book(Market.USDTM, (("109", "1"), ("108" if recurring else "107", "2")), (("110" if recurring else "111", "3"),), symbol)
    if recurring:
        return spot, replace(spot, observed_at=CLOSED), perp, replace(perp, observed_at=CLOSED)
    return (
        spot,
        book(Market.SPOT, (("103", "1"), ("101", "2")), (("105", "3"),), symbol, CLOSED),
        perp,
        book(Market.USDTM, (("107", "3"),), (("109", "1"), ("111", "2")), symbol, CLOSED),
    )


def assembly_for(entry_books, indicative_rate="0.01"):
    spot, perp = entry_books
    symbol = spot.symbol
    provenance = dict(
        started_at=OPENED - timedelta(milliseconds=10), finished_at=OPENED,
        received_at=OPENED, elapsed_ms=10, status_code=200,
        response_headers=(), raw_response=b'{"synthetic":true}', body_truncated=False,
        error=None,
    )
    sources = SnapshotSources(
        spot_depth=SpotDepthResult(
            request=prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": symbol, "limit": 100}),
            order_book=spot, last_update_id=1, **provenance,
        ),
        perp_depth=UsdtmFetchResult(
            request=prepare_get(PublicEndpoint.PERP_DEPTH, {"symbol": symbol, "limit": 100}),
            data=PerpDepth(perp, 2, OPENED), **provenance,
        ),
        premium_index=UsdtmFetchResult(
            request=prepare_get(PublicEndpoint.PERP_PREMIUM_INDEX, {"symbol": symbol}),
            data=PremiumIndex(symbol, D("110"), D("100"), D(indicative_rate), SETTLED, OPENED, OPENED),
            **provenance,
        ),
        open_interest=UsdtmFetchResult(
            request=prepare_get(PublicEndpoint.PERP_OPEN_INTEREST, {"symbol": symbol}),
            data=OpenInterest(symbol, D("123.456"), OPENED, OPENED), **provenance,
        ),
    )
    return assemble_snapshot(symbol, sources, assembled_at=OPENED)


def config(**changes):
    values = dict(
        spot_rules=QuantityRules(D("0.1")), perp_rules=QuantityRules(D("0.1")),
        fee_rates=RATES, target_quantity=D("2"), other_costs=D("0.1"),
    )
    values.update(changes)
    return EstimateConfig(**values)


def scenario_for(all_books, rate="0.01", closed_at=CLOSED, no_events=False):
    _, spot_exit, _, perp_exit = all_books
    events = () if no_events else (FundingEvent(
        event_id="synthetic-settlement", settled_at=SETTLED,
        settlement_mark_price=D("110"), funding_rate=D(rate), symbol=spot_exit.symbol,
    ),)
    return ExitFundingScenario(
        spot_exit, perp_exit, OPENED, closed_at, events,
        "Synthetic exit depth; explicit settlement events only; fees charged in quote asset.",
    )


def estimate(all_books=None, **config_changes):
    all_books = books() if all_books is None else all_books
    return engine.estimate_once(
        assembly_for((all_books[0], all_books[2])), config(**config_changes), scenario_for(all_books),
    )


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_complete_estimate_matches_full_hand_calculation(symbol):
    """q=2: -4 + 2.2 - (.204+.408+.108+.176) - .1 = -2.796 USDT."""
    source_books = books(symbol)
    assembly = assembly_for((source_books[0], source_books[2]))
    settings, scenario = config(), scenario_for(source_books)
    result = engine.estimate_once(assembly, settings, scenario)
    assert result.complete and result.status is EstimateStatus.ESTIMATED
    assert result.assembly is assembly and result.config is settings and result.scenario is scenario
    assert result.symbol == symbol
    assert result.matched_quantity == D("2") and result.residual_delta == D("0")
    assert tuple(fill.quote_notional for fill in result.fills) == (D("204"), D("204"), D("216"), D("220"))
    assert result.spot_entry_vwap == D("102") and result.perp_entry_vwap == D("108")
    assert result.executable_entry_basis == D("8")
    assert result.basis_pct == basis_pct(D("8"), D("101"))
    assert result.entry_vwap_basis_pct == basis_pct(D("6"), D("102"))
    assert result.funding_rate == D("0.01") and result.estimated_funding == D("2.2")
    assert result.fees == FourLegFees(D(".204"), D(".408"), D(".108"), D(".176"))
    assert result.mid_basis_pnl == D("12") and result.price_pnl == D("-4")
    assert result.decomposition.spread_costs == result.decomposition.slippage_costs == D("8")
    assert result.net_profit == D("-2.796")
    assert result.capital_occupied == D("420.996")
    assert result.net_return == net_return(D("-2.796"), D("420.996"))
    assert result.net_profit == decomposed_net_profit(result.decomposition, D("2.2"), result.fees, D(".1"))
    assert result.calculation == round_trip_from_fills(*result.fills, result.fees, D("2.2"), D(".1"))
    assert all(isinstance(value, Decimal) for value in (
        result.matched_quantity, result.residual_delta, result.spot_entry_vwap,
        result.perp_entry_vwap, result.executable_entry_basis, result.basis_pct,
        result.entry_vwap_basis_pct, result.funding_rate, result.estimated_funding,
        result.price_pnl, result.net_profit, result.capital_occupied, result.net_return,
        result.fees.spot_entry, result.fees.spot_exit, result.fees.perp_entry, result.fees.perp_exit,
    ))


@pytest.mark.parametrize("missing", ["spot_depth", "perp_depth", "premium_index", "open_interest"])
def test_incomplete_snapshot_stops_before_math(missing, monkeypatch):
    source_books = books()
    complete = assembly_for((source_books[0], source_books[2]))
    incomplete = assemble_snapshot(
        complete.symbol, replace(complete.sources, **{missing: None}), assembled_at=OPENED,
    )
    forbidden = Mock(side_effect=AssertionError("incomplete input must not reach math engine"))
    for name in ("size_position", "simulate_fill", "accrue_short_funding", "round_trip_from_fills", "decompose_fills"):
        monkeypatch.setattr(engine, name, forbidden)
    result = engine.estimate_once(incomplete, config(), scenario_for(source_books))
    assert result.status is EstimateStatus.INCOMPLETE_SNAPSHOT and not result.complete
    assert result.fills == () and result.sizing is None
    assert result.calculation is result.decomposition is result.fees is None
    assert result.net_profit is result.net_return is None
    forbidden.assert_not_called()


@pytest.mark.parametrize("leg_index,side", [(0, "asks"), (1, "bids"), (2, "bids"), (3, "asks")])
def test_any_insufficient_leg_cannot_produce_profit(leg_index, side):
    source_books = list(books())
    source_books[leg_index] = replace(source_books[leg_index], **{side: getattr(source_books[leg_index], side)[:1]})
    result = estimate(source_books)
    assert result.status is EstimateStatus.INSUFFICIENT_DEPTH and not result.complete
    assert result.fills[leg_index].filled_quantity == D("1")
    assert result.fills[leg_index].vwap is None
    assert result.calculation is result.decomposition is result.fees is None
    assert result.net_profit is result.net_return is None


@pytest.mark.parametrize(
    "rate,closed_at,no_events,funding,profit",
    [
        ("0.01", CLOSED, False, "3.3", "-11.1255"),
        ("-0.01", CLOSED, False, "-3.3", "-17.7255"),
        ("0", CLOSED, False, "0", "-14.4255"),
        ("0.01", SETTLED - timedelta(seconds=1), False, "0", "-14.4255"),
        ("0.01", CLOSED, True, "0", "-14.4255"),
    ],
    ids=["positive", "negative", "zero", "settlement_not_crossed", "no_events"],
)
def test_recurring_vwap_reconciles_and_unchanged_basis_has_no_convergence_profit(
    rate, closed_at, no_events, funding, profit,
):
    source_books = books(recurring=True)
    result = engine.estimate_once(
        assembly_for((source_books[0], source_books[2])), config(target_quantity=D("3")),
        scenario_for(source_books, rate, closed_at, no_events),
    )
    assert result.complete
    assert result.spot_entry_vwap == D("101." + "6" * 46 + "7")
    assert result.perp_entry_vwap == D("108." + "3" * 47)
    assert result.funding_rate == D("0.01")  # observation never replaces settlement assumptions
    assert result.estimated_funding == D(funding)
    assert result.mid_basis_pnl == 0 and result.price_pnl == D("-13")
    assert result.decomposition.spread_costs == D("9")
    assert result.decomposition.slippage_costs == D("4")
    assert result.fees == FourLegFees(D(".305"), D(".594"), D(".1625"), D(".264"))
    assert result.net_profit == D(profit)
    assert result.net_profit == decomposed_net_profit(result.decomposition, D(funding), result.fees, D(".1"))
    assert result.calculation == round_trip_from_fills(*result.fills, result.fees, D(funding), D(".1"))


def test_spread_slippage_and_each_fee_are_charged_once():
    charged = estimate()
    zero_fees = FeeRates(*(D("0") for _ in range(4)))
    free = estimate(fee_rates=zero_fees)
    assert free.net_profit - charged.net_profit == charged.fees.total == D(".896")
    assert charged.net_profit == D("-2.796")  # additional spread/slippage subtraction would be -18.796


def test_decimal_calculation_does_not_use_or_mutate_callers_context():
    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        previous_flags = context.flags.copy()
        result = estimate(books(recurring=True), target_quantity=D("3"))
        assert context.prec == 4 and context.rounding == ROUND_DOWN
        assert context.traps[Inexact] and context.flags == previous_flags
    assert result.net_profit == D("-11.1255")


def test_quantity_is_floored_to_common_lot_and_remains_matched():
    result = estimate(
        target_quantity=D("2.19"), spot_rules=QuantityRules(D(".2")), perp_rules=QuantityRules(D(".3")),
    )
    assert result.complete and result.matched_quantity == D("1.8")
    assert result.residual_delta == D("0")
    assert all(fill.filled_quantity == D("1.8") for fill in result.fills)


def test_quantity_below_minimum_is_rejected_before_fills():
    result = estimate(target_quantity=D(".1"), spot_rules=QuantityRules(D(".1"), min_quantity=D("1")))
    assert result.status is EstimateStatus.SIZING_REJECTED
    assert result.fills == () and result.net_profit is result.net_return is None


def test_exit_fill_minimum_notional_is_checked_at_actual_prices():
    result = estimate(books(recurring=True), spot_rules=QuantityRules(D(".1"), min_notional=D("200")))
    assert result.status is EstimateStatus.SIZING_REJECTED
    assert "SPOT_EXIT_MIN_NOTIONAL_AT_FILL" in result.issues
    assert result.net_profit is result.net_return is None


@pytest.mark.parametrize("budget,status", [("204", EstimateStatus.ESTIMATED), ("202", EstimateStatus.BUDGET_EXCEEDED)])
def test_spot_budget_uses_actual_notional_including_book_slippage(budget, status):
    result = estimate(target_quantity=None, spot_budget=D(budget))
    assert result.status is status
    assert result.fills[0].quote_notional == D("204")
    if status is EstimateStatus.BUDGET_EXCEEDED:
        assert result.net_profit is result.net_return is None
        assert result.issues == ("SPOT_BUDGET_EXCEEDED",)


@pytest.mark.parametrize("limit,status", [("420.996", EstimateStatus.ESTIMATED), ("420.995", EstimateStatus.BUDGET_EXCEEDED)])
def test_total_capital_includes_full_perp_reserve_fees_and_other_costs(limit, status):
    result = estimate(capital_limit=D(limit))
    assert result.status is status and result.capital_occupied == D("420.996")
    if status is EstimateStatus.BUDGET_EXCEEDED:
        assert result.net_profit is result.net_return is None


def test_stress_reserve_changes_capital_but_is_not_a_cash_cost():
    result = estimate(stress_reserve=D("10"))
    assert result.capital_occupied == D("430.996")
    assert result.net_profit == D("-2.796")


@pytest.mark.parametrize("field", ["target_quantity", "spot_budget", "capital_limit", "stress_reserve", "other_costs"])
def test_config_rejects_float(field):
    changes = {field: 0.1}
    if field == "spot_budget":
        changes["target_quantity"] = None
    with pytest.raises(TypeError):
        config(**changes)


@pytest.mark.parametrize("changes", [{"target_quantity": None}, {"spot_budget": D("202")}])
def test_config_requires_exactly_one_sizing_input(changes):
    with pytest.raises(ValueError, match="exactly one"):
        config(**changes)


def test_exit_symbol_must_match_snapshot():
    source_books = books()
    with pytest.raises(ValueError, match="same symbol"):
        engine.estimate_once(assembly_for((source_books[0], source_books[2])), config(), scenario_for(books("ETHUSDT")))


def test_exit_markets_cannot_be_swapped():
    scenario = scenario_for(books())
    with pytest.raises(ValueError, match="Spot and USDT-M"):
        replace(scenario, spot_exit_book=scenario.perp_exit_book, perp_exit_book=scenario.spot_exit_book)


def test_scenario_rejects_reversed_holding_window():
    with pytest.raises(ValueError, match="must not precede"):
        scenario_for(books(), closed_at=OPENED - timedelta(seconds=1))
