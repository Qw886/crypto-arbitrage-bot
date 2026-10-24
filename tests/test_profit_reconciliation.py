"""Offline regression of exact fill cash flow against cost attribution.

All books, fees and settlement events below are synthetic. Expected amounts
are hand-computed from execution slices, independently of rounded VWAPs.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.arbitrage.fees import (
    FeeRates,
    FourLegFees,
    calculate_fee,
    four_leg_fees_for_quantity,
)
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent, accrue_short_funding
from crypto_arbitrage_bot.arbitrage.profit import (
    decompose_fills,
    decomposed_net_profit,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.models.calculation import SimulatedFill, TradeLeg
from crypto_arbitrage_bot.models.market import BidAsk, Market, OrderBook, OrderBookLevel

D = Decimal
OPENED = datetime(2026, 1, 1, 7, tzinfo=UTC)
SETTLED = datetime(2026, 1, 1, 8, tzinfo=UTC)
CLOSED = datetime(2026, 1, 1, 9, tzinfo=UTC)
LEGS = (
    TradeLeg.SPOT_ENTRY,
    TradeLeg.SPOT_EXIT,
    TradeLeg.PERP_ENTRY,
    TradeLeg.PERP_EXIT,
)
QUOTE_NAMES = (
    "spot_entry_quote", "spot_exit_quote", "perp_entry_quote", "perp_exit_quote",
)
RATES = FeeRates(D("0.001"), D("0.002"), D("0.0005"), D("0.0008"))
ZERO_FEES = FourLegFees(D("0"), D("0"), D("0"), D("0"))


def make_book(
    market: Market,
    bids: tuple[tuple[str, str], ...],
    asks: tuple[tuple[str, str], ...],
    *,
    symbol: str = "BTCUSDT",
    observed_at: datetime = OPENED,
) -> OrderBook:
    return OrderBook(
        symbol=symbol,
        market=market,
        bids=tuple(OrderBookLevel(D(price), D(quantity)) for price, quantity in bids),
        asks=tuple(OrderBookLevel(D(price), D(quantity)) for price, quantity in asks),
        observed_at=observed_at,
    )


def hand_calculation_books(symbol: str = "BTCUSDT") -> tuple[OrderBook, ...]:
    return (
        make_book(Market.SPOT, (("99", "3"),), (("101", "1"), ("103", "2")), symbol=symbol),
        make_book(
            Market.SPOT, (("103", "1"), ("101", "2")), (("105", "3"),),
            symbol=symbol, observed_at=CLOSED,
        ),
        make_book(Market.USDTM, (("109", "1"), ("107", "2")), (("111", "3"),), symbol=symbol),
        make_book(
            Market.USDTM, (("107", "3"),), (("109", "1"), ("111", "2")),
            symbol=symbol, observed_at=CLOSED,
        ),
    )


def recurring_vwap_books() -> tuple[OrderBook, ...]:
    spot = make_book(Market.SPOT, (("99", "3"),), (("101", "1"), ("102", "2")))
    perp = make_book(Market.USDTM, (("109", "1"), ("108", "2")), (("110", "3"),))
    # No change in either market's prices/depth across the holding window.
    return spot, replace(spot, observed_at=CLOSED), perp, replace(perp, observed_at=CLOSED)


def fills_and_quotes(
    books: tuple[OrderBook, ...], quantity: Decimal,
) -> tuple[tuple[SimulatedFill, ...], dict[str, BidAsk]]:
    fills = tuple(simulate_fill(book, leg, quantity) for book, leg in zip(books, LEGS, strict=True))
    quotes = {
        name: BidAsk(book.bids[0].price, book.asks[0].price)
        for name, book in zip(QUOTE_NAMES, books, strict=True)
    }
    return fills, quotes


def event_funding(
    quantity: Decimal, rate: str, *, symbol: str = "BTCUSDT", closed_at: datetime = CLOSED,
) -> Decimal:
    event = FundingEvent(
        event_id="synthetic-settlement-1", settled_at=SETTLED,
        settlement_mark_price=D("110"), funding_rate=D(rate), symbol=symbol,
    )
    return accrue_short_funding(quantity, OPENED, closed_at, (event,), symbol)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_hand_calculated_four_leg_net_profit_reconciles_exactly(symbol: str) -> None:
    """q=2: net = -4 + 2.2 - (.204+.408+.108+.176) - .1 = -2.796."""
    fills, quotes = fills_and_quotes(hand_calculation_books(symbol), D("2"))
    assert tuple(fill.quote_notional for fill in fills) == (D("204"), D("204"), D("216"), D("220"))
    assert tuple(fill.vwap for fill in fills) == (D("102"), D("102"), D("108"), D("110"))
    # Each fill consumes one whole first level and only half of the second.
    assert all(tuple(part.quantity for part in fill.fills) == (D("1"), D("1")) for fill in fills)

    fees = four_leg_fees_for_quantity(
        D("2"), spot_entry_price=D("102"), spot_exit_price=D("102"),
        perp_entry_price=D("108"), perp_exit_price=D("110"), rates=RATES,
    )
    assert fees == FourLegFees(D("0.204"), D("0.408"), D("0.108"), D("0.176"))
    assert fees.total == D("0.896")
    funding = event_funding(D("2"), "0.01", symbol=symbol)
    assert funding == D("2.2")

    cash = round_trip_from_fills(*fills, fees, funding, D("0.1"))
    decomposition = decompose_fills(*fills, **quotes)
    decomposed_net = decomposed_net_profit(decomposition, funding, fees, D("0.1"))
    # Mid basis: 10 -> 4; 2*(10-4)=12. Four half-spreads: 8.
    # Four order-book slippages: 8. Exact execution PnL: 12-8-8=-4.
    assert decomposition.mid_basis_pnl == D("12")
    assert decomposition.spread_costs == decomposition.slippage_costs == D("8")
    assert decomposition.rounding_residual == D("0")
    assert cash.price_pnl == decomposition.price_pnl == D("-4")
    assert cash.entry_basis == D("6") and cash.exit_basis == D("8")
    assert cash.spot_fees == D("0.612") and cash.perp_fees == D("0.284")
    assert cash.total_fees == D("0.896")
    assert cash.net_profit == decomposed_net == D("-2.796")
    # Adding spread/slippage again would give -18.796, which is incorrect.
    assert cash.net_profit != D("-18.796")
    without_fees = round_trip_from_fills(*fills, ZERO_FEES, funding, D("0.1"))
    assert without_fees.net_profit - cash.net_profit == fees.total
    assert (
        decomposed_net_profit(decomposition, funding, ZERO_FEES, D("0.1")) - decomposed_net
        == fees.total
    )


@pytest.mark.parametrize(
    "rate,closed_at,expected_funding,expected_net",
    [
        ("0.01", CLOSED, "3.3", "-11.1255"),
        ("-0.01", CLOSED, "-3.3", "-17.7255"),
        ("0", CLOSED, "0", "-14.4255"),
        ("0.01", SETTLED - timedelta(seconds=1), "0", "-14.4255"),
    ],
    ids=["short_receives", "short_pays", "zero_rate", "no_settlement_crossed"],
)
def test_recurring_vwap_unchanged_basis_reconciles_with_signed_event_funding(
    rate: str, closed_at: datetime, expected_funding: str, expected_net: str,
) -> None:
    fills, quotes = fills_and_quotes(recurring_vwap_books(), D("3"))
    assert tuple(fill.quote_notional for fill in fills) == (D("305"), D("297"), D("325"), D("330"))
    assert fills[0].vwap == D("101." + "6" * 46 + "7")  # 305/3, 50 significant digits
    assert fills[2].vwap == D("108." + "3" * 47)  # 325/3, 50 significant digits
    # Fees follow exact executed slices, not rounded recurring VWAP multiplied
    # back by quantity. Every leg applies its own rate once; no fee rounding.
    amounts = tuple(
        sum((calculate_fee(part.price, part.quantity, fee_rate) for part in fill.fills), D("0"))
        for fill, fee_rate in zip(
            fills, (RATES.spot_entry, RATES.spot_exit, RATES.perp_entry, RATES.perp_exit), strict=True,
        )
    )
    fees = FourLegFees(*amounts)
    assert fees == FourLegFees(D("0.305"), D("0.594"), D("0.1625"), D("0.264"))
    assert fees.total == D("1.3255")
    funding = event_funding(D("3"), rate, closed_at=closed_at)
    assert funding == D(expected_funding)

    cash = round_trip_from_fills(*fills, fees, funding, D("0.1"))
    decomposition = decompose_fills(*fills, **quotes)
    # The positive mid basis stays 9.5 at entry and exit: no convergence PnL.
    # Cash: (297-305)+(325-330)=-13. Attribution: 0-9-4=-13.
    assert decomposition.mid_basis_pnl == D("0")
    assert decomposition.spread_costs == D("9")
    assert decomposition.slippage_costs == D("4")
    assert decomposition.rounding_residual == D("0")
    assert cash.price_pnl == decomposition.price_pnl == D("-13")
    assert cash.total_fees == D("1.3255")
    assert cash.net_profit == decomposed_net_profit(decomposition, funding, fees, D("0.1"))
    assert cash.net_profit == D(expected_net)
    assert cash.net_profit < 0


@pytest.mark.parametrize("quote_name", QUOTE_NAMES)
def test_fill_decomposition_rejects_mismatched_execution_quote(quote_name: str) -> None:
    fills, quotes = fills_and_quotes(hand_calculation_books(), D("2"))
    quote = quotes[quote_name]
    quotes[quote_name] = BidAsk(quote.bid + D("0.1"), quote.ask + D("0.1"))
    with pytest.raises(ValueError, match="best price must equal"):
        decompose_fills(*fills, **quotes)


@pytest.mark.parametrize("index", range(4), ids=[leg.value for leg in LEGS])
def test_fill_decomposition_cannot_turn_insufficient_depth_into_profit(index: int) -> None:
    books = hand_calculation_books()
    fills, quotes = fills_and_quotes(books, D("2"))
    incomplete = simulate_fill(books[index], LEGS[index], D("4"))
    assert not incomplete.complete
    assert incomplete.filled_quantity == D("3")
    assert incomplete.vwap is None
    supplied = list(fills)
    supplied[index] = incomplete
    with pytest.raises(ValueError, match="all four fills must be complete"):
        decompose_fills(*supplied, **quotes)


def test_recurring_fill_reconciliation_is_independent_of_caller_decimal_context() -> None:
    fills, quotes = fills_and_quotes(recurring_vwap_books(), D("3"))
    fees = FourLegFees(D("0.305"), D("0.594"), D("0.1625"), D("0.264"))
    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        original_flags = context.flags.copy()
        decomposition = decompose_fills(*fills, **quotes)
        decomposed_net = decomposed_net_profit(decomposition, D("3.3"), fees, D("0.1"))
        cash = round_trip_from_fills(*fills, fees, D("3.3"), D("0.1"))
        assert context.prec == 4 and context.rounding == ROUND_DOWN
        assert context.traps[Inexact] is True and context.flags == original_flags
    assert decomposition.rounding_residual == D("0")
    assert cash.net_profit == decomposed_net == D("-11.1255")
