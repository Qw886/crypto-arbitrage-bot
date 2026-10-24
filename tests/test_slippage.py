from datetime import UTC, datetime
from decimal import Decimal, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.models.calculation import FillStatus, TradeLeg
from crypto_arbitrage_bot.models.market import Market, OrderBook, OrderBookLevel

D = Decimal
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_book(
    market: Market = Market.SPOT,
    *,
    bids: tuple[tuple[str, str], ...] = (("99", "1"), ("98", "2")),
    asks: tuple[tuple[str, str], ...] = (("101", "1"), ("102", "2")),
) -> OrderBook:
    return OrderBook(
        symbol="BTCUSDT",
        market=market,
        bids=tuple(OrderBookLevel(D(price), D(quantity)) for price, quantity in bids),
        asks=tuple(OrderBookLevel(D(price), D(quantity)) for price, quantity in asks),
        observed_at=NOW,
    )


@pytest.mark.parametrize(
    ("leg", "expected_price"),
    [
        (TradeLeg.SPOT_ENTRY, "101"),
        (TradeLeg.SPOT_EXIT, "99"),
        (TradeLeg.PERP_ENTRY, "99"),
        (TradeLeg.PERP_EXIT, "101"),
    ],
)
def test_all_four_directions_consume_correct_side(leg: TradeLeg, expected_price: str) -> None:
    fill = simulate_fill(make_book(leg.market), leg, D("0.5"))
    assert fill.status is FillStatus.FILLED
    assert fill.complete
    assert fill.vwap == D(expected_price)
    assert fill.best_price == D(expected_price)
    assert fill.quote_notional == D(expected_price) * D("0.5")
    assert len(fill.fills) == 1


def test_multiple_levels_and_partial_last_level() -> None:
    fill = simulate_fill(make_book(), TradeLeg.SPOT_ENTRY, D("2"))
    # 1 @ 101 + 1 @ 102 = 203 USDT, VWAP = 101.5 USDT/base.
    assert fill.vwap == D("101.5")
    assert fill.quote_notional == D("203")
    assert tuple(part.quantity for part in fill.fills) == (D("1"), D("1"))
    assert fill.filled_quantity == fill.requested_quantity == D("2")


def test_multiple_bid_levels_for_short_entry() -> None:
    fill = simulate_fill(make_book(Market.USDTM), TradeLeg.PERP_ENTRY, D("2.5"))
    assert fill.quote_notional == D("246")
    assert fill.vwap == D("98.4")
    assert tuple(part.quantity for part in fill.fills) == (D("1"), D("1.5"))


def test_exact_book_capacity_is_complete() -> None:
    fill = simulate_fill(make_book(), TradeLeg.SPOT_ENTRY, D("3"))
    assert fill.complete
    assert fill.filled_quantity == D("3")
    assert fill.quote_notional == D("305")


def test_insufficient_depth_preserves_partial_fills_without_extrapolation() -> None:
    fill = simulate_fill(make_book(), TradeLeg.SPOT_ENTRY, D("4"))
    assert fill.status is FillStatus.INSUFFICIENT_DEPTH
    assert not fill.complete
    assert fill.requested_quantity == D("4")
    assert fill.filled_quantity == D("3")
    assert fill.quote_notional == D("305")
    assert fill.vwap is None
    assert sum((part.quantity for part in fill.fills), D("0")) == D("3")


def test_empty_consumed_side_is_insufficient_depth() -> None:
    fill = simulate_fill(make_book(asks=()), TradeLeg.SPOT_ENTRY, D("1"))
    assert fill.status is FillStatus.INSUFFICIENT_DEPTH
    assert fill.vwap is None
    assert fill.best_price is None
    assert fill.filled_quantity == fill.quote_notional == D("0")
    assert fill.fills == ()


def test_decimal_values_are_not_converted_through_float() -> None:
    book = make_book(
        bids=(("0.09", "1"),),
        asks=(("0.1", "0.1"), ("0.2", "0.2")),
    )
    fill = simulate_fill(book, TradeLeg.SPOT_ENTRY, D("0.3"))
    assert fill.filled_quantity == D("0.3")
    assert fill.quote_notional == D("0.05")
    assert fill.vwap == D("0.16666666666666666666666666666666666666666666666667")


def test_calculation_is_independent_of_callers_decimal_precision() -> None:
    with localcontext() as context:
        context.prec = 3
        fill = simulate_fill(make_book(), TradeLeg.SPOT_ENTRY, D("2.3456789"))
    assert fill.filled_quantity == D("2.3456789")
    assert fill.quote_notional == D("238.2592478")


def test_quantity_precision_loss_cannot_conceal_a_partial_fill() -> None:
    book = make_book(bids=(("0.9", "1"),), asks=(("1", "1"),))
    with pytest.raises(ValueError, match="precision"):
        simulate_fill(book, TradeLeg.SPOT_ENTRY, D("1E+60"))


def test_tiny_level_cannot_be_lost_before_a_complete_fill() -> None:
    book = make_book(
        bids=(("0.9", "1"),),
        asks=(("1", "1E-60"), ("2", "1")),
    )
    with pytest.raises(ValueError, match="precision"):
        simulate_fill(book, TradeLeg.SPOT_ENTRY, D("1"))


def test_notional_precision_loss_is_explicit() -> None:
    book = make_book(
        bids=(("1", "1"),),
        asks=(("1.12345678901234567890123456789012345678901234567890", "2"),),
    )
    with pytest.raises(ValueError, match="precision"):
        simulate_fill(book, TradeLeg.SPOT_ENTRY, D("1.123456789012345678901"))


@pytest.mark.parametrize("quantity", [0.1, 1, "1"])
def test_non_decimal_quantity_is_rejected(quantity: object) -> None:
    with pytest.raises(TypeError, match="Decimal"):
        simulate_fill(make_book(), TradeLeg.SPOT_ENTRY, quantity)  # type: ignore[arg-type]


@pytest.mark.parametrize("quantity", [D("0"), D("-1"), D("NaN"), D("Infinity")])
def test_invalid_quantity_is_rejected(quantity: Decimal) -> None:
    with pytest.raises(ValueError):
        simulate_fill(make_book(), TradeLeg.SPOT_ENTRY, quantity)


@pytest.mark.parametrize(
    "leg", [TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT]
)
def test_market_mismatch_is_rejected(leg: TradeLeg) -> None:
    with pytest.raises(ValueError, match="market"):
        simulate_fill(make_book(), leg, D("1"))


@pytest.mark.parametrize(
    ("bids", "asks"),
    [
        ((("98", "1"), ("99", "1")), (("101", "1"),)),
        ((("99", "1"),), (("102", "1"), ("101", "1"))),
        ((("99", "1"), ("99", "1")), (("101", "1"),)),
        ((("99", "1"),), (("101", "1"), ("101", "1"))),
        ((("102", "1"),), (("101", "1"),)),
    ],
)
def test_invalid_book_order_is_rejected(
    bids: tuple[tuple[str, str], ...], asks: tuple[tuple[str, str], ...]
) -> None:
    with pytest.raises(ValueError):
        simulate_fill(make_book(bids=bids, asks=asks), TradeLeg.SPOT_ENTRY, D("1"))
