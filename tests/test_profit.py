from dataclasses import replace
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.arbitrage.profit import (
    basis_pnl,
    decompose_price_pnl,
    net_profit,
    net_return,
    occupied_capital,
    price_pnl,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.models.calculation import (
    FillSlice,
    FillStatus,
    SimulatedFill,
    TradeLeg,
)
from crypto_arbitrage_bot.models.market import BidAsk
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation

D = Decimal
ZERO_FEES = FourLegFees(D("0"), D("0"), D("0"), D("0"))


@decimal_calculation
def make_fill(leg, price, quantity="2", *, symbol="BTCUSDT", slices=None):
    quantity = D(quantity)
    parts = slices or (FillSlice(D(price), quantity),)
    notional = sum((part.price * part.quantity for part in parts), D("0"))
    return SimulatedFill(
        symbol=symbol, leg=leg, requested_quantity=quantity, filled_quantity=quantity,
        quote_notional=notional, vwap=notional / quantity, status=FillStatus.FILLED,
        fills=parts, best_price=parts[0].price,
    )


def four_fills(quantity="2"):
    return (
        make_fill(TradeLeg.SPOT_ENTRY, "101", quantity),
        make_fill(TradeLeg.SPOT_EXIT, "109", quantity),
        make_fill(TradeLeg.PERP_ENTRY, "105", quantity),
        make_fill(TradeLeg.PERP_EXIT, "111", quantity),
    )


def decomposition_inputs():
    return dict(
        quantity=D("2"),
        spot_entry_quote=BidAsk(D("99"), D("101")),
        spot_exit_quote=BidAsk(D("108"), D("112")),
        perp_entry_quote=BidAsk(D("104"), D("106")),
        perp_exit_quote=BidAsk(D("111"), D("115")),
        spot_entry=D("101.5"), spot_exit=D("107.5"),
        perp_entry=D("103.5"), perp_exit=D("115.5"),
    )


@pytest.mark.parametrize(
    "entry, exit, expected", [("4", "2", "4"), ("2", "4", "-4"), ("-2", "-4", "4")],
)
def test_basis_change_pnl(entry, exit, expected):
    assert basis_pnl(D("2"), D(entry), D(exit)) == D(expected)


def test_positive_entry_basis_is_not_income_when_exit_basis_is_equal():
    assert basis_pnl(D("2"), D("10"), D("10")) == D("0")
    assert price_pnl(D("2"), D("100"), D("110"), D("110"), D("120")) == D("0")


@pytest.mark.parametrize(
    "changed_field, expected",
    [("spot_entry", "-2"), ("spot_exit", "2"), ("perp_entry", "2"), ("perp_exit", "-2")],
)
def test_four_execution_price_directions(changed_field, expected):
    prices = dict(spot_entry=D("100"), spot_exit=D("100"), perp_entry=D("100"), perp_exit=D("100"))
    prices[changed_field] = D("101")
    assert price_pnl(D("2"), **prices) == D(expected)


def test_no_funding_unchanged_mid_basis_has_negative_round_trip_profit():
    # Spot midpoint 100, perp midpoint 105 at both entry and exit.
    execution_pnl = price_pnl(D("2"), D("101"), D("99"), D("104"), D("106"))
    assert execution_pnl == D("-8")
    assert net_profit(execution_pnl, D("0"), ZERO_FEES) == D("-8")
    assert net_profit(execution_pnl, D("0"), FourLegFees(D("1"), D("1"), D("1"), D("1"))) == D("-12")


def test_cash_flow_equals_midpoint_spread_slippage_attribution_with_different_exit_books():
    result = decompose_price_pnl(**decomposition_inputs())
    assert result.mid_basis_pnl == D("4")
    assert result.spread_costs == D("12")
    assert result.slippage_costs == D("4")
    assert result.price_pnl == D("-12")
    assert result.rounding_residual == D("0")
    assert result.mid_basis_pnl - result.spread_costs - result.slippage_costs == result.price_pnl
    # Funding is added to the already executable PnL, with no second cost deduction.
    assert net_profit(result.price_pnl, D("20"), ZERO_FEES) == D("8")


@pytest.mark.parametrize(
    "field, value",
    [("spot_entry", "100"), ("spot_exit", "109"), ("perp_entry", "105"), ("perp_exit", "114")],
)
def test_impossible_vwap_better_than_book_is_rejected(field, value):
    values = decomposition_inputs()
    values[field] = D(value)
    with pytest.raises(ValueError, match="VWAP"):
        decompose_price_pnl(**values)


def test_zero_spread_and_zero_slippage_are_allowed():
    inputs = decomposition_inputs()
    for name in ("spot_entry", "spot_exit", "perp_entry", "perp_exit"):
        inputs[name] = D("100")
        inputs[f"{name}_quote"] = BidAsk(D("100"), D("100"))
    result = decompose_price_pnl(**inputs)
    assert result.spread_costs == result.slippage_costs == result.price_pnl == D("0")


@pytest.mark.parametrize("funding, expected", [("10", "15"), ("-10", "-5"), ("0", "5")])
def test_net_profit_has_four_separate_fees_and_signed_funding(funding, expected):
    fees = FourLegFees(D("1"), D("2"), D("3"), D("4"))
    assert net_profit(D("20"), D(funding), fees, D("5")) == D(expected)


def test_fill_round_trip_preserves_own_execution_notionals_and_fee_totals():
    fees = FourLegFees(D("0.202"), D("0.436"), D("0.630"), D("0.888"))
    result = round_trip_from_fills(*four_fills(), fees, D("3"), D("0.5"))
    assert result.symbol == "BTCUSDT"
    assert result.quantity == D("2")
    assert result.entry_basis == D("4")
    assert result.exit_basis == D("2")
    assert result.price_pnl == D("4")
    assert result.spot_fees == D("0.638")
    assert result.perp_fees == D("1.518")
    assert result.total_fees == D("2.156")
    assert result.net_profit == D("4.344")


def test_fill_cashflow_does_not_multiply_back_a_recurring_vwap():
    entry = make_fill(
        TradeLeg.SPOT_ENTRY, "100", "3",
        slices=(FillSlice(D("100"), D("1")), FillSlice(D("101"), D("2"))),
    )
    result = round_trip_from_fills(
        entry, make_fill(TradeLeg.SPOT_EXIT, "100", "3"),
        make_fill(TradeLeg.PERP_ENTRY, "103", "3"),
        make_fill(TradeLeg.PERP_EXIT, "103", "3"), ZERO_FEES, D("0"),
    )
    assert result.price_pnl == D("-2")
    assert result.net_profit == D("-2")


@pytest.mark.parametrize("index", range(4))
def test_any_incomplete_leg_prevents_profit(index):
    fills = list(four_fills())
    fills[index] = replace(
        fills[index], requested_quantity=D("3"), status=FillStatus.INSUFFICIENT_DEPTH, vwap=None,
    )
    with pytest.raises(ValueError, match="complete"):
        round_trip_from_fills(*fills, ZERO_FEES, D("0"))


def test_wrong_leg_order_is_rejected():
    spot_entry, spot_exit, perp_entry, perp_exit = four_fills()
    with pytest.raises(ValueError, match="SPOT_ENTRY"):
        round_trip_from_fills(spot_exit, spot_entry, perp_entry, perp_exit, ZERO_FEES, D("0"))


def test_mixed_symbols_are_rejected():
    fills = list(four_fills())
    fills[1] = replace(fills[1], symbol="ETHUSDT")
    with pytest.raises(ValueError, match="same symbol"):
        round_trip_from_fills(*fills, ZERO_FEES, D("0"))


def test_base_fee_inventory_mismatch_is_rejected_in_quote_fee_round_trip():
    fills = list(four_fills())
    fills[1] = make_fill(TradeLeg.SPOT_EXIT, "109", "1.998")
    with pytest.raises(ValueError, match="identical base quantities"):
        round_trip_from_fills(*fills, ZERO_FEES, D("0"))


@pytest.mark.parametrize("replacement", [dict(quote_notional=D("999")), dict(vwap=D("99")), dict(fills=())])
def test_internally_inconsistent_fill_is_rejected(replacement):
    fills = list(four_fills())
    fills[0] = replace(fills[0], **replacement)
    with pytest.raises(ValueError):
        round_trip_from_fills(*fills, ZERO_FEES, D("0"))


def test_wrong_side_book_walk_is_rejected_even_when_vwap_is_consistent():
    fills = list(four_fills())
    fills[0] = make_fill(
        TradeLeg.SPOT_ENTRY, "102", "2",
        slices=(FillSlice(D("102"), D("1")), FillSlice(D("101"), D("1"))),
    )
    with pytest.raises(ValueError, match="ascending"):
        round_trip_from_fills(*fills, ZERO_FEES, D("0"))


def test_occupied_capital_includes_both_legs_and_separate_reserves():
    capital = occupied_capital(D("1000"), D("1010"), D("10"), D("80"))
    assert capital == D("2100")
    assert net_return(D("21"), capital) == D("0.01")
    assert net_return(D("-21"), capital) == D("-0.01")


@pytest.mark.parametrize("capital", [D("0"), D("-1"), D("NaN"), D("Infinity")])
def test_invalid_return_denominator_is_rejected(capital):
    with pytest.raises(ValueError):
        net_return(D("1"), capital)


@pytest.mark.parametrize("index", range(4))
def test_negative_capital_component_is_rejected(index):
    values = [D("100"), D("100"), D("1"), D("1")]
    values[index] = D("-1")
    with pytest.raises(ValueError):
        occupied_capital(*values)


@pytest.mark.parametrize("index", [0, 1])
def test_main_capital_components_must_be_positive(index):
    values = [D("100"), D("100"), D("0"), D("0")]
    values[index] = D("0")
    with pytest.raises(ValueError):
        occupied_capital(*values)


@pytest.mark.parametrize("field", ["quantity", "spot_entry", "spot_exit", "perp_entry", "perp_exit"])
def test_price_pnl_rejects_float(field):
    values = dict(quantity=D("1"), spot_entry=D("100"), spot_exit=D("100"), perp_entry=D("100"), perp_exit=D("100"))
    values[field] = 1.0
    with pytest.raises(TypeError):
        price_pnl(**values)


@pytest.mark.parametrize(
    "call",
    [
        lambda: basis_pnl(1.0, D("1"), D("0")),
        lambda: basis_pnl(D("1"), 1.0, D("0")),
        lambda: basis_pnl(D("1"), D("1"), 0.0),
        lambda: net_profit(1.0, D("0"), ZERO_FEES),
        lambda: net_profit(D("1"), 0.0, ZERO_FEES),
        lambda: net_profit(D("1"), D("0"), ZERO_FEES, 0.0),
        lambda: net_return(D("1"), 100.0),
        lambda: net_return(1.0, D("100")),
        lambda: occupied_capital(D("100"), D("100"), 1.0, D("0")),
    ],
)
def test_other_profit_apis_reject_float(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("value", [D("NaN"), D("Infinity"), D("-Infinity")])
def test_nonfinite_profit_or_funding_is_rejected(value):
    with pytest.raises(ValueError):
        net_profit(value, D("0"), ZERO_FEES)
    with pytest.raises(ValueError):
        net_profit(D("0"), value, ZERO_FEES)


def test_negative_other_cost_is_rejected():
    with pytest.raises(ValueError):
        net_profit(D("1"), D("0"), ZERO_FEES, D("-1"))


def test_calculation_context_is_50_digits_and_does_not_pollute_caller():
    fills = four_fills()
    with localcontext() as context:
        context.prec = 5
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        before_flags = context.flags.copy()
        third = net_return(D("1"), D("3"))
        exact_pnl = price_pnl(D("0.123456789"), D("100"), D("101"), D("105"), D("102"))
        result = round_trip_from_fills(*fills, ZERO_FEES, D("0.123456789"))
        decomposition = decompose_price_pnl(**decomposition_inputs())
        assert context.prec == 5
        assert context.rounding == ROUND_DOWN
        assert context.traps[Inexact] is True
        assert context.flags == before_flags
    assert third == D("0." + "3" * 50)
    assert exact_pnl == D("0.493827156")
    assert result.net_profit == D("4.123456789")
    assert decomposition.price_pnl == D("-12")
