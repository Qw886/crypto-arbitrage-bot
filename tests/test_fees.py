from decimal import Decimal, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.fees import (
    FeeRates,
    FourLegFees,
    calculate_fee,
    four_leg_fees,
    four_leg_fees_for_quantity,
)

D = Decimal


def test_fee_uses_decimal_fraction_and_actual_notional():
    assert calculate_fee(D("60000"), D("0.2"), D("0.001")) == D("12")


def test_each_leg_uses_own_price_quantity_and_rate():
    result = four_leg_fees(
        spot_entry_price=D("100"), spot_entry_quantity=D("2"),
        spot_exit_price=D("110"), spot_exit_quantity=D("1.9"),
        perp_entry_price=D("105"), perp_entry_quantity=D("1.8"),
        perp_exit_price=D("115"), perp_exit_quantity=D("1.7"),
        rates=FeeRates(D("0.001"), D("0.002"), D("0.003"), D("0.004")),
    )
    assert result.spot_entry == D("0.200")
    assert result.spot_exit == D("0.418")
    assert result.perp_entry == D("0.567")
    assert result.perp_exit == D("0.782")
    assert result.spot_total == D("0.618")
    assert result.perp_total == D("1.349")
    assert result.total == D("1.967")


def test_same_quantity_helper_remains_explicit():
    result = four_leg_fees_for_quantity(
        D("0.2"), spot_entry_price=D("60000"), spot_exit_price=D("61000"),
        perp_entry_price=D("60100"), perp_exit_price=D("61050"),
        rates=FeeRates(D("0.001"), D("0.001"), D("0.0005"), D("0.0005")),
    )
    assert result == FourLegFees(D("12"), D("12.2"), D("6.01"), D("6.105"))
    assert result.total == D("36.315")


@pytest.mark.parametrize("quantity, rate", [("0", "0.001"), ("0.2", "0"), ("0", "0")])
def test_zero_fee_or_zero_quantity(quantity, rate):
    assert calculate_fee(D("60000"), D(quantity), D(rate)) == D("0")


@pytest.mark.parametrize("rate", [D("-0.001"), D("1"), D("10"), D("NaN"), D("Infinity")])
def test_invalid_fee_fraction_rejected(rate):
    with pytest.raises(ValueError):
        calculate_fee(D("100"), D("1"), rate)
    with pytest.raises(ValueError):
        FeeRates(rate, D("0"), D("0"), D("0"))


@pytest.mark.parametrize("field", ["price", "quantity", "rate"])
def test_float_inputs_rejected(field):
    values = {"price": D("100"), "quantity": D("1"), "rate": D("0.001")}
    values[field] = 0.1
    with pytest.raises(TypeError):
        calculate_fee(**values)


def test_fee_rate_model_rejects_float():
    with pytest.raises(TypeError):
        FeeRates(D("0.001"), 0.001, D("0.0005"), D("0.0005"))


@pytest.mark.parametrize("price, quantity", [("0", "1"), ("-1", "1"), ("100", "-1")])
def test_invalid_fee_price_or_quantity_rejected(price, quantity):
    with pytest.raises(ValueError):
        calculate_fee(D(price), D(quantity), D("0.001"))


def test_fee_amount_model_rejects_negative_cost():
    with pytest.raises(ValueError):
        FourLegFees(D("-1"), D("0"), D("0"), D("0"))


def test_decimal_fee_calculation_does_not_round_to_callers_context():
    with localcontext() as context:
        context.prec = 3
        amount = calculate_fee(D("0.123456789"), D("0.987654321"), D("0.001"))
        result = FourLegFees(amount, amount, amount, amount)
        total = result.total
        assert context.prec == 3
    assert amount == D("0.000121932631112635269")
    assert total == D("0.000487730524450541076")
