from decimal import Decimal, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.basis import (
    basis_pct,
    executable_entry_basis,
    mid_basis,
    mid_price,
)

D = Decimal


def test_midpoint_is_exact_for_decimal_inputs():
    assert mid_price(D("0.1"), D("0.2")) == D("0.15")


def test_mid_basis_and_fraction_units():
    spot_mid = mid_price(D("9999"), D("10001"))
    perp_mid = mid_price(D("10099"), D("10101"))
    assert mid_basis(spot_mid, perp_mid) == D("100")
    assert basis_pct(mid_basis(spot_mid, perp_mid), spot_mid) == D("0.01")


def test_entry_basis_uses_spot_ask_and_perp_bid():
    assert executable_entry_basis(D("10001"), D("10099")) == D("98")


@pytest.mark.parametrize("perp, expected", [("99", "-1"), ("100", "0"), ("101", "1")])
def test_basis_supports_negative_zero_positive(perp, expected):
    assert mid_basis(D("100"), D(perp)) == D(expected)
    assert executable_entry_basis(D("100"), D(perp)) == D(expected)


def test_negative_basis_fraction():
    assert basis_pct(D("-2"), D("100")) == D("-0.02")


def test_equal_bid_and_ask_is_valid():
    assert mid_price(D("100"), D("100")) == D("100")


def test_crossed_bid_ask_is_rejected():
    with pytest.raises(ValueError, match="bid"):
        mid_price(D("101"), D("100"))


@pytest.mark.parametrize("bad", [D("0"), D("-1"), D("NaN"), D("Infinity")])
def test_nonpositive_or_nonfinite_prices_rejected(bad):
    for calculate in (
        lambda: mid_price(bad, D("100")),
        lambda: mid_basis(bad, D("100")),
        lambda: basis_pct(D("1"), bad),
        lambda: executable_entry_basis(bad, D("100")),
    ):
        with pytest.raises(ValueError):
            calculate()


@pytest.mark.parametrize("bad", [0.1, 1, "0.1", True])
def test_nondecimal_inputs_are_rejected_without_float_coercion(bad):
    with pytest.raises(TypeError):
        mid_price(bad, D("1"))
    with pytest.raises(TypeError):
        basis_pct(bad, D("1"))


def test_precision_is_independent_of_callers_decimal_context():
    with localcontext() as context:
        context.prec = 3
        result = mid_basis(D("10000000000000000000000000000.1"), D("10000000000000000000000000000.2"))
        fraction = basis_pct(D("1"), D("3"))
        assert context.prec == 3
    assert result == D("0.1")
    assert fraction == D("0." + "3" * 50)
