from decimal import Decimal, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.sizing import (
    QuantityRules,
    SizingStatus,
    common_quantity_step,
    size_position,
)

D = Decimal


@pytest.mark.parametrize(
    ("spot_step", "perp_step", "expected"),
    [
        ("0.001", "0.005", "0.005"),
        ("0.0025", "0.006", "0.03"),
        ("0.0200", "0.03", "0.06"),
        ("2", "3", "6"),
        ("1E+2", "3E+1", "300"),
        ("0.00000000000000000002", "0.00000000000000000003", "0.00000000000000000006"),
    ],
)
def test_common_step_is_exact_integer_lcm(spot_step: str, perp_step: str, expected: str) -> None:
    assert common_quantity_step(D(spot_step), D(perp_step)) == D(expected)


def test_zero_fee_matches_base_quantity_not_usdt_notional() -> None:
    result = size_position(
        D("1000"), D("100"), D("102"),
        QuantityRules(D("0.01")), QuantityRules(D("0.1")),
    )
    assert result.status is SizingStatus.READY
    assert result.gross_spot_quantity == result.net_spot_quantity == result.perp_quantity == D("10")
    assert result.residual_delta == D("0")
    assert result.reference_spot_notional == D("1000")
    assert result.reference_perp_notional == D("1020")
    assert result.reference_spot_notional != result.reference_perp_notional


def test_unusual_steps_round_down_to_common_grid() -> None:
    result = size_position(
        D("1"), D("10"), D("11"),
        QuantityRules(D("0.0025")), QuantityRules(D("0.006")),
    )
    assert result.acceptable
    assert result.common_step == D("0.03")
    assert result.perp_quantity == result.gross_spot_quantity == D("0.09")
    assert result.reference_spot_notional == D("0.90")


def test_minimum_quantity_and_notional_constraints_are_reported() -> None:
    result = size_position(
        D("1"), D("100"), D("101"),
        QuantityRules(D("0.001"), min_quantity=D("0.02"), min_notional=D("5")),
        QuantityRules(D("0.001"), min_quantity=D("0.03"), min_notional=D("4")),
    )
    assert result.status is SizingStatus.CONSTRAINT_VIOLATION
    assert "SPOT_ENTRY_MIN_QUANTITY" in result.reasons
    assert "SPOT_EXIT_MIN_QUANTITY_AT_REFERENCE" in result.reasons
    assert "PERP_MIN_QUANTITY" in result.reasons
    assert "SPOT_ENTRY_MIN_NOTIONAL" in result.reasons
    assert "SPOT_EXIT_MIN_NOTIONAL_AT_REFERENCE" in result.reasons
    assert "PERP_MIN_NOTIONAL" in result.reasons


@pytest.mark.parametrize(("spot_max", "perp_max", "expected"), [("0.07", "0.10", "0.06"), ("0.10", "0.04", "0.03")])
def test_maximum_quantities_cap_proposal(spot_max: str, perp_max: str, expected: str) -> None:
    result = size_position(
        D("100"), D("100"), D("101"),
        QuantityRules(D("0.0025"), max_quantity=D(spot_max)),
        QuantityRules(D("0.006"), max_quantity=D(perp_max)),
    )
    assert result.acceptable
    assert result.gross_spot_quantity == result.perp_quantity == D(expected)


def test_base_asset_fee_reduces_holdings_and_reports_residual() -> None:
    result = size_position(
        D("1000"), D("100"), D("102"),
        QuantityRules(D("0.01")), QuantityRules(D("0.1")),
        spot_entry_base_fee_rate=D("0.001"),
    )
    assert result.gross_spot_quantity == D("10")
    assert result.spot_entry_base_fee == D("0.01")
    assert result.net_spot_quantity == D("9.99")
    assert result.perp_quantity == D("9.9")
    assert result.residual_delta == D("0.09")
    assert not result.acceptable
    assert result.reasons == ("RESIDUAL_DELTA_EXCEEDS_LIMIT",)


def test_explicit_residual_tolerance_does_not_erase_delta() -> None:
    result = size_position(
        D("1000"), D("100"), D("102"),
        QuantityRules(D("0.01")), QuantityRules(D("0.1")),
        spot_entry_base_fee_rate=D("0.001"), max_residual_delta=D("0.09"),
    )
    assert result.acceptable
    assert result.residual_delta == D("0.09")


def test_fee_adjustment_keeps_hedged_spot_exit_on_both_lot_grids() -> None:
    result = size_position(
        D("100"), D("100"), D("102"),
        QuantityRules(D("0.03")), QuantityRules(D("0.02")),
        spot_entry_base_fee_rate=D("0.01"), max_residual_delta=D("0.021"),
    )
    assert result.gross_spot_quantity == D("0.99")
    assert result.net_spot_quantity == D("0.9801")
    assert result.perp_quantity == D("0.96")
    assert result.residual_delta == D("0.0201")
    assert result.perp_quantity % D("0.03") == 0
    assert result.perp_quantity % D("0.02") == 0
    assert result.acceptable


def test_base_fee_with_perp_maximum_rounds_conservatively() -> None:
    result = size_position(
        D("10000"), D("100"), D("102"),
        QuantityRules(D("0.01")), QuantityRules(D("0.1"), max_quantity=D("1")),
        spot_entry_base_fee_rate=D("0.001"), max_residual_delta=D("0.1"),
    )
    assert result.gross_spot_quantity == D("1")
    assert result.net_spot_quantity == D("0.999")
    assert result.perp_quantity == D("0.9")
    assert result.residual_delta == D("0.099")
    assert result.acceptable


def test_budget_below_one_common_lot_is_not_ready() -> None:
    result = size_position(
        D("0.001"), D("100"), D("100"),
        QuantityRules(D("0.001")), QuantityRules(D("0.01")),
    )
    assert not result.acceptable
    assert result.gross_spot_quantity == result.perp_quantity == D("0")
    assert "ZERO_HEDGE_QUANTITY" in result.reasons


def test_decimal_budget_does_not_round_up_a_lot() -> None:
    result = size_position(
        D("0.3"), D("0.1"), D("0.11"),
        QuantityRules(D("0.1")), QuantityRules(D("0.1")),
    )
    assert result.perp_quantity == D("3")
    assert result.reference_spot_notional == D("0.3")


def test_sizing_is_independent_of_ambient_decimal_context() -> None:
    with localcontext() as context:
        context.prec = 3
        result = size_position(
            D("123.456789"), D("10"), D("10.1"),
            QuantityRules(D("0.000001")), QuantityRules(D("0.000001")),
        )
    assert result.gross_spot_quantity == D("12.345678")
    assert result.reference_spot_notional == D("123.456780")


def test_subprecision_base_fee_cannot_be_silently_lost() -> None:
    with pytest.raises(ValueError, match="precision"):
        size_position(
            D("1"), D("1"), D("1"),
            QuantityRules(D("0.01")), QuantityRules(D("0.01")),
            spot_entry_base_fee_rate=D("1E-60"),
        )


def test_subprecision_reference_notional_cannot_exceed_budget_silently() -> None:
    with pytest.raises(ValueError, match="precision"):
        size_position(
            D("1"), D("1.12345678901234567890123456789012345678901234567890"), D("1"),
            QuantityRules(D("0.000000000000000000001")),
            QuantityRules(D("0.000000000000000000001")),
        )


@pytest.mark.parametrize("bad_value", [0.01, 1, "0.01"])
def test_quantity_rules_reject_non_decimal(bad_value: object) -> None:
    with pytest.raises(TypeError, match="Decimal"):
        QuantityRules(bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_fee", [D("-0.1"), D("1"), D("NaN"), D("Infinity")])
def test_invalid_fee_is_rejected(bad_fee: Decimal) -> None:
    with pytest.raises(ValueError):
        size_position(
            D("100"), D("100"), D("100"),
            QuantityRules(D("0.01")), QuantityRules(D("0.01")),
            spot_entry_base_fee_rate=bad_fee,
        )


def test_float_budget_is_rejected() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        size_position(
            0.3, D("0.1"), D("0.11"),  # type: ignore[arg-type]
            QuantityRules(D("0.1")), QuantityRules(D("0.1")),
        )
