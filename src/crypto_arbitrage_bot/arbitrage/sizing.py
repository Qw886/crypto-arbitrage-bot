"""Reference-price sizing and base-asset hedge matching using exact lot grids.

This is a preliminary budget calculation. Actual VWAP, fees, available depth,
and capital occupancy must be checked separately. No exchange requests occur.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, Inexact, localcontext
from enum import StrEnum
from functools import wraps
from math import lcm
from typing import ParamSpec, TypeVar

from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, require_decimal

P = ParamSpec("P")
R = TypeVar("R")


def _exact_sizing_calculation(function: Callable[P, R]) -> Callable[P, R]:
    """Do not allow finite precision to conceal a base fee or residual delta."""
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            with localcontext() as context:
                context.traps[Inexact] = True
                return function(*args, **kwargs)
        except Inexact as exc:
            raise ValueError("sizing quantity/notional exceeds supported Decimal precision") from exc
    return wrapped


@dataclass(frozen=True)
class QuantityRules:
    """Base-asset quantities and a quote-asset minimum notional."""

    step_size: Decimal
    min_quantity: Decimal = Decimal("0")
    min_notional: Decimal = Decimal("0")
    max_quantity: Decimal | None = None

    def __post_init__(self) -> None:
        require_decimal(self.step_size, "step_size", positive=True)
        require_decimal(self.min_quantity, "min_quantity", nonnegative=True)
        require_decimal(self.min_notional, "min_notional", nonnegative=True)
        if self.max_quantity is not None:
            require_decimal(self.max_quantity, "max_quantity", positive=True)
            if self.max_quantity < self.min_quantity:
                raise ValueError("max_quantity must be at least min_quantity")


class SizingStatus(StrEnum):
    READY = "READY"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"


@dataclass(frozen=True)
class SizingResult:
    """A proposal only; READY does not imply executable depth or profitability.

    ``perp_quantity`` is also the spot quantity that can be sold on both lot
    grids. Net spot holdings minus that short quantity is residual_delta.
    A positive base-asset entry fee may leave explicit unhedged dust.
    """

    gross_spot_quantity: Decimal
    net_spot_quantity: Decimal
    perp_quantity: Decimal
    residual_delta: Decimal
    spot_entry_base_fee: Decimal
    reference_spot_notional: Decimal
    reference_perp_notional: Decimal
    common_step: Decimal
    status: SizingStatus
    reasons: tuple[str, ...]

    @property
    def acceptable(self) -> bool:
        return self.status is SizingStatus.READY


def _coefficient_and_exponent(value: Decimal) -> tuple[int, int]:
    components = value.as_tuple()
    exponent = components.exponent
    if not isinstance(exponent, int):
        raise ValueError("decimal must be finite")
    coefficient = 0
    for digit in components.digits:
        coefficient = coefficient * 10 + digit
    return coefficient, exponent


def _decimal_from_coefficient(coefficient: int, exponent: int) -> Decimal:
    # Construct directly, without float or ambient-context rounding.
    digits = tuple(int(character) for character in str(coefficient))
    return Decimal((0, digits, exponent))


def _step_multiple(step: Decimal, count: int) -> Decimal:
    coefficient, exponent = _coefficient_and_exponent(step)
    return _decimal_from_coefficient(coefficient * count, exponent)


@decimal_calculation
def common_quantity_step(spot_step: Decimal, perp_step: Decimal) -> Decimal:
    """Least positive quantity on BOTH zero-origin lot grids, using integer LCM."""
    require_decimal(spot_step, "spot_step", positive=True)
    require_decimal(perp_step, "perp_step", positive=True)
    spot_coefficient, spot_exponent = _coefficient_and_exponent(spot_step)
    perp_coefficient, perp_exponent = _coefficient_and_exponent(perp_step)
    exponent = min(spot_exponent, perp_exponent)
    spot_integer = spot_coefficient * 10 ** (spot_exponent - exponent)
    perp_integer = perp_coefficient * 10 ** (perp_exponent - exponent)
    return _decimal_from_coefficient(lcm(spot_integer, perp_integer), exponent)


def _floor_ratio_to_step(numerator: Decimal, denominator: Decimal, step: Decimal) -> Decimal:
    """Floor numerator / denominator onto a lot grid without rounded division."""
    numerator_n, numerator_d = numerator.as_integer_ratio()
    denominator_n, denominator_d = denominator.as_integer_ratio()
    step_n, step_d = step.as_integer_ratio()
    count = (numerator_n * denominator_d * step_d) // (
        numerator_d * denominator_n * step_n
    )
    return _step_multiple(step, count)


@decimal_calculation
@_exact_sizing_calculation
def size_position(
    spot_budget: Decimal,
    spot_reference_price: Decimal,
    perp_reference_price: Decimal,
    spot_rules: QuantityRules,
    perp_rules: QuantityRules,
    *,
    spot_entry_base_fee_rate: Decimal = Decimal("0"),
    max_residual_delta: Decimal = Decimal("0"),
) -> SizingResult:
    """Propose matched base quantities using reference prices and lot rules.

    The budget covers the spot purchase notional only, excluding quote fees.
    A base-asset entry fee reduces net spot holdings. We round conservatively
    and report residual holdings; we do not silently claim an exact hedge.
    Constraints cover the proposed entry quantities and the matched spot exit
    quantity at today's reference price, not unknown future exchange rules or
    exit prices. This routine does not search smaller lots for an exact hedge.
    Quantity and notional accounting that would lose information beyond the
    50-digit calculation context raises ValueError instead of hiding a delta.
    """
    require_decimal(spot_budget, "spot_budget", positive=True)
    require_decimal(spot_reference_price, "spot_reference_price", positive=True)
    require_decimal(perp_reference_price, "perp_reference_price", positive=True)
    require_decimal(spot_entry_base_fee_rate, "spot_entry_base_fee_rate", nonnegative=True)
    require_decimal(max_residual_delta, "max_residual_delta", nonnegative=True)
    if spot_entry_base_fee_rate >= 1:
        raise ValueError("spot_entry_base_fee_rate must be below one")
    if not isinstance(spot_rules, QuantityRules) or not isinstance(perp_rules, QuantityRules):
        raise TypeError("spot_rules and perp_rules must be QuantityRules")

    common_step = common_quantity_step(spot_rules.step_size, perp_rules.step_size)
    retained_fraction = Decimal("1") - spot_entry_base_fee_rate
    entry_step = spot_rules.step_size if spot_entry_base_fee_rate else common_step
    gross = _floor_ratio_to_step(spot_budget, spot_reference_price, entry_step)
    if spot_rules.max_quantity is not None:
        gross = min(
            gross,
            _floor_ratio_to_step(spot_rules.max_quantity, Decimal("1"), entry_step),
        )
    if perp_rules.max_quantity is not None:
        gross = min(
            gross,
            _floor_ratio_to_step(perp_rules.max_quantity, retained_fraction, entry_step),
        )

    base_fee = gross * spot_entry_base_fee_rate
    net = gross - base_fee
    short = _floor_ratio_to_step(net, Decimal("1"), common_step)
    residual = net - short
    spot_notional = gross * spot_reference_price
    perp_notional = short * perp_reference_price
    reasons: list[str] = []
    if gross == 0 or short == 0:
        reasons.append("ZERO_HEDGE_QUANTITY")
    if gross < spot_rules.min_quantity:
        reasons.append("SPOT_ENTRY_MIN_QUANTITY")
    if short < spot_rules.min_quantity:
        reasons.append("SPOT_EXIT_MIN_QUANTITY_AT_REFERENCE")
    if short < perp_rules.min_quantity:
        reasons.append("PERP_MIN_QUANTITY")
    if spot_notional < spot_rules.min_notional:
        reasons.append("SPOT_ENTRY_MIN_NOTIONAL")
    if short * spot_reference_price < spot_rules.min_notional:
        reasons.append("SPOT_EXIT_MIN_NOTIONAL_AT_REFERENCE")
    if perp_notional < perp_rules.min_notional:
        reasons.append("PERP_MIN_NOTIONAL")
    if residual > max_residual_delta:
        reasons.append("RESIDUAL_DELTA_EXCEEDS_LIMIT")

    return SizingResult(
        gross_spot_quantity=gross,
        net_spot_quantity=net,
        perp_quantity=short,
        residual_delta=residual,
        spot_entry_base_fee=base_fee,
        reference_spot_notional=spot_notional,
        reference_perp_notional=perp_notional,
        common_step=common_step,
        status=SizingStatus.CONSTRAINT_VIOLATION if reasons else SizingStatus.READY,
        reasons=tuple(reasons),
    )
