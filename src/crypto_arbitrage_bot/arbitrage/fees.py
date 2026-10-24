"""Four independent fee calculations in quote-asset equivalent units.

Each leg supplies its actual executed gross quantity and VWAP. A base-asset
fee affects inventory sizing separately; its quote equivalent must not be
charged again when a cash-flow model already accounts for that inventory cost.
"""

from dataclasses import dataclass
from decimal import Decimal

from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, require_decimal


def _require_rate(value: Decimal, name: str) -> None:
    require_decimal(value, name, nonnegative=True)
    if value >= Decimal(1):
        raise ValueError(f"{name} must be smaller than 1; rates use fractions, not percentages")


@dataclass(frozen=True)
class FeeRates:
    """Configured nonnegative fee fractions for the four individual legs."""

    spot_entry: Decimal
    spot_exit: Decimal
    perp_entry: Decimal
    perp_exit: Decimal

    def __post_init__(self) -> None:
        for name in ("spot_entry", "spot_exit", "perp_entry", "perp_exit"):
            _require_rate(getattr(self, name), name)


@dataclass(frozen=True)
class FourLegFees:
    """The four fee amounts, all expressed in quote-asset equivalent units."""

    spot_entry: Decimal
    spot_exit: Decimal
    perp_entry: Decimal
    perp_exit: Decimal

    def __post_init__(self) -> None:
        for name in ("spot_entry", "spot_exit", "perp_entry", "perp_exit"):
            require_decimal(getattr(self, name), name, nonnegative=True)

    @property
    @decimal_calculation
    def spot_total(self) -> Decimal:
        return self.spot_entry + self.spot_exit

    @property
    @decimal_calculation
    def perp_total(self) -> Decimal:
        return self.perp_entry + self.perp_exit

    @property
    @decimal_calculation
    def total(self) -> Decimal:
        return self.spot_total + self.perp_total


@decimal_calculation
def calculate_fee(price: Decimal, quantity: Decimal, rate: Decimal) -> Decimal:
    """Calculate price × executed gross quantity × fee fraction, in quote units."""
    require_decimal(price, "price", positive=True)
    require_decimal(quantity, "quantity", nonnegative=True)
    _require_rate(rate, "rate")
    return price * quantity * rate


@decimal_calculation
def four_leg_fees(
    *,
    spot_entry_price: Decimal,
    spot_entry_quantity: Decimal,
    spot_exit_price: Decimal,
    spot_exit_quantity: Decimal,
    perp_entry_price: Decimal,
    perp_entry_quantity: Decimal,
    perp_exit_price: Decimal,
    perp_exit_quantity: Decimal,
    rates: FeeRates,
) -> FourLegFees:
    """Price each leg independently; no shared quantity is silently assumed."""
    if not isinstance(rates, FeeRates):
        raise TypeError("rates must be a FeeRates instance")
    return FourLegFees(
        spot_entry=calculate_fee(spot_entry_price, spot_entry_quantity, rates.spot_entry),
        spot_exit=calculate_fee(spot_exit_price, spot_exit_quantity, rates.spot_exit),
        perp_entry=calculate_fee(perp_entry_price, perp_entry_quantity, rates.perp_entry),
        perp_exit=calculate_fee(perp_exit_price, perp_exit_quantity, rates.perp_exit),
    )


@decimal_calculation
def four_leg_fees_for_quantity(
    quantity: Decimal,
    *,
    spot_entry_price: Decimal,
    spot_exit_price: Decimal,
    perp_entry_price: Decimal,
    perp_exit_price: Decimal,
    rates: FeeRates,
) -> FourLegFees:
    """Convenience for a four-leg calculation with explicitly identical quantities."""
    require_decimal(quantity, "quantity", nonnegative=True)
    return four_leg_fees(
        spot_entry_price=spot_entry_price,
        spot_entry_quantity=quantity,
        spot_exit_price=spot_exit_price,
        spot_exit_quantity=quantity,
        perp_entry_price=perp_entry_price,
        perp_entry_quantity=quantity,
        perp_exit_price=perp_exit_price,
        perp_exit_quantity=quantity,
        rates=rates,
    )
