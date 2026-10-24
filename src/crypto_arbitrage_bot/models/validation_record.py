"""Immutable persistence projection of a hypothetical forward validation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import ClassVar


@dataclass(frozen=True)
class ForwardValidationRecord:
    id: str
    opportunity_id: str
    symbol: str
    entry_time: datetime | None
    validation_time: datetime | None
    holding_duration: timedelta | None
    status: str
    reason_codes: tuple[str, ...]
    original_estimated_net_profit: Decimal | None
    original_estimated_net_return: Decimal | None
    validated_simulated_net_profit: Decimal | None
    validated_simulated_net_return: Decimal | None
    profit_error: Decimal | None
    return_error: Decimal | None
    original_entry_basis: Decimal | None
    validated_exit_basis: Decimal | None
    basis_change: Decimal | None
    funding_total: Decimal | None
    spot_exit_vwap: Decimal | None
    perp_exit_vwap: Decimal | None
    spot_pnl: Decimal | None
    perp_pnl: Decimal | None
    price_pnl: Decimal | None
    spot_entry_fee: Decimal | None
    perp_entry_fee: Decimal | None
    spot_exit_fee: Decimal | None
    perp_exit_fee: Decimal | None
    other_costs: Decimal | None
    matched_quantity: Decimal | None
    capital_occupied: Decimal | None
    created_at: datetime
    details_json: str

    DECIMAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "original_estimated_net_profit", "original_estimated_net_return",
        "validated_simulated_net_profit", "validated_simulated_net_return",
        "profit_error", "return_error", "original_entry_basis", "validated_exit_basis",
        "basis_change", "funding_total", "spot_exit_vwap", "perp_exit_vwap",
        "spot_pnl", "perp_pnl", "price_pnl", "spot_entry_fee", "perp_entry_fee",
        "spot_exit_fee", "perp_exit_fee", "other_costs", "matched_quantity", "capital_occupied",
    )
    DATETIME_FIELDS: ClassVar[tuple[str, ...]] = ("entry_time", "validation_time", "created_at")
