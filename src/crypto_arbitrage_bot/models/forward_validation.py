"""Forward research outcomes are hypothetical executions, never realized PnL."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.models.calculation import (
    CalculationResult,
    PricePnlDecomposition,
    SimulatedFill,
)
from crypto_arbitrage_bot.models.data_quality import DataQualityResult


class ValidationStatus(StrEnum):
    VALIDATED = "VALIDATED"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


class ValidationReason(StrEnum):
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    RESEARCH_IDENTITY_MISMATCH = "RESEARCH_IDENTITY_MISMATCH"
    VALIDATION_TIME_NOT_AFTER_ENTRY = "VALIDATION_TIME_NOT_AFTER_ENTRY"
    INVALID_VALIDATION_TIME = "INVALID_VALIDATION_TIME"
    INVALID_EXIT_MARKET_DATA = "INVALID_EXIT_MARKET_DATA"
    INSUFFICIENT_EXIT_DEPTH = "INSUFFICIENT_EXIT_DEPTH"
    FUNDING_EVENT_OUTSIDE_HOLDING_WINDOW = "FUNDING_EVENT_OUTSIDE_HOLDING_WINDOW"
    INVALID_FUNDING_EVENTS = "INVALID_FUNDING_EVENTS"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    MISSING_ENTRY_EVIDENCE = "MISSING_ENTRY_EVIDENCE"
    INVALID_ENTRY_EVIDENCE = "INVALID_ENTRY_EVIDENCE"


@dataclass(frozen=True)
class ForwardValidationResult:
    opportunity_id: str
    symbol: str
    status: ValidationStatus
    reason_codes: tuple[ValidationReason, ...]
    validation_time: datetime | None
    original_estimated_net_profit: Decimal | None
    original_estimated_net_return: Decimal | None
    entry_time: datetime | None = None
    holding_duration: timedelta | None = None
    matched_quantity: Decimal | None = None
    capital_occupied: Decimal | None = None
    entry_fills: tuple[SimulatedFill, ...] = ()
    exit_fills: tuple[SimulatedFill, ...] = ()
    exit_data_quality: DataQualityResult | None = None
    funding_events: tuple[FundingEvent, ...] = ()
    fees: FourLegFees | None = None
    calculation: CalculationResult | None = None
    decomposition: PricePnlDecomposition | None = None
    spot_pnl: Decimal | None = None
    perp_pnl: Decimal | None = None
    validated_simulated_net_return: Decimal | None = None
    profit_error: Decimal | None = None
    return_error: Decimal | None = None
    original_entry_basis: Decimal | None = None
    basis_change: Decimal | None = None

    @property
    def validated_simulated_net_profit(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.net_profit

    @property
    def validated_exit_basis(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.exit_basis

    @property
    def price_pnl(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.price_pnl

    @property
    def funding_total(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.funding_total

    @property
    def other_costs(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.other_costs
