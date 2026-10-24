"""Explicit inputs and outputs for one offline, quote-fee research estimate."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.arbitrage.fees import FeeRates, FourLegFees
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.sizing import QuantityRules, SizingResult
from crypto_arbitrage_bot.models.calculation import (
    CalculationResult,
    PricePnlDecomposition,
    SimulatedFill,
)
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig, DataQualityResult
from crypto_arbitrage_bot.models.market import Market, OrderBook, Timestamp
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class EstimateStatus(StrEnum):
    ESTIMATED = "ESTIMATED"
    INCOMPLETE_SNAPSHOT = "INCOMPLETE_SNAPSHOT"
    INVALID_MARKET_DATA = "INVALID_MARKET_DATA"
    SIZING_REJECTED = "SIZING_REJECTED"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


@dataclass(frozen=True)
class EstimateConfig:
    """Exactly one quantity cap or spot purchase budget, plus explicit lot rules.

    spot_budget excludes fees and the full perpetual reserve. capital_limit,
    when supplied, caps the total occupied research capital. All four fees are
    separate quote costs; no base-asset fee inventory adjustment is assumed.
    """

    spot_rules: QuantityRules
    perp_rules: QuantityRules
    fee_rates: FeeRates
    target_quantity: Decimal | None = None
    spot_budget: Decimal | None = None
    capital_limit: Decimal | None = None
    stress_reserve: Decimal = Decimal("0")
    other_costs: Decimal = Decimal("0")
    data_quality: DataQualityConfig = DataQualityConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.spot_rules, QuantityRules) or not isinstance(self.perp_rules, QuantityRules):
            raise TypeError("spot_rules and perp_rules must be QuantityRules")
        if not isinstance(self.fee_rates, FeeRates):
            raise TypeError("fee_rates must be FeeRates")
        if not isinstance(self.data_quality, DataQualityConfig):
            raise TypeError("data_quality must be DataQualityConfig")
        if (self.target_quantity is None) == (self.spot_budget is None):
            raise ValueError("provide exactly one target_quantity or spot_budget")
        for name in ("target_quantity", "spot_budget", "capital_limit"):
            value = getattr(self, name)
            if value is not None:
                require_decimal(value, name, positive=True)
        for name in ("stress_reserve", "other_costs"):
            require_decimal(getattr(self, name), name, nonnegative=True)


@dataclass(frozen=True)
class ExitFundingScenario:
    """Caller-supplied exit books and settlement events, never a forecast.

    An empty events tuple explicitly means no funding cash flow is assumed.
    Book timestamps are retained; opened_at/closed_at define the hypothetical
    holding window. The same books may be explicitly reused for a static exit.
    """

    spot_exit_book: OrderBook
    perp_exit_book: OrderBook
    opened_at: Timestamp
    closed_at: Timestamp
    funding_events: tuple[FundingEvent, ...]
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.spot_exit_book, OrderBook) or not isinstance(self.perp_exit_book, OrderBook):
            raise TypeError("exit books must be OrderBook")
        if self.spot_exit_book.market is not Market.SPOT or self.perp_exit_book.market is not Market.USDTM:
            raise ValueError("exit scenario requires Spot and USDT-M books")
        if self.spot_exit_book.symbol != self.perp_exit_book.symbol:
            raise ValueError("exit books must have the same symbol")
        for name in ("opened_at", "closed_at"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name), name))
        if self.closed_at < self.opened_at:
            raise ValueError("closed_at must not precede opened_at")
        if not isinstance(self.funding_events, tuple) or any(
            not isinstance(event, FundingEvent) for event in self.funding_events
        ):
            raise TypeError("funding_events must be a tuple of FundingEvent")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("an explicit scenario description is required")


@dataclass(frozen=True)
class OfflineEstimate:
    """An estimate or a failed attempt, not an opportunity/trading decision.

    Fills are ordered spot entry, spot exit, perp entry, perp exit. Failed
    attempts never expose profit or return. The original inputs retain every
    source timestamp and every exit/funding/fee/capital assumption.
    """

    status: EstimateStatus
    assembly: SnapshotAssembly
    config: EstimateConfig
    scenario: ExitFundingScenario
    sizing: SizingResult | None = None
    fills: tuple[SimulatedFill, ...] = ()
    fees: FourLegFees | None = None
    calculation: CalculationResult | None = None
    decomposition: PricePnlDecomposition | None = None
    executable_entry_basis: Decimal | None = None
    basis_pct: Decimal | None = None
    entry_vwap_basis_pct: Decimal | None = None
    capital_occupied: Decimal | None = None
    net_return: Decimal | None = None
    issues: tuple[str, ...] = ()
    data_quality: DataQualityResult | None = None

    @property
    def complete(self) -> bool:
        return self.status is EstimateStatus.ESTIMATED and self.calculation is not None

    @property
    def symbol(self) -> str:
        return self.assembly.symbol

    @property
    def matched_quantity(self) -> Decimal | None:
        return None if self.sizing is None else self.sizing.perp_quantity

    @property
    def residual_delta(self) -> Decimal | None:
        return None if self.sizing is None else self.sizing.residual_delta

    @property
    def spot_entry_vwap(self) -> Decimal | None:
        return self.fills[0].vwap if self.fills else None

    @property
    def perp_entry_vwap(self) -> Decimal | None:
        return self.fills[2].vwap if len(self.fills) == 4 else None

    @property
    def funding_rate(self) -> Decimal | None:
        return None if self.assembly.snapshot is None else self.assembly.snapshot.funding_rate

    @property
    def estimated_funding(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.funding_total

    @property
    def net_profit(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.net_profit

    @property
    def price_pnl(self) -> Decimal | None:
        return None if self.calculation is None else self.calculation.price_pnl

    @property
    def mid_basis_pnl(self) -> Decimal | None:
        return None if self.decomposition is None else self.decomposition.mid_basis_pnl
