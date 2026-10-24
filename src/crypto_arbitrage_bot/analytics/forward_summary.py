"""Descriptive forward simulation outcomes; no live performance or win rates."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum

from crypto_arbitrage_bot.analytics.summary import NumericSummary, numeric_summary
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


@dataclass(frozen=True)
class ForwardAnalyticsConfig:
    """Tolerance in USDT/base (exit VWAP basis minus entry VWAP basis).

    Horizon upper bounds must be positive and strictly increasing. Empty
    bounds mean a single open-ended bucket. All thresholds are caller supplied.
    """

    basis_change_tolerance: Decimal
    horizon_upper_bounds: tuple[timedelta, ...]
    research_instruments: tuple[ResearchInstrument, ...] = ()

    def __post_init__(self) -> None:
        require_decimal(self.basis_change_tolerance, "basis_change_tolerance", nonnegative=True)
        if (not isinstance(self.research_instruments, tuple)
                or any(not isinstance(item, ResearchInstrument) for item in self.research_instruments)):
            raise TypeError("research_instruments must be a tuple of explicit research identities")
        if len({item.symbol for item in self.research_instruments}) != len(self.research_instruments):
            raise ValueError("duplicate research instruments")
        if not isinstance(self.horizon_upper_bounds, tuple):
            raise TypeError("horizon_upper_bounds must be a tuple")
        previous = timedelta(0)
        for bound in self.horizon_upper_bounds:
            if not isinstance(bound, timedelta):
                raise TypeError("horizon bounds must be timedelta")
            if bound <= previous:
                raise ValueError("horizon bounds must be positive and strictly increasing")
            previous = bound

    @property
    def symbols(self) -> tuple[str, ...]:
        """Descriptive grouping only, never an execution authorization."""
        return (*SUPPORTED_SYMBOLS, *sorted(item.symbol for item in self.research_instruments
                                           if item.symbol not in SUPPORTED_SYMBOLS))


@dataclass(frozen=True)
class ValidationObservation:
    id: str
    opportunity_id: str
    symbol: str
    original_decision: Decision
    status: ValidationStatus
    validation_time: datetime | None
    holding_duration: timedelta | None
    net_profit: Decimal | None
    net_return: Decimal | None
    profit_error: Decimal | None
    return_error: Decimal | None
    basis_change: Decimal | None
    funding_total: Decimal | None


@dataclass(frozen=True)
class StatusFrequency:
    status: ValidationStatus
    count: int
    proportion: Decimal | None


@dataclass(frozen=True)
class OutcomeSummary:
    validation_count: int
    statuses: tuple[StatusFrequency, ...]
    simulated_net_profit: NumericSummary
    simulated_net_return: NumericSummary
    positive_count: int
    zero_count: int
    negative_count: int
    positive_outcome_rate: Decimal | None


@dataclass(frozen=True)
class ErrorSummary:
    profit_error: NumericSummary
    return_error: NumericSummary
    absolute_profit_error: NumericSummary
    absolute_return_error: NumericSummary


class BasisOutcome(StrEnum):
    CONVERGENCE = "CONVERGENCE"
    UNCHANGED = "UNCHANGED"
    ADVERSE = "ADVERSE"
    MISSING = "MISSING"


class FundingOutcome(StrEnum):
    POSITIVE = "POSITIVE"
    ZERO = "ZERO"
    NEGATIVE = "NEGATIVE"
    MISSING = "MISSING"


@dataclass(frozen=True)
class OutcomeGroup:
    key: str
    outcomes: OutcomeSummary


@dataclass(frozen=True)
class HorizonSummary:
    lower_exclusive: timedelta | None
    upper_inclusive: timedelta | None
    outcomes: OutcomeSummary


@dataclass(frozen=True)
class ForwardAnalyticsSummary:
    total_opportunities: int
    opportunities_with_validation: int
    unvalidated_opportunities: int
    overall: OutcomeSummary
    by_decision: tuple[OutcomeGroup, ...]
    prediction_errors: ErrorSummary
    by_basis: tuple[OutcomeGroup, ...]
    by_funding: tuple[OutcomeGroup, ...]
    by_symbol: tuple[OutcomeGroup, ...]
    horizons: tuple[HorizonSummary, ...]
    missing_horizon: OutcomeSummary
    earliest_validation_time: datetime | None
    latest_validation_time: datetime | None
    config: ForwardAnalyticsConfig
    interpretation: str = (
        "Forward validation is simulated, not actual execution. PASS is not a buy. "
        "Positive-outcome rate is positive VALIDATED outcomes / all validation records, not a live win rate."
    )


def _ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        return Decimal(numerator) / Decimal(denominator)


def outcome_summary(observations: Iterable[ValidationObservation]) -> OutcomeSummary:
    rows = tuple(observations)
    eligible = tuple(row for row in rows if row.status is ValidationStatus.VALIDATED)
    profits = [require_decimal(row.net_profit, "net_profit") for row in eligible if row.net_profit is not None]
    positive = sum(value > 0 for value in profits)
    return OutcomeSummary(
        len(rows), tuple(StatusFrequency(status, sum(row.status is status for row in rows),
                                         _ratio(sum(row.status is status for row in rows), len(rows)))
                         for status in ValidationStatus),
        numeric_summary(row.net_profit for row in eligible),
        numeric_summary(row.net_return for row in eligible),
        positive, sum(value == 0 for value in profits), sum(value < 0 for value in profits),
        _ratio(positive, len(rows)),
    )


def summarize_forward_observations(
    opportunity_ids: frozenset[str], observations: Iterable[ValidationObservation],
    config: ForwardAnalyticsConfig,
) -> ForwardAnalyticsSummary:
    """Every stored validation is a sample; no best-horizon or per-opportunity selection.

    Errors are the saved validated-minus-estimated fields, not recomputed PnL.
    Funding and basis groups use VALIDATED samples only. Horizons and decision/
    symbol counts include failures, while all financial summaries exclude them.
    """
    rows = tuple(observations)
    if any(row.opportunity_id not in opportunity_ids for row in rows):
        raise ValueError("validation references an unknown opportunity")
    if any(row.symbol not in config.symbols for row in rows):
        raise ValueError("validation symbol requires an explicit research identity")
    valid = tuple(row for row in rows if row.status is ValidationStatus.VALIDATED)
    tolerance = config.basis_change_tolerance

    def basis_group(row: ValidationObservation) -> BasisOutcome:
        if row.basis_change is None:
            return BasisOutcome.MISSING
        change = require_decimal(row.basis_change, "basis_change")
        if change < tolerance.copy_negate():
            return BasisOutcome.CONVERGENCE
        if change > tolerance:
            return BasisOutcome.ADVERSE
        return BasisOutcome.UNCHANGED

    def funding_group(row: ValidationObservation) -> FundingOutcome:
        if row.funding_total is None:
            return FundingOutcome.MISSING
        total = require_decimal(row.funding_total, "funding_total")
        return FundingOutcome.POSITIVE if total > 0 else FundingOutcome.NEGATIVE if total < 0 else FundingOutcome.ZERO

    errors = ErrorSummary(
        numeric_summary(row.profit_error for row in valid),
        numeric_summary(row.return_error for row in valid),
        numeric_summary(None if row.profit_error is None else row.profit_error.copy_abs() for row in valid),
        numeric_summary(None if row.return_error is None else row.return_error.copy_abs() for row in valid),
    )
    horizons = []
    lower = None
    for upper in (*config.horizon_upper_bounds, None):
        horizons.append(HorizonSummary(lower, upper, outcome_summary(
            row for row in rows if row.holding_duration is not None
            and (lower is None or row.holding_duration > lower)
            and (upper is None or row.holding_duration <= upper)
        )))
        lower = upper
    times = [row.validation_time for row in rows if row.validation_time is not None]
    with_validation = len({row.opportunity_id for row in rows})
    return ForwardAnalyticsSummary(
        len(opportunity_ids), with_validation, len(opportunity_ids) - with_validation, outcome_summary(rows),
        tuple(OutcomeGroup(decision.value, outcome_summary(row for row in rows if row.original_decision is decision))
              for decision in Decision), errors,
        tuple(OutcomeGroup(kind.value, outcome_summary(row for row in valid if basis_group(row) is kind))
              for kind in BasisOutcome),
        tuple(OutcomeGroup(kind.value, outcome_summary(row for row in valid if funding_group(row) is kind))
              for kind in FundingOutcome),
        tuple(OutcomeGroup(symbol, outcome_summary(row for row in rows if row.symbol == symbol))
              for symbol in config.symbols),
        tuple(horizons), outcome_summary(row for row in rows if row.holding_duration is None),
        min(times) if times else None, max(times) if times else None, config,
    )
