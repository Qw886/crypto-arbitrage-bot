"""Descriptive historical Funding coverage, never a forecast or strategy PnL.

The reference cost is supplied from an existing recorded estimate. Comparing it
with past Funding at a constant reference quantity is a research screen only:
past entry/exit prices, basis PnL, and actual execution are not known here.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.analytics.summary import NumericSummary, numeric_summary
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent, accrue_short_funding
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure, FundingWindow
from crypto_arbitrage_bot.models.market import require_market_symbol
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)


class FundingStudyStatus(StrEnum):
    COMPLETE = "COMPLETE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class HistoricalFundingSlice:
    opened_at: datetime
    closed_at: datetime
    duration: timedelta
    settlement_count: int
    positive_rate_count: int
    zero_rate_count: int
    negative_rate_count: int
    observed_rates: NumericSummary
    observed_settlement_marks: NumericSummary
    historical_short_funding_at_reference_quantity: Decimal
    historical_funding_to_reference_cost_ratio: Decimal | None
    first_settlement_at: datetime | None
    last_settlement_at: datetime | None


@dataclass(frozen=True)
class HistoricalFundingStudy:
    symbol: str
    research_instrument: ResearchInstrument | None
    status: FundingStudyStatus
    evaluated_at: datetime
    source_opened_at: datetime
    source_closed_at: datetime
    reference_quantity: Decimal
    reference_round_trip_costs: Decimal | None
    slices: tuple[HistoricalFundingSlice, ...]
    failure: FundingFailure | None
    interpretation: str = (
        "Observed historical settlements at a hypothetical constant reference quantity. "
        "Reference costs come from a separate recorded book/estimate, not historical executions. "
        "This excludes historical price/basis PnL and is not NetProfit, a forecast, a trading signal, "
        "or proof that future Funding will cover costs. No hourly extrapolation."
    )


@decimal_calculation
def summarize_observed_funding(
    window: FundingWindow, *, reference_quantity: Decimal,
    reference_round_trip_costs: Decimal | None, lookback_windows: tuple[timedelta, ...],
    evaluated_at: datetime, research_instrument: ResearchInstrument | None = None,
) -> HistoricalFundingStudy:
    """Pure descriptive composition through the original Funding and stats functions.

    Each lookback ends at the original history's closed_at and is fully covered
    by that query window. It never invents missing settlements or timestamps.
    Adapters/callers supply and retain original pages; this function cannot
    independently authenticate the historical origin of supplied events.
    """
    if not isinstance(window, FundingWindow):
        raise TypeError("FundingWindow is required")
    require_market_symbol(window.symbol, research_instrument)
    require_decimal(reference_quantity, "reference_quantity", positive=True)
    if reference_round_trip_costs is not None:
        require_decimal(reference_round_trip_costs, "reference_round_trip_costs", nonnegative=True)
    opened, closed, at = (utc_timestamp(value) for value in (
        window.opened_at, window.closed_at, evaluated_at))
    if not opened < closed <= at:
        raise ValueError("history must be a strictly positive fully observed past window")
    if (not isinstance(lookback_windows, tuple) or not lookback_windows
            or any(not isinstance(d, timedelta) for d in lookback_windows)):
        raise TypeError("provide a nonempty tuple of explicit lookback durations")
    if (len(set(lookback_windows)) != len(lookback_windows)
            or any(not timedelta(0) < d <= closed-opened for d in lookback_windows)):
        raise ValueError("lookbacks must be unique, positive and covered by original history")
    if window.failure is not None and not isinstance(window.failure, FundingFailure):
        raise TypeError("history failure must be FundingFailure")
    if window.failure is not None:
        return HistoricalFundingStudy(window.symbol, research_instrument, FundingStudyStatus.UNAVAILABLE,
            at, opened, closed, reference_quantity, reference_round_trip_costs, (), window.failure)
    if (not isinstance(window.events, tuple)
            or any(not isinstance(event, FundingEvent) for event in window.events)):
        raise TypeError("history events must be a tuple of FundingEvent")
    if any(event.symbol != window.symbol or not opened < event.settled_at <= closed
           or event.event_id.startswith("FORECAST_ONLY:") for event in window.events):
        raise ValueError("history must contain same-symbol actual events inside its original window")
    # Validate uniqueness across the entire history, including events omitted
    # from shorter slices. Original accrual owns the signed cash-flow formula.
    accrue_short_funding(reference_quantity, opened, closed, window.events, window.symbol)
    slices = []
    for duration in lookback_windows:
        start = closed-duration
        events = tuple(event for event in window.events if start < event.settled_at <= closed)
        funding = accrue_short_funding(reference_quantity, start, closed, events, window.symbol)
        ratio = (None if reference_round_trip_costs is None or reference_round_trip_costs == 0
                 else funding/reference_round_trip_costs)
        times = tuple(event.settled_at for event in events)
        slices.append(HistoricalFundingSlice(
            start, closed, duration, len(events), sum(e.funding_rate > 0 for e in events),
            sum(e.funding_rate == 0 for e in events), sum(e.funding_rate < 0 for e in events),
            numeric_summary(e.funding_rate for e in events),
            numeric_summary(e.settlement_mark_price for e in events), funding, ratio,
            min(times) if times else None, max(times) if times else None,
        ))
    return HistoricalFundingStudy(window.symbol, research_instrument, FundingStudyStatus.COMPLETE,
        at, opened, closed, reference_quantity, reference_round_trip_costs, tuple(slices), None)
