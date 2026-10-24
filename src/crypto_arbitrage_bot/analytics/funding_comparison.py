"""Compare observed settlements over common periods, never rank trading signals."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.funding_history_summary import (
    FundingStudyStatus,
    HistoricalFundingStudy,
    summarize_observed_funding,
)
from crypto_arbitrage_bot.arbitrage.fees import FeeRates, FourLegFees
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.models.calculation import PricePnlDecomposition
from crypto_arbitrage_bot.models.funding_info import FundingAdjustment
from crypto_arbitrage_bot.models.market import require_market_symbol
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)


class FundingComparisonStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class FundingCostReference:
    symbol: str
    quantity: Decimal
    round_trip_costs: Decimal | None
    opportunity_id: str
    record_created_at: datetime
    research_instrument: ResearchInstrument | None = None

    def __post_init__(self) -> None:
        require_market_symbol(self.symbol, self.research_instrument)
        require_decimal(self.quantity, "reference quantity", positive=True)
        if self.round_trip_costs is not None:
            require_decimal(self.round_trip_costs, "reference costs", nonnegative=True)
        if type(self.opportunity_id) is not str or not self.opportunity_id.strip():
            raise ValueError("explicit recorded opportunity identity required")
        object.__setattr__(self, "record_created_at", utc_timestamp(self.record_created_at))


@dataclass(frozen=True)
class FundingReferenceEvidence:
    reference: FundingCostReference
    record_sha256: str
    fees: FourLegFees
    fee_rates: FeeRates
    spread_costs: Decimal
    slippage_costs: Decimal
    other_costs: Decimal
    capital_occupied: Decimal
    matched_notional: Decimal | None


def _saved_decimal(value: object) -> Decimal:
    if type(value) is not str:
        raise ValueError("recorded financial values must be Decimal text")
    return require_decimal(Decimal(value), "recorded cost component")


@decimal_calculation
def cost_reference_from_opportunity(record: OpportunityRecord) -> FundingReferenceEvidence:
    """Map existing cost attribution only; do not compute a new trade PnL.

    These four fills include a hypothetical exit. Their fees/depth costs are a
    dated reference, not observed executions throughout the history window.
    """
    entry = decode_entry(record)
    quantity = record.matched_quantity
    assert quantity is not None and record.capital_occupied is not None
    if (entry.spot.symbol != record.symbol or entry.perp.symbol != record.symbol
            or entry.spot.filled_quantity != quantity or entry.perp.filled_quantity != quantity):
        raise ValueError("entry identity/quantity differs from saved matched quantity")
    estimate = json.loads(record.details_json)["estimate"]
    fees = FourLegFees(**{key: _saved_decimal(value) for key, value in estimate["fees"].items()})
    decomposition = PricePnlDecomposition(**{
        key: _saved_decimal(value) for key, value in estimate["decomposition"].items()})
    if (fees.spot_entry != entry.spot_fee or fees.perp_entry != entry.perp_fee
            or fees.spot_total != record.spot_fees or fees.perp_total != record.perp_fees):
        raise ValueError("saved fee components disagree with recorded totals")
    costs = fees.total + decomposition.spread_costs + decomposition.slippage_costs + entry.other_costs
    reference = FundingCostReference(record.symbol, quantity, costs, record.id, record.created_at,
                                     entry.research_instrument)
    return FundingReferenceEvidence(reference, hashlib.sha256(exact_json(record).encode()).hexdigest(),
        fees, entry.rates, decomposition.spread_costs, decomposition.slippage_costs, entry.other_costs,
        record.capital_occupied, record.matched_notional)


@dataclass(frozen=True)
class FundingComparisonRow:
    reference: FundingCostReference
    study: HistoricalFundingStudy
    observed_adjustment: FundingAdjustment | None


@dataclass(frozen=True)
class FundingHorizonComparison:
    duration: timedelta
    ranked_symbols: tuple[str, ...]
    unranked_symbols: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalFundingComparison:
    status: FundingComparisonStatus
    opened_at: datetime
    closed_at: datetime
    evaluated_at: datetime
    rows: tuple[FundingComparisonRow, ...]
    horizons: tuple[FundingHorizonComparison, ...]
    interpretation: str = (
        "Common-period observed Funding at recorded reference quantities, not NetProfit, "
        "a backtest, a forecast or entry authorization. Costs are separate recorded book/scenario "
        "references, not historical executions; historical basis PnL is unavailable. "
        "Ranking is descriptive cost coverage only. Observed current adjustments do not "
        "reconstruct past schedules or manufacture future settlements."
    )


def compare_historical_funding(
    references: tuple[FundingCostReference, ...], windows: tuple[FundingWindow, ...], *,
    lookbacks: tuple[timedelta, ...], evaluated_at: datetime,
    adjustments: tuple[FundingAdjustment, ...] = (),
) -> HistoricalFundingComparison:
    """Compose the existing accrual/statistics for exactly aligned query windows.

    Source failures remain visible and unranked; no symbol or missing cost drops
    out of the denominator. This pure seam does not authenticate supplied pages.
    """
    at = utc_timestamp(evaluated_at)
    if (type(references) is not tuple or not references
            or any(not isinstance(r, FundingCostReference) for r in references)
            or type(windows) is not tuple or any(not isinstance(w, FundingWindow) for w in windows)):
        raise TypeError("provide explicit reference and Funding window tuples")
    symbols = tuple(r.symbol for r in references)
    if (len(set(symbols)) != len(symbols) or len(windows) != len(references)
            or {w.symbol for w in windows} != set(symbols)):
        raise ValueError("unique same-symbol reference/history coverage required")
    bounds = {(utc_timestamp(w.opened_at), utc_timestamp(w.closed_at)) for w in windows}
    if len(bounds) != 1:
        raise ValueError("comparison requires identical query start/end, not differently dated histories")
    opened, closed = bounds.pop()
    if any(r.record_created_at > at for r in references):
        raise ValueError("future cost references are unavailable")
    if (type(adjustments) is not tuple or any(not isinstance(a, FundingAdjustment) for a in adjustments)
            or len({a.symbol for a in adjustments}) != len(adjustments)
            or any(a.symbol not in symbols or a.observed_at > at for a in adjustments)):
        raise ValueError("adjustments require unique same-symbol observed metadata")
    by_symbol = {w.symbol: w for w in windows}
    metadata = {a.symbol: a for a in adjustments}
    rows = tuple(FundingComparisonRow(reference, summarize_observed_funding(by_symbol[reference.symbol],
        reference_quantity=reference.quantity, reference_round_trip_costs=reference.round_trip_costs,
        lookback_windows=lookbacks, evaluated_at=at, research_instrument=reference.research_instrument),
        metadata.get(reference.symbol)) for reference in references)
    horizons = []
    for index, duration in enumerate(lookbacks):
        ranked = []
        unranked = []
        for row in rows:
            ratio = None if row.study.status is FundingStudyStatus.UNAVAILABLE else (
                row.study.slices[index].historical_funding_to_reference_cost_ratio)
            if ratio is None:
                unranked.append(row.reference.symbol)
            else:
                ranked.append((ratio, row.reference.symbol))
        # Unary negation must not round under a caller's low Decimal precision.
        ranked.sort(key=lambda pair: (pair[0].copy_negate(), pair[1]))
        horizons.append(FundingHorizonComparison(duration, tuple(s for _, s in ranked), tuple(unranked)))
    complete = sum(row.study.status is FundingStudyStatus.COMPLETE
                   and row.reference.round_trip_costs is not None and row.reference.round_trip_costs > 0
                   for row in rows)
    status = (FundingComparisonStatus.COMPLETE if complete == len(rows) else FundingComparisonStatus.PARTIAL
              if complete else FundingComparisonStatus.UNAVAILABLE)
    return HistoricalFundingComparison(status, opened, closed, at, rows, tuple(horizons))
