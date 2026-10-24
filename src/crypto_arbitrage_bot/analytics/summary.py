"""Descriptive statistics of recorded research estimates, never realized returns."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.scenario import ScenarioType
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


@dataclass(frozen=True)
class RecordedScenario:
    scenario_type: ScenarioType
    net_profit: Decimal | None
    net_return: Decimal | None


@dataclass(frozen=True)
class RecordedObservation:
    id: str
    created_at: datetime
    symbol: str
    decision: Decision
    net_profit: Decimal | None
    net_return: Decimal | None
    funding_rate: Decimal | None
    data_quality_status: str | None
    decision_reason_codes: tuple[str, ...]
    data_quality_reason_codes: tuple[str, ...]
    scenarios: tuple[RecordedScenario, ...]


@dataclass(frozen=True)
class NumericSummary:
    count: int
    null_count: int
    minimum: Decimal | None
    maximum: Decimal | None
    mean: Decimal | None
    median: Decimal | None


@dataclass(frozen=True)
class Frequency:
    value: str
    count: int


@dataclass(frozen=True)
class DecisionFrequency:
    decision: Decision
    count: int
    proportion: Decimal | None


@dataclass(frozen=True)
class FundingCounts:
    positive: int
    zero: int
    negative: int
    missing: int


@dataclass(frozen=True)
class ScenarioSummary:
    scenario_type: ScenarioType
    count: int
    opportunity_count: int
    missing_opportunity_count: int
    net_profit: NumericSummary
    net_return: NumericSummary


@dataclass(frozen=True)
class ResearchSummary:
    total_opportunities: int
    decisions: tuple[DecisionFrequency, ...]
    symbols: tuple[Frequency, ...]
    net_profit: NumericSummary
    net_return: NumericSummary
    scenarios: tuple[ScenarioSummary, ...]
    funding_rates: FundingCounts
    invalid_data_quality_count: int
    missing_data_quality_count: int
    decision_reasons: tuple[Frequency, ...]
    data_quality_reasons: tuple[Frequency, ...]
    earliest_created_at: datetime | None
    latest_created_at: datetime | None
    interpretation: str = "Recorded research estimates only; not realized PnL or a strategy win rate."


def numeric_summary(values: Iterable[Decimal | None]) -> NumericSummary:
    """Skip/count NULLs. Exact decimal sums, rounded division at >=50 digits.

    Working precision grows with input exponent span and sample count to avoid
    losing small terms during summation/cancellation. No binary float or SQL AVG.
    """
    present = []
    missing = 0
    for value in values:
        if value is None:
            missing += 1
        else:
            present.append(require_decimal(value, "recorded value"))
    if not present:
        return NumericSummary(0, missing, None, None, None, None)
    present.sort()
    count = len(present)
    precision = max(50, max(value.adjusted() for value in present)
                    - min(int(value.as_tuple().exponent) for value in present)
                    + len(str(count)) + 3)
    with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN)):
        mean = sum(present, Decimal(0)) / Decimal(count)
        middle = count // 2
        median = (present[middle] if count % 2 else
                  (present[middle - 1] + present[middle]) / Decimal(2))
    return NumericSummary(count, missing, present[0], present[-1], mean, median)


def _frequencies(values: Iterable[str]) -> tuple[Frequency, ...]:
    return tuple(Frequency(value, count) for value, count in
                 sorted(Counter(values).items(), key=lambda item: (-item[1], item[0])))


def summarize_observations(observations: Iterable[RecordedObservation]) -> ResearchSummary:
    """Count saved observations without reclassifying or filtering decisions.

    Each saved scenario variant is one sample. Reason frequency counts records
    containing the code (duplicates within a record count once).
    """
    rows = tuple(observations)
    total = len(rows)
    decisions = Counter(row.decision for row in rows)
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        decision_counts = tuple(DecisionFrequency(
            decision, decisions[decision],
            Decimal(decisions[decision]) / Decimal(total) if total else None,
        ) for decision in Decision)
    scenario_summaries = []
    for kind in ScenarioType:
        samples = tuple(item for row in rows for item in row.scenarios if item.scenario_type is kind)
        containing = sum(any(item.scenario_type is kind for item in row.scenarios) for row in rows)
        scenario_summaries.append(ScenarioSummary(
            kind, len(samples), containing, total - containing,
            numeric_summary(item.net_profit for item in samples),
            numeric_summary(item.net_return for item in samples),
        ))
    rates = [require_decimal(row.funding_rate, "funding_rate")
             for row in rows if row.funding_rate is not None]
    times = [row.created_at for row in rows]
    return ResearchSummary(
        total, decision_counts, _frequencies(row.symbol for row in rows),
        numeric_summary(row.net_profit for row in rows),
        numeric_summary(row.net_return for row in rows), tuple(scenario_summaries),
        FundingCounts(sum(rate > 0 for rate in rates), sum(rate == 0 for rate in rates),
                      sum(rate < 0 for rate in rates), total - len(rates)),
        sum(row.data_quality_status == "INVALID" for row in rows),
        sum(row.data_quality_status is None for row in rows),
        _frequencies(code for row in rows for code in set(row.decision_reason_codes)),
        _frequencies(code for row in rows for code in set(row.data_quality_reason_codes)),
        min(times) if times else None, max(times) if times else None,
    )
