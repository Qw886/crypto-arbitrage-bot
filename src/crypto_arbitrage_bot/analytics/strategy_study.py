"""Assess a complete supplied cohort using recorded simulated outcomes only.

No SQL, network, orders, PnL/fee/funding recalculation or selection by outcome.
This produces research evidence for review, never a trading approval.
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.summary import numeric_summary
from crypto_arbitrage_bot.analytics.validation_evidence import (
    read_validation_attribution,
    require_validation_amounts,
)
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.strategy_study import (
    StrategyStudyAssessment,
    StrategyStudyProtocol,
    StudyIssue,
    StudyReason,
    StudyStatus,
    SymbolStudyEvidence,
)
from crypto_arbitrage_bot.models.validation_record import ForwardValidationRecord
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp

R = StudyReason


def _production_sources(record: OpportunityRecord, registered_at: datetime) -> bool:
    endpoints = {"spot_depth": PublicEndpoint.SPOT_DEPTH.value, "perp_depth": PublicEndpoint.PERP_DEPTH.value,
                 "premium_index": PublicEndpoint.PERP_PREMIUM_INDEX.value,
                 "open_interest": PublicEndpoint.PERP_OPEN_INTEREST.value}
    if len(record.source_timings) != 4 or {s.source for s in record.source_timings} != set(endpoints):
        return False
    for source in record.source_timings:
        if source.request_started_at is None or utc_timestamp(source.request_started_at) < registered_at:
            return False
        request = json.loads(source.details_json)["request"]
        if (request.get("endpoint") != endpoints[source.source] or request.get("method", "GET") != "GET"
                or request.get("actual_url", endpoints[source.source]) != endpoints[source.source]
                or dict(request["params"]).get("symbol") != record.symbol):
            return False
    return True


def assess_strategy_study(
    protocol: StrategyStudyProtocol, opportunities: tuple[OpportunityRecord, ...],
    validations: tuple[ForwardValidationRecord, ...], *, evaluated_at: datetime,
) -> StrategyStudyAssessment:
    """Choose earliest PASS entries at fixed spacing BEFORE reading outcomes.

    Caller must supply the entire fixed cohort, including failed/missing exits;
    an independent read-only registry/reader is still required to attest cohort
    coverage, registration, full strategy parameters and execution risk proof.
    Each selected entry uses exactly one validation at its saved planned exit.
    Other horizons are retained by their caller but cannot improve this result.
    Separated entries reduce overlap; they do not establish statistical IID.
    """
    if not isinstance(protocol, StrategyStudyProtocol):
        raise TypeError("fixed StrategyStudyProtocol required")
    if (not isinstance(opportunities, tuple) or not isinstance(validations, tuple)
            or any(not isinstance(r, OpportunityRecord) for r in opportunities)
            or any(not isinstance(v, ForwardValidationRecord) for v in validations)):
        raise TypeError("immutable full opportunity/validation cohorts required")
    at = utc_timestamp(evaluated_at)
    issues: list[StudyIssue] = []
    def add(code: StudyReason, symbol: str | None = None, identifier: str | None = None) -> None:
        issue = StudyIssue(code, symbol, identifier)
        if issue not in issues:
            issues.append(issue)
    if at < protocol.study_end + protocol.max_holding_period + protocol.max_exit_lateness:
        add(R.STUDY_NOT_FINISHED)
    ids = {r.id for r in opportunities}
    if len(ids) != len(opportunities) or len({v.id for v in validations}) != len(validations):
        add(R.INVALID_COHORT)
    indexed: dict[str, list[ForwardValidationRecord]] = defaultdict(list)
    for v in validations:
        if v.opportunity_id not in ids:
            add(R.INVALID_COHORT, v.symbol, v.opportunity_id)
        indexed[v.opportunity_id].append(v)
    expected = dict(protocol.config_fingerprints)
    eligible: dict[str, list[tuple[OpportunityRecord, datetime, datetime]]] = defaultdict(list)
    for record in opportunities:
        try:
            created = utc_timestamp(record.created_at)
            if record.symbol not in protocol.symbols or not protocol.study_start <= created < protocol.study_end:
                add(R.INVALID_COHORT, record.symbol, record.id)
                continue
            if (record.formula_version != protocol.formula_version or record.config_version != protocol.config_version
                    or record.config_fingerprint != expected[record.symbol]
                    or hashlib.sha256(record.config_json.encode("utf-8")).hexdigest() != record.config_fingerprint):
                add(R.CONFIG_MISMATCH, record.symbol, record.id)
                continue
            if not _production_sources(record, protocol.registered_at):
                add(R.NON_PRODUCTION_SOURCE, record.symbol, record.id)
                continue
            # Non-PASS research is retained in the supplied cohort, not relabeled
            # retroactively because some later horizon happened to be positive.
            if record.decision != "PASS":
                continue
            if (record.net_profit is None or record.net_return is None
                    or record.net_profit <= 0 or record.net_return <= 0):
                raise ValueError("PASS entry must retain positive estimate metrics")
            entry = decode_entry(record)
            opened = entry.opened_at
            closed = utc_timestamp(datetime.fromisoformat(json.loads(record.details_json)["estimate"]["scenario"]["closed_at"]))
            if not protocol.study_start <= opened <= created or not timedelta(0) < closed-opened <= protocol.max_holding_period:
                raise ValueError("entry time/window differs from protocol")
            eligible[record.symbol].append((record, opened, closed))
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, DecimalException):
            add(R.INVALID_ENTRY_EVIDENCE, record.symbol, record.id)
    summaries = []
    for symbol in protocol.symbols:
        rows = sorted(eligible[symbol], key=lambda row: (row[1], row[0].id))
        selected: list[tuple[OpportunityRecord, datetime, datetime]] = []
        for row in rows:
            if not selected or row[1]-selected[-1][1] >= protocol.min_entry_separation:
                selected.append(row)
        if not rows:
            add(R.NO_ELIGIBLE_ENTRY, symbol)
        if len(selected) < protocol.min_entries_per_symbol:
            add(R.TOO_FEW_SEPARATED_ENTRIES, symbol)
        days = len({row[1].date() for row in selected})
        if days < protocol.min_entry_days_per_symbol:
            add(R.TOO_FEW_ENTRY_DAYS, symbol)
        profits: list[Decimal | None] = []
        returns: list[Decimal | None] = []
        validation_ids = []
        for record, opened, closed in selected:
            candidates = []
            for v in indexed[record.id]:
                try:
                    if v.symbol != symbol or v.entry_time != opened:
                        raise ValueError("validation identity differs from original entry")
                    when = None if v.validation_time is None else utc_timestamp(v.validation_time)
                    if when is not None and closed <= when <= closed+protocol.max_exit_lateness:
                        candidates.append(v)
                except (ValueError, TypeError):
                    add(R.INVALID_VALIDATION_EVIDENCE, symbol, record.id)
            if len(candidates) != 1:
                add(R.MISSING_PLANNED_EXIT_VALIDATION if not candidates else
                    R.AMBIGUOUS_PLANNED_EXIT_VALIDATION, symbol, record.id)
                profits.append(None)
                returns.append(None)
                continue
            v = candidates[0]
            try:
                entry = decode_entry(record)
                if (v.status != "VALIDATED" or v.reason_codes or v.validation_time is None
                        or utc_timestamp(v.validation_time) > at or v.created_at > at
                        or not v.validation_time <= v.created_at
                        or v.holding_duration != v.validation_time-opened
                        or v.matched_quantity != record.matched_quantity or v.capital_occupied != record.capital_occupied
                        or v.original_entry_basis != entry.entry_basis
                        or v.spot_entry_fee != entry.spot_fee or v.perp_entry_fee != entry.perp_fee
                        or v.other_costs != entry.other_costs
                        or v.original_estimated_net_profit != record.net_profit
                        or v.original_estimated_net_return != record.net_return):
                    raise ValueError("validation is not a complete matching exit")
                if v.validated_simulated_net_profit is None or v.validated_simulated_net_return is None:
                    raise ValueError("missing validated outcome")
                profit = require_decimal(v.validated_simulated_net_profit, "validated profit")
                rate = require_decimal(v.validated_simulated_net_return, "validated return")
                require_validation_amounts(v)
                read_validation_attribution(v)
                profits.append(profit)
                returns.append(rate)
                validation_ids.append(v.id)
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError, DecimalException):
                add(R.INVALID_VALIDATION_EVIDENCE, symbol, record.id)
                profits.append(None)
                returns.append(None)
        profit_summary, return_summary = numeric_summary(profits), numeric_summary(returns)
        summaries.append(SymbolStudyEvidence(symbol, len(rows), tuple(r.id for r, _, _ in selected),
                         tuple(validation_ids), days, profit_summary, return_summary))
        if return_summary.count:
            assert return_summary.mean is not None and return_summary.median is not None
            assert return_summary.minimum is not None
            if return_summary.mean <= 0 or return_summary.median <= 0:
                add(R.NON_POSITIVE_OUTCOME_SUMMARY, symbol)
            if return_summary.mean < protocol.min_mean_net_return or return_summary.median < protocol.min_median_net_return:
                add(R.RETURN_BELOW_STUDY_THRESHOLD, symbol)
            if return_summary.minimum < protocol.max_loss_return.copy_negate():
                add(R.WORST_LOSS_EXCEEDS_LIMIT, symbol)
    economic = {R.NON_POSITIVE_OUTCOME_SUMMARY, R.RETURN_BELOW_STUDY_THRESHOLD, R.WORST_LOSS_EXCEEDS_LIMIT}
    status = StudyStatus.INSUFFICIENT_EVIDENCE if any(i.code not in economic for i in issues) else (
        StudyStatus.UNSUPPORTED if issues else StudyStatus.READY_FOR_EXECUTION_REVIEW)
    if not issues:
        issues.append(StudyIssue(R.STUDY_RULES_MET))
    return StrategyStudyAssessment(protocol.id, at, status, tuple(issues), tuple(summaries))
