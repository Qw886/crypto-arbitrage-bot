"""Independent evidence guard for normal automatic TESTNET entry only."""

from dataclasses import replace
from datetime import datetime
from decimal import Decimal, DecimalException

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.registered_study import StudyReadStatus, read_registered_study
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.risk_review import (
    RiskReviewError,
    _pinned,
    _sha,
    execution_policy_fingerprint,
    implementation_fingerprint,
    read_execution_risk_review,
)
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.strategy_admission import (
    AdmissionReason,
    AdmissionStatus,
    ExecutionRiskCase,
    StrategyAdmissionResult,
)
from crypto_arbitrage_bot.models.strategy_study import StudyStatus
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_registry import (
    StudyArchiveError,
    full_research_policy,
    load_registered_study,
)
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp

R = AdmissionReason


def review_automatic_entry(settings: Settings, record: OpportunityRecord, *, evaluated_at: datetime) -> StrategyAdmissionResult:
    """Market PASS remains necessary but cannot replace either evidence review.

    Production research and TESTNET execution use separate archives. The live
    testnet record is not inserted into the historic production study cohort.
    No network, credentials, fee/funding/PnL recalculation or order submission.
    """
    at = utc_timestamp(evaluated_at)
    result = StrategyAdmissionResult(AdmissionStatus.DENIED, record.symbol, record.id,
                                     record.matched_quantity if isinstance(record.matched_quantity, Decimal)
                                     and record.matched_quantity.is_finite() else None, at, ())
    if settings.testnet is None or settings.testnet.admission is None:
        return replace(result, reason_codes=(R.NOT_CONFIGURED,))
    config = settings.testnet.admission
    reasons = []
    risk = None
    try:
        try:
            if record.net_profit is None or record.net_return is None or record.matched_quantity is None:
                raise ValueError("missing admission metrics")
            require_decimal(record.net_profit, "admission estimate profit", positive=True)
            require_decimal(record.net_return, "admission estimate return", positive=True)
            require_decimal(record.matched_quantity, "admission quantity", positive=True)
            if (record.symbol not in settings.symbols or record.decision != "PASS" or record.data_quality_status != "VALID"
                    or record.snapshot_status != "COMPLETE" or record.created_at > at):
                raise ValueError("ineligible current estimate")
            entry = decode_entry(record)
            if (settings.research is None or not entry.opened_at <= record.created_at <= at
                    or not at-settings.research.data_quality.max_data_age <= entry.opened_at <= at
                    or not at-settings.research.data_quality.max_data_age <= record.snapshot_assembled_at <= at):
                raise ValueError("current opportunity is stale or time-inconsistent")
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, DecimalException):
            reasons.append(R.INVALID_OPPORTUNITY)
        study = load_registered_study(config.study.path, config.study.sha256)
        evidence = read_registered_study(config.study.path, config.study.sha256, evaluated_at=at)
        if evidence.status is not StudyReadStatus.READ or evidence.assessment is None:
            reasons.append(R.STUDY_UNAVAILABLE)
        elif evidence.assessment.status is not StudyStatus.READY_FOR_EXECUTION_REVIEW:
            reasons.append(R.STUDY_NOT_SUPPORTED)
        if (record.id in evidence.opportunity_ids or record.created_at < study.protocol.study_end
                + study.protocol.max_holding_period + study.protocol.max_exit_lateness):
            reasons.append(R.INVALID_OPPORTUNITY)
        if (study.policy_json != full_research_policy(settings) or record.formula_version != study.protocol.formula_version
                or record.config_version != study.protocol.config_version
                or record.config_fingerprint != dict(study.protocol.config_fingerprints).get(record.symbol)
                or _sha(record.config_json) != record.config_fingerprint):
            reasons.append(R.STUDY_POLICY_MISMATCH)
        if record.symbol not in study.protocol.symbols:
            reasons.append(R.SYMBOL_NOT_QUALIFIED)
    except (StudyArchiveError, OSError, ValueError, TypeError, KeyError, AttributeError, DecimalException):
        reasons.append(R.STUDY_UNAVAILABLE)
    try:
        risk = read_execution_risk_review(config.execution_review, settings, evaluated_at=at)
        limit = dict(risk.qualified_quantities).get(record.symbol)
        if limit is None:
            reasons.append(R.SYMBOL_NOT_QUALIFIED)
        elif (not isinstance(record.matched_quantity, Decimal) or not record.matched_quantity.is_finite()
                or record.matched_quantity > limit):
            reasons.append(R.QUANTITY_EXCEEDS_QUALIFICATION)
        cases = {(symbol, case) for symbol, case, _ in risk.cases}
        if any((record.symbol, case) not in cases for case in ExecutionRiskCase):
            reasons.append(R.RISK_REVIEW_INCOMPLETE)
        if not risk.reviewed_at <= at < risk.expires_at:
            reasons.append(R.RISK_REVIEW_EXPIRED)
        if risk.policy_sha256 != execution_policy_fingerprint(settings):
            reasons.append(R.RISK_POLICY_MISMATCH)
        if risk.implementation_sha256 != implementation_fingerprint():
            reasons.append(R.IMPLEMENTATION_MISMATCH)
    except RiskReviewError as exc:
        reasons.append(exc.reason)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, DecimalException):
        reasons.append(R.RISK_REVIEW_UNAVAILABLE)
    if reasons:
        return replace(result, reason_codes=tuple(dict.fromkeys(reasons)), study_sha256=config.study.sha256,
                       risk_review_sha256=config.execution_review.sha256)
    assert risk is not None and settings.research is not None
    return replace(result, status=AdmissionStatus.ELIGIBLE_FOR_TESTNET_PREFLIGHT,
                   reason_codes=(R.ALL_EVIDENCE_RULES_MET,), study_sha256=config.study.sha256,
                   risk_review_sha256=config.execution_review.sha256,
                   valid_until=min(risk.expires_at, at+settings.research.data_quality.max_data_age),
                   policy_sha256=risk.policy_sha256, implementation_sha256=risk.implementation_sha256,
                   opportunity_sha256=_sha(exact_json(record)))


def refresh_entry_admission(result: StrategyAdmissionResult, settings: Settings,
                            record: OpportunityRecord, *, evaluated_at: datetime) -> StrategyAdmissionResult:
    """Cheap post-preflight check; expensive historic reads occur before quoting.

    The existing plan/quote/quantity/deadline guards are still mandatory. This
    cannot turn a denial into approval, or be used to authorize a different entry.
    """
    at = utc_timestamp(evaluated_at)
    if not result.eligible:
        return result
    reasons = []
    try:
        config = None if settings.testnet is None else settings.testnet.admission
        if config is None:
            reasons.append(R.NOT_CONFIGURED)
        elif (result.study_sha256 != config.study.sha256 or result.risk_review_sha256 != config.execution_review.sha256):
            reasons.append(R.EVIDENCE_CHECK_FAILED)
        else:
            load_registered_study(config.study.path, config.study.sha256)
            _pinned(config.execution_review.path, config.execution_review.sha256)
        if (result.opportunity_id != record.id or result.symbol != record.symbol
                or result.quantity != record.matched_quantity or result.opportunity_sha256 != _sha(exact_json(record))):
            reasons.append(R.INVALID_OPPORTUNITY)
        if result.valid_until is None or not result.checked_at <= at < result.valid_until:
            reasons.append(R.RISK_REVIEW_EXPIRED)
        if result.policy_sha256 != execution_policy_fingerprint(settings):
            reasons.append(R.RISK_POLICY_MISMATCH)
        if result.implementation_sha256 != implementation_fingerprint():
            reasons.append(R.IMPLEMENTATION_MISMATCH)
    except (StudyArchiveError, RiskReviewError, OSError, ValueError, TypeError, KeyError, AttributeError, DecimalException):
        reasons.append(R.EVIDENCE_CHECK_FAILED)
    if reasons:
        return replace(result, status=AdmissionStatus.DENIED, reason_codes=tuple(dict.fromkeys(reasons)))
    return result
