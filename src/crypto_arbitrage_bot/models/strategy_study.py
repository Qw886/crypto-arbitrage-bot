"""Fixed study criteria for research evidence, never permission to submit orders."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.analytics.summary import NumericSummary
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class StudyStatus(StrEnum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    READY_FOR_EXECUTION_REVIEW = "READY_FOR_EXECUTION_REVIEW"


class StudyReason(StrEnum):
    STUDY_NOT_FINISHED = "STUDY_NOT_FINISHED"
    INVALID_COHORT = "INVALID_COHORT"
    CONFIG_MISMATCH = "CONFIG_MISMATCH"
    NON_PRODUCTION_SOURCE = "NON_PRODUCTION_SOURCE"
    INVALID_ENTRY_EVIDENCE = "INVALID_ENTRY_EVIDENCE"
    NO_ELIGIBLE_ENTRY = "NO_ELIGIBLE_ENTRY"
    TOO_FEW_SEPARATED_ENTRIES = "TOO_FEW_SEPARATED_ENTRIES"
    TOO_FEW_ENTRY_DAYS = "TOO_FEW_ENTRY_DAYS"
    MISSING_PLANNED_EXIT_VALIDATION = "MISSING_PLANNED_EXIT_VALIDATION"
    AMBIGUOUS_PLANNED_EXIT_VALIDATION = "AMBIGUOUS_PLANNED_EXIT_VALIDATION"
    INVALID_VALIDATION_EVIDENCE = "INVALID_VALIDATION_EVIDENCE"
    NON_POSITIVE_OUTCOME_SUMMARY = "NON_POSITIVE_OUTCOME_SUMMARY"
    RETURN_BELOW_STUDY_THRESHOLD = "RETURN_BELOW_STUDY_THRESHOLD"
    WORST_LOSS_EXCEEDS_LIMIT = "WORST_LOSS_EXCEEDS_LIMIT"
    STUDY_RULES_MET = "STUDY_RULES_MET"
    MISSING_CAPTURE_RECEIPT = "MISSING_CAPTURE_RECEIPT"
    CAPTURE_RECEIPT_MISMATCH = "CAPTURE_RECEIPT_MISMATCH"
    ORPHAN_CAPTURE_RECEIPT = "ORPHAN_CAPTURE_RECEIPT"
    TASK_VALIDATION_CONFLICT = "TASK_VALIDATION_CONFLICT"
    PLANNED_TASK_NOT_COMPLETED = "PLANNED_TASK_NOT_COMPLETED"
    MISSING_ATTEMPT_COVERAGE = "MISSING_ATTEMPT_COVERAGE"
    CORRUPT_ATTEMPT_LOG = "CORRUPT_ATTEMPT_LOG"
    INCOMPLETE_ATTEMPT = "INCOMPLETE_ATTEMPT"
    ATTEMPT_RECORD_MISMATCH = "ATTEMPT_RECORD_MISMATCH"
    SAMPLING_GAP_TOO_LARGE = "SAMPLING_GAP_TOO_LARGE"


@dataclass(frozen=True)
class StrategyStudyProtocol:
    """Caller-injected criteria must be fixed before the evaluation cohort starts.

    Date assertions are not external timestamp attestations. A future registry
    must preserve the immutable registration and bind full strategy policy plus
    execution risk proof. Cost fingerprints below are existing saved record
    fingerprints; they alone do not capture all ScenarioSettings.
    """

    id: str
    registered_at: datetime
    study_start: datetime
    study_end: datetime
    symbols: tuple[str, ...]
    formula_version: str
    config_version: str
    config_fingerprints: tuple[tuple[str, str], ...]
    max_holding_period: timedelta
    min_entry_separation: timedelta
    max_exit_lateness: timedelta
    min_entries_per_symbol: int
    min_entry_days_per_symbol: int
    min_mean_net_return: Decimal
    min_median_net_return: Decimal
    max_loss_return: Decimal

    def __post_init__(self) -> None:
        for name in ("id", "formula_version", "config_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be explicit")
        for name in ("registered_at", "study_start", "study_end"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name)))
        if not self.registered_at <= self.study_start < self.study_end:
            raise ValueError("registration must precede the fixed study window")
        if (not isinstance(self.symbols, tuple) or not self.symbols
                or len(set(self.symbols)) != len(self.symbols)
                or any(not isinstance(s, str) or not s.endswith("USDT") for s in self.symbols)):
            raise ValueError("explicit distinct study symbols required")
        if not isinstance(self.config_fingerprints, tuple):
            raise TypeError("fixed fingerprints must be a tuple")
        fingerprints = dict(self.config_fingerprints)
        if len(fingerprints) != len(self.config_fingerprints) or set(fingerprints) != set(self.symbols):
            raise ValueError("exactly one fingerprint per study symbol required")
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
               for h in fingerprints.values()):
            raise ValueError("SHA256 fingerprints required")
        for name in ("max_holding_period", "min_entry_separation", "max_exit_lateness"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value < timedelta(0):
                raise ValueError("study durations must be nonnegative timedeltas")
        if self.max_holding_period <= timedelta(0) or self.min_entry_separation < (
            self.max_holding_period + self.max_exit_lateness
        ):
            raise ValueError("separated entry windows must not overlap the allowed exit window")
        for name in ("min_entries_per_symbol", "min_entry_days_per_symbol"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("sample requirements must be positive integers")
        if self.min_entry_days_per_symbol > self.min_entries_per_symbol:
            raise ValueError("minimum entry days cannot exceed minimum entries")
        for name in ("min_mean_net_return", "min_median_net_return", "max_loss_return"):
            require_decimal(getattr(self, name), name, nonnegative=True)


@dataclass(frozen=True)
class StudyIssue:
    code: StudyReason
    symbol: str | None = None
    opportunity_id: str | None = None


@dataclass(frozen=True)
class SymbolStudyEvidence:
    symbol: str
    eligible_entries: int
    selected_entry_ids: tuple[str, ...]
    selected_validation_ids: tuple[str, ...]
    entry_days: int
    profit: NumericSummary
    net_return: NumericSummary


@dataclass(frozen=True)
class StrategyStudyAssessment:
    protocol_id: str
    evaluated_at: datetime
    status: StudyStatus
    issues: tuple[StudyIssue, ...]
    symbols: tuple[SymbolStudyEvidence, ...]

    @property
    def reason_codes(self) -> tuple[StudyReason, ...]:
        return tuple(dict.fromkeys(issue.code for issue in self.issues))
