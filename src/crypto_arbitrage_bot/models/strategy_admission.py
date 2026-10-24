"""Normal TESTNET entry evidence, independent of market PASS and PnL."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class AdmissionStatus(StrEnum):
    DENIED = "DENIED"
    ELIGIBLE_FOR_TESTNET_PREFLIGHT = "ELIGIBLE_FOR_TESTNET_PREFLIGHT"


class AdmissionReason(StrEnum):
    NOT_CONFIGURED = "ADMISSION_NOT_CONFIGURED"
    INVALID_OPPORTUNITY = "INVALID_ADMISSION_OPPORTUNITY"
    STUDY_UNAVAILABLE = "STUDY_EVIDENCE_UNAVAILABLE"
    STUDY_NOT_SUPPORTED = "STUDY_NOT_SUPPORTED"
    STUDY_POLICY_MISMATCH = "STUDY_POLICY_MISMATCH"
    SYMBOL_NOT_QUALIFIED = "SYMBOL_NOT_QUALIFIED"
    RISK_REVIEW_UNAVAILABLE = "EXECUTION_RISK_REVIEW_UNAVAILABLE"
    RISK_REVIEW_INCOMPLETE = "EXECUTION_RISK_REVIEW_INCOMPLETE"
    RISK_REVIEW_EXPIRED = "EXECUTION_RISK_REVIEW_EXPIRED"
    RISK_POLICY_MISMATCH = "EXECUTION_RISK_POLICY_MISMATCH"
    IMPLEMENTATION_MISMATCH = "EXECUTION_IMPLEMENTATION_MISMATCH"
    QUANTITY_EXCEEDS_QUALIFICATION = "QUANTITY_EXCEEDS_QUALIFICATION"
    EVIDENCE_CHECK_FAILED = "ADMISSION_EVIDENCE_CHECK_FAILED"
    ALL_EVIDENCE_RULES_MET = "ALL_ADMISSION_EVIDENCE_RULES_MET"


class ExecutionRiskCase(StrEnum):
    NORMAL_ENTRY_EXIT = "NORMAL_ENTRY_EXIT"
    SPOT_ACK_RECOVERY = "SPOT_ACK_RECOVERY"
    PERP_UNKNOWN_RECOVERY = "PERP_UNKNOWN_RECOVERY"
    PARTIAL_ENTRY_RECOVERY = "PARTIAL_ENTRY_RECOVERY"
    PARTIAL_EXIT_RECOVERY = "PARTIAL_EXIT_RECOVERY"
    RESTART_IDEMPOTENCY = "RESTART_IDEMPOTENCY"
    FEES_FUNDING_BALANCE = "FEES_FUNDING_BALANCE"


@dataclass(frozen=True)
class ExecutionRiskReview:
    sha256: str
    policy_sha256: str
    implementation_sha256: str
    reviewed_at: datetime
    expires_at: datetime
    qualified_quantities: tuple[tuple[str, Decimal], ...]
    cases: tuple[tuple[str, ExecutionRiskCase, str], ...]

    def __post_init__(self) -> None:
        for name in ("sha256", "policy_sha256", "implementation_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("explicit SHA256 evidence bindings required")
        for name in ("reviewed_at", "expires_at"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name)))
        if self.reviewed_at >= self.expires_at:
            raise ValueError("explicit finite risk review validity required")
        if (not isinstance(self.qualified_quantities, tuple)
                or len(dict(self.qualified_quantities)) != len(self.qualified_quantities)):
            raise ValueError("distinct qualified symbol quantities required")
        for symbol, quantity in self.qualified_quantities:
            if symbol not in ("BTCUSDT", "ETHUSDT"):
                raise ValueError("unsupported TESTNET qualification symbol")
            require_decimal(quantity, "qualified quantity", positive=True)
        if not isinstance(self.cases, tuple) or any(not isinstance(case, ExecutionRiskCase) or not identifier
                                                   for _, case, identifier in self.cases):
            raise ValueError("typed execution case references required")


@dataclass(frozen=True)
class StrategyAdmissionResult:
    status: AdmissionStatus
    symbol: str
    opportunity_id: str
    quantity: Decimal | None
    checked_at: datetime
    reason_codes: tuple[AdmissionReason, ...]
    study_sha256: str | None = None
    risk_review_sha256: str | None = None
    valid_until: datetime | None = None
    policy_sha256: str | None = None
    implementation_sha256: str | None = None
    opportunity_sha256: str | None = None

    @property
    def eligible(self) -> bool:
        return self.status is AdmissionStatus.ELIGIBLE_FOR_TESTNET_PREFLIGHT
