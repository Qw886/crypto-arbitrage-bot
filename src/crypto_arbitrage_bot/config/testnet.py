"""Non-sensitive testnet limits within the existing TOML configuration system."""

from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from crypto_arbitrage_bot.config.research import ResearchModel, StudyRegistrationSettings


class StrategyAdmissionSettings(ResearchModel):
    study: StudyRegistrationSettings
    execution_review: StudyRegistrationSettings


class TestnetSettings(ResearchModel):
    environment: Literal["TESTNET"]
    journal_path: Path
    max_order_notional: Decimal
    max_price_deviation: Decimal
    collateral_buffer: Decimal
    holding_seconds: int = Field(gt=0, le=86400, strict=True)
    max_active_pairs: int = Field(gt=0, le=2, strict=True)
    max_total_attempts: int = Field(gt=0, le=10000, strict=True)
    timeout_seconds: int = Field(gt=0, le=30, strict=True)
    max_exit_recovery_attempts: int = Field(default=0, ge=0, le=3, strict=True)
    max_recovery_price_deviation: Decimal = Decimal("0")
    funding_check_interval_seconds: int = Field(default=60, gt=0, le=3600, strict=True)
    funding_check_max_attempts: int = Field(default=5, gt=0, le=10, strict=True)
    balance_evidence_enabled: bool = Field(default=False, strict=True)
    balance_evidence_directory: Path | None = None
    balance_check_interval_seconds: int = Field(default=60, gt=0, le=3600, strict=True)
    balance_check_max_attempts: int = Field(default=3, gt=0, le=10, strict=True)
    balance_check_lease_seconds: int = Field(default=300, ge=30, le=3600, strict=True)
    admission: StrategyAdmissionSettings | None = None

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if not self.max_order_notional.is_finite() or self.max_order_notional <= 0:
            raise ValueError("max_order_notional must be positive")
        if not self.max_price_deviation.is_finite() or not 0 <= self.max_price_deviation <= Decimal("0.01"):
            raise ValueError("max_price_deviation must be in [0,0.01]")
        if not self.collateral_buffer.is_finite() or not 0 <= self.collateral_buffer <= 1:
            raise ValueError("collateral_buffer must be in [0,1]")
        if (not self.max_recovery_price_deviation.is_finite()
                or not 0 <= self.max_recovery_price_deviation <= Decimal("0.01")):
            raise ValueError("max_recovery_price_deviation must be in [0,0.01]")
        if self.balance_evidence_enabled and self.balance_check_lease_seconds < 40 * self.timeout_seconds:
            raise ValueError("balance check lease must cover bounded account/trade requests")
        return self
