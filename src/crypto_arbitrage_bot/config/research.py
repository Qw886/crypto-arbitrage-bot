"""Research runner section of the existing TOML/CAB_ settings system."""

import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crypto_arbitrage_bot.arbitrage.fees import FeeRates
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.sizing import QuantityRules
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig
from crypto_arbitrage_bot.models.estimate import EstimateConfig
from crypto_arbitrage_bot.models.scenario_decision import ScenarioDecisionConfig


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def reject_float(cls, value: Any) -> Any:
        def check(item: Any) -> None:
            if isinstance(item, float):
                raise ValueError("research settings require decimal strings, never floats")
            if isinstance(item, dict):
                for child in item.values():
                    check(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    check(child)
        check(value)
        return value


class StudyRegistrationSettings(ResearchModel):
    """Pinned preregistration reference; no trading enablement or default policy."""

    path: Path
    sha256: str

    @field_validator("sha256")
    @classmethod
    def pinned_digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("pinned lowercase SHA256 required")
        return value


class SizingSettings(ResearchModel):
    spot_rules: QuantityRules
    perp_rules: QuantityRules
    target_quantity: Decimal | None = None
    spot_budget: Decimal | None = None
    capital_limit: Decimal | None = None
    stress_reserve: Decimal
    other_costs: Decimal

    def as_estimate(self, fees: FeeRates, quality: DataQualityConfig) -> EstimateConfig:
        return EstimateConfig(self.spot_rules, self.perp_rules, fees, self.target_quantity,
                              self.spot_budget, self.capital_limit, self.stress_reserve,
                              self.other_costs, quality)


class CandidateCatalogSettings(ResearchModel):
    """Public metadata research filters; never an execution symbol whitelist."""

    min_perp_age_seconds: int = Field(ge=0, strict=True)
    min_remaining_lifetime_seconds: int = Field(gt=0, strict=True)
    max_metadata_age_seconds: int = Field(gt=0, le=3600, strict=True)
    max_sampling_skew_seconds: int = Field(ge=0, le=60, strict=True)
    blocked_assets: tuple[str, ...] = ()

    @field_validator("min_perp_age_seconds", "min_remaining_lifetime_seconds",
                     "max_metadata_age_seconds", "max_sampling_skew_seconds", mode="before")
    @classmethod
    def environment_integer(cls, value: Any) -> Any:
        # Existing nested CAB_ environment sources supply scalar strings.
        # No float, bool, unit suffix or fractional conversion is accepted.
        if type(value) is str and re.fullmatch(r"0|[1-9][0-9]*", value):
            return int(value)
        return value

    @model_validator(mode="after")
    def validate_assets(self) -> Self:
        if (len(self.blocked_assets) != len(set(self.blocked_assets))
                or any(not re.fullmatch(r"[A-Z0-9]{1,32}", asset) for asset in self.blocked_assets)):
            raise ValueError("blocked assets must be unique uppercase ASCII asset identifiers")
        return self


class FundingProjectionSettings(ResearchModel):
    """One hypothetical future settlement, never actual funding history."""

    mode: Literal["next_settlement"]
    positive_rate_multiplier: Decimal
    max_wait_seconds: int = Field(gt=0, le=86400, strict=True)
    exit_delay_seconds: int = Field(gt=0, le=3600, strict=True)
    min_entry_lead_seconds: int = Field(gt=0, le=3600, strict=True)
    entry_window_seconds: int = Field(gt=0, le=86400, strict=True)

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if (not self.positive_rate_multiplier.is_finite()
                or not 0 <= self.positive_rate_multiplier <= 1):
            raise ValueError("positive funding multiplier must be in [0,1]")
        if not self.min_entry_lead_seconds < self.entry_window_seconds <= self.max_wait_seconds:
            raise ValueError("entry window must exceed minimum lead and not exceed maximum wait")
        return self


class ScenarioSettings(ResearchModel):
    baseline: Literal["reverse_execution"]
    holding_seconds: int = Field(gt=0, strict=True)
    spot_exit_mid_mode: Literal["entry_mid", "explicit"]
    spot_exit_mid: Decimal | None = None
    convergence_exit_basis_pct: Decimal
    adverse_exit_basis_pct: Decimal
    funding_events: tuple[FundingEvent, ...]  # () explicitly assumes no settlements.
    funding_projection: FundingProjectionSettings | None = None
    adverse_exit_basis_mode: Literal["absolute", "entry_offset"] = "absolute"

    @model_validator(mode="after")
    def validate_assumptions(self) -> Self:
        if self.funding_projection is not None and self.funding_events:
            raise ValueError("explicit settlements and future funding projection cannot be mixed")
        if self.adverse_exit_basis_mode == "entry_offset" and self.adverse_exit_basis_pct <= 0:
            raise ValueError("adverse entry basis widening must be positive")
        if self.adverse_exit_basis_mode == "entry_offset" and self.funding_projection is None:
            raise ValueError("entry-offset stress mode requires an explicit funding projection")
        if self.spot_exit_mid_mode == "explicit":
            if self.spot_exit_mid is None or not self.spot_exit_mid.is_finite() or self.spot_exit_mid <= 0:
                raise ValueError("explicit exit midpoint must be positive")
        elif self.spot_exit_mid is not None:
            raise ValueError("entry_mid mode must not also supply spot_exit_mid")
        for rate in (self.convergence_exit_basis_pct, self.adverse_exit_basis_pct):
            if not rate.is_finite() or rate <= -1:
                raise ValueError("exit basis fraction must imply a positive price")
        return self


class ValidationSchedulerSettings(ResearchModel):
    # Explicit integer seconds: equivalent to a list of duration strings, without
    # ambiguous units or floating-point duration conversion.
    validation_horizons_seconds: tuple[int, ...]
    batch_size: int = Field(default=100, gt=0, le=1000, strict=True)
    lease_seconds: int = Field(default=300, gt=0, strict=True)
    max_attempts: int = Field(default=3, gt=0, le=10, strict=True)
    funding_page_limit: int = Field(default=1000, gt=0, le=1000, strict=True)
    funding_max_pages: int = Field(default=10, gt=0, le=100, strict=True)
    # None preserves legacy configurations; long-running research opts into a
    # deadline so restart catch-up cannot masquerade as an intended horizon.
    max_lateness_seconds: int | None = Field(default=None, ge=0, le=3600, strict=True)
    funding_time_tolerance_ms: int = Field(default=1000, ge=0, le=5000, strict=True)

    @model_validator(mode="before")
    @classmethod
    def validate_horizons(cls, value: Any) -> Any:
        if isinstance(value, dict):
            horizons = value.get("validation_horizons_seconds")
            if (not isinstance(horizons, (tuple, list)) or not horizons
                    or any(type(item) is not int or item <= 0 for item in horizons)
                    or len(set(horizons)) != len(horizons)):
                raise ValueError("validation horizons must be distinct positive integer seconds")
        return value


class CandidateEconomicsSettings(ResearchModel):
    """Explicit public-research budgets and named holding hypotheses, no trading."""

    spot_budget: Decimal
    capital_limit: Decimal | None = None
    stress_reserve: Decimal
    other_costs: Decimal
    windows: dict[str, ScenarioSettings]

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if not self.spot_budget.is_finite() or self.spot_budget <= 0:
            raise ValueError("candidate spot budget must be positive")
        if self.capital_limit is not None and (not self.capital_limit.is_finite() or self.capital_limit <= 0):
            raise ValueError("candidate capital limit must be positive")
        if any(not v.is_finite() or v < 0 for v in (self.stress_reserve, self.other_costs)):
            raise ValueError("candidate reserves/costs must be nonnegative")
        if (not 1 <= len(self.windows) <= 16 or any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) or not 0 < profile.holding_seconds <= 604800
            for name, profile in self.windows.items()
        )):
            raise ValueError("provide 1..16 named candidate windows, each no longer than seven days")
        return self


class FundingScreenSettings(ResearchModel):
    """Preselection budget only; never replaces financial/risk Decision thresholds."""

    max_candidates: int = Field(ge=3, le=10, strict=True)
    benchmark_symbols: tuple[Literal["BTCUSDT", "ETHUSDT"], ...] = Field(min_length=2, max_length=2)
    min_observed_funding_rate: Decimal
    max_next_settlement_wait_seconds: int = Field(gt=0, le=86400, strict=True)

    @model_validator(mode="after")
    def validate_screen(self) -> Self:
        if (not self.min_observed_funding_rate.is_finite() or self.min_observed_funding_rate < 0
                or len(set(self.benchmark_symbols)) != len(self.benchmark_symbols)):
            raise ValueError("screen rate must be finite/nonnegative and benchmarks distinct")
        return self


class ResearchSettings(ResearchModel):
    sqlite_path: Path
    sizing: dict[Literal["BTCUSDT", "ETHUSDT"], SizingSettings]
    scenarios: dict[Literal["BTCUSDT", "ETHUSDT"], ScenarioSettings]
    decision: ScenarioDecisionConfig
    data_quality: DataQualityConfig
    timeout_seconds: int = Field(gt=0, le=60, strict=True)
    formula_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    validation_scheduler: ValidationSchedulerSettings | None = None
    candidate_catalog: CandidateCatalogSettings | None = None
    candidate_economics: CandidateEconomicsSettings | None = None
    funding_screen: FundingScreenSettings | None = None
    study_registration: StudyRegistrationSettings | None = None
