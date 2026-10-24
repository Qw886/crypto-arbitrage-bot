"""Injected scenario risk limits and research decisions with source diagnostics."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.models.decision import Decision, DecisionConfig, DecisionReason
from crypto_arbitrage_bot.models.scenario import ScenarioResult, ScenarioType
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


@dataclass(frozen=True)
class ScenarioDecisionConfig:
    """All limits are explicit: amounts in USDT, returns as decimal fractions.

    decision supplies the existing baseline hard and higher PASS thresholds.
    Its non-profit safety checks also apply to every other scenario. Loss
    limits are nonnegative magnitudes, not signed returns. Scenario margin is
    the worst baseline net profit minus the largest stress loss in USDT;
    it is a risk comparison, never a new PnL or another cost deduction.
    """

    decision: DecisionConfig
    min_baseline_net_profit: Decimal
    min_baseline_net_return: Decimal
    max_stress_loss: Decimal
    max_stress_loss_return: Decimal
    min_convergence_profit: Decimal
    min_scenario_margin: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.decision, DecisionConfig):
            raise TypeError("decision must be DecisionConfig")
        for name in (
            "min_baseline_net_profit", "min_baseline_net_return", "max_stress_loss",
            "max_stress_loss_return", "min_convergence_profit", "min_scenario_margin",
        ):
            require_decimal(getattr(self, name), name, nonnegative=True)


class ScenarioDecisionReason(StrEnum):
    BASELINE_NON_POSITIVE = "BASELINE_NON_POSITIVE"
    BASELINE_BELOW_THRESHOLD = "BASELINE_BELOW_THRESHOLD"
    STRESS_LOSS_TOO_LARGE = "STRESS_LOSS_TOO_LARGE"
    STRESS_RETURN_TOO_LOW = "STRESS_RETURN_TOO_LOW"
    CONVERGENCE_PROFIT_INSUFFICIENT = "CONVERGENCE_PROFIT_INSUFFICIENT"
    SCENARIO_MARGIN_INSUFFICIENT = "SCENARIO_MARGIN_INSUFFICIENT"
    SCENARIO_DATA_INCOMPLETE = "SCENARIO_DATA_INCOMPLETE"
    SCENARIO_COMPARISON_MISMATCH = "SCENARIO_COMPARISON_MISMATCH"
    SCENARIO_MARGIN_UNAVAILABLE = "SCENARIO_MARGIN_UNAVAILABLE"
    SCENARIO_PASS_ALL_RULES = "SCENARIO_PASS_ALL_RULES"


@dataclass(frozen=True)
class ScenarioDecisionIssue:
    code: ScenarioDecisionReason | DecisionReason
    scenario_name: str | None = None
    scenario_type: ScenarioType | None = None


@dataclass(frozen=True)
class ScenarioDecisionResult:
    """Original scenario evidence is retained, with no trading authorization.

    PASS only means worth further simulation/recording. scenario_margin is
    unavailable when the supplied results are incomplete or incomparable.
    """

    status: Decision
    issues: tuple[ScenarioDecisionIssue, ...]
    scenarios: tuple[ScenarioResult, ...]
    config: ScenarioDecisionConfig
    scenario_margin: Decimal | None

    @property
    def reason_codes(self) -> tuple[ScenarioDecisionReason | DecisionReason, ...]:
        return tuple(dict.fromkeys(issue.code for issue in self.issues))
