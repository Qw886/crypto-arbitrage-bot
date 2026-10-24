"""Pure multi-scenario decisions over existing estimates; no PnL recalculation."""

from decimal import Decimal, DecimalException, Inexact, localcontext

from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.models.decision import Decision, DecisionReason
from crypto_arbitrage_bot.models.scenario import ScenarioResult, ScenarioType
from crypto_arbitrage_bot.models.scenario_decision import (
    ScenarioDecisionConfig,
    ScenarioDecisionIssue,
    ScenarioDecisionReason,
    ScenarioDecisionResult,
)
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation

# Only these economic rules are baseline-specific. All other 7A failures
# remain hard failures in every role, including negative-profit stress cases.
BASELINE_PROFIT_RULES = frozenset({
    DecisionReason.NON_POSITIVE_NET_PROFIT,
    DecisionReason.NON_POSITIVE_NET_RETURN,
    DecisionReason.NET_PROFIT_BELOW_THRESHOLD,
    DecisionReason.NET_RETURN_BELOW_THRESHOLD,
    DecisionReason.PROFIT_BUFFER_INSUFFICIENT,
})
HIGHER_PASS_RULES = frozenset({
    DecisionReason.PASS_NET_PROFIT_BELOW_TARGET,
    DecisionReason.PASS_NET_RETURN_BELOW_TARGET,
})
INVALID_RESULT_RULES = frozenset({
    DecisionReason.INVALID_MARKET_DATA,
    DecisionReason.INCOMPLETE_ESTIMATE,
    DecisionReason.INVALID_ESTIMATE_VALUES,
    DecisionReason.INCONSISTENT_EXECUTION,
    DecisionReason.INSUFFICIENT_DEPTH,
})
ZERO = Decimal("0")


def _finite(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite()


def _loss(value: Decimal) -> Decimal:
    return value.copy_negate() if value < ZERO else ZERO


@decimal_calculation
def decide_scenarios(
    scenarios: tuple[ScenarioResult, ...], config: ScenarioDecisionConfig,
) -> ScenarioDecisionResult:
    """Require all three roles and consider every supplied result within each.

    No network, clock, scenario generation, funding or profit calculation.
    Baselines reuse 7A unchanged, including its higher PASS targets. Other
    roles reuse its data/execution/size/delta guards, while their economic
    outcomes use the explicit convergence target and stress loss limits.
    """
    if not isinstance(config, ScenarioDecisionConfig):
        raise TypeError("config must be ScenarioDecisionConfig")
    if not isinstance(scenarios, tuple) or any(
        not isinstance(item, ScenarioResult) for item in scenarios
    ):
        raise TypeError("scenarios must be a tuple of ScenarioResult")

    rejects: list[ScenarioDecisionIssue] = []
    observations: list[ScenarioDecisionIssue] = []
    evidence_valid = True
    by_type = {kind: tuple(item for item in scenarios if item.scenario_type is kind)
               for kind in ScenarioType}
    if any(not items for items in by_type.values()):
        rejects.append(ScenarioDecisionIssue(ScenarioDecisionReason.SCENARIO_DATA_INCOMPLETE))
        evidence_valid = False
    if len({item.name for item in scenarios}) != len(scenarios):
        rejects.append(ScenarioDecisionIssue(ScenarioDecisionReason.SCENARIO_COMPARISON_MISMATCH))
        evidence_valid = False

    baseline_profits: list[Decimal] = []
    stress_losses: list[Decimal] = []
    for item in scenarios:
        source = (item.name, item.scenario_type)
        single = decide(item.estimate, config.decision)
        invalid = bool(INVALID_RESULT_RULES.intersection(single.reason_codes))
        if invalid:
            rejects.append(ScenarioDecisionIssue(
                ScenarioDecisionReason.SCENARIO_DATA_INCOMPLETE, *source,
            ))
            evidence_valid = False
        if item.definition.exit_scenario != item.estimate.scenario:
            rejects.append(ScenarioDecisionIssue(
                ScenarioDecisionReason.SCENARIO_COMPARISON_MISMATCH, *source,
            ))
            evidence_valid = False

        for reason in single.reason_codes:
            if reason is DecisionReason.PASS_ALL_RULES:
                continue
            if item.scenario_type is ScenarioType.NO_CONVERGENCE:
                destination = observations if reason in HIGHER_PASS_RULES else rejects
                destination.append(ScenarioDecisionIssue(reason, *source))
            elif reason not in BASELINE_PROFIT_RULES | HIGHER_PASS_RULES:
                rejects.append(ScenarioDecisionIssue(reason, *source))

        profit, net_return = item.net_profit, item.net_return
        if not _finite(profit) or not _finite(net_return):
            continue  # Already rejected by the existing 7A validation.
        assert profit is not None and net_return is not None
        if item.scenario_type is ScenarioType.NO_CONVERGENCE:
            baseline_profits.append(profit)
            if profit <= ZERO or net_return <= ZERO:
                rejects.append(ScenarioDecisionIssue(
                    ScenarioDecisionReason.BASELINE_NON_POSITIVE, *source,
                ))
            if (
                profit < config.min_baseline_net_profit
                or net_return < config.min_baseline_net_return
            ):
                rejects.append(ScenarioDecisionIssue(
                    ScenarioDecisionReason.BASELINE_BELOW_THRESHOLD, *source,
                ))
        elif item.scenario_type is ScenarioType.ADVERSE:
            stress_losses.append(_loss(profit))
            if _loss(profit) > config.max_stress_loss:
                rejects.append(ScenarioDecisionIssue(
                    ScenarioDecisionReason.STRESS_LOSS_TOO_LARGE, *source,
                ))
            if _loss(net_return) > config.max_stress_loss_return:
                rejects.append(ScenarioDecisionIssue(
                    ScenarioDecisionReason.STRESS_RETURN_TOO_LOW, *source,
                ))
        elif profit < config.min_convergence_profit:
            observations.append(ScenarioDecisionIssue(
                ScenarioDecisionReason.CONVERGENCE_PROFIT_INSUFFICIENT, *source,
            ))

    # A comparison must concern the same entry exposure and cost configuration.
    # Closing times and explicit funding/exit hypotheses are intentionally free
    # to differ, as supported by the existing Scenario Engine.
    if evidence_valid:
        reference = by_type[ScenarioType.NO_CONVERGENCE][0].estimate
        for item in scenarios:
            estimate = item.estimate
            if (
                estimate.assembly != reference.assembly or estimate.config != reference.config
                or estimate.scenario.opened_at != reference.scenario.opened_at
                or estimate.matched_quantity != reference.matched_quantity
                or estimate.residual_delta != reference.residual_delta
                or estimate.fills[0] != reference.fills[0]
                or estimate.fills[2] != reference.fills[2]
            ):
                rejects.append(ScenarioDecisionIssue(
                    ScenarioDecisionReason.SCENARIO_COMPARISON_MISMATCH,
                    item.name, item.scenario_type,
                ))
                evidence_valid = False

    margin = None
    if evidence_valid:
        try:
            with localcontext() as context:
                context.traps[Inexact] = True
                margin = min(baseline_profits) - max(stress_losses)
        except DecimalException:
            rejects.append(ScenarioDecisionIssue(ScenarioDecisionReason.SCENARIO_MARGIN_UNAVAILABLE))
        if margin is not None and margin < config.min_scenario_margin:
            observations.append(ScenarioDecisionIssue(
                ScenarioDecisionReason.SCENARIO_MARGIN_INSUFFICIENT,
            ))

    issues = tuple(rejects + observations)
    status = Decision.REJECT if rejects else Decision.OBSERVE if observations else Decision.PASS
    if status is Decision.PASS:
        issues = (ScenarioDecisionIssue(ScenarioDecisionReason.SCENARIO_PASS_ALL_RULES),)
    return ScenarioDecisionResult(status, issues, scenarios, config, margin)
