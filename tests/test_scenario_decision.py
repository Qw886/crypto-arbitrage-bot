"""Offline decisions compare final scenario evidence, never redo its cash flows."""

from dataclasses import replace
from datetime import timedelta
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_decision import final_values, thresholds
from test_offline_estimate import CLOSED, OPENED, assembly_for, books, config
from test_scenario import baseline, event, target

from crypto_arbitrage_bot.arbitrage import estimate as estimate_engine
from crypto_arbitrage_bot.arbitrage import scenario as scenario_engine
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.decision import Decision, DecisionReason
from crypto_arbitrage_bot.models.scenario import ScenarioType
from crypto_arbitrage_bot.models.scenario_decision import (
    ScenarioDecisionConfig,
    ScenarioDecisionReason,
)

D = Decimal
R = ScenarioDecisionReason


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("scenario decisions must stay offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


def limits(**changes):
    """Synthetic injected limits, deliberately unrelated to live trading."""
    values = dict(
        decision=thresholds(), min_baseline_net_profit=D("1"),
        min_baseline_net_return=D(".001"), max_stress_loss=D("4"),
        max_stress_loss_return=D(".01"), min_convergence_profit=D("20"),
        min_scenario_margin=D("5"),
    )
    values.update(changes)
    return ScenarioDecisionConfig(**values)


def scenarios(symbol="BTCUSDT", baseline_rate=".1", stress_pct=".16", **config_changes):
    source = books(symbol)
    definitions = (
        baseline(source, (event(baseline_rate, symbol=symbol),)),
        target(source, events=(event(".1", symbol=symbol),)),
        target(source, ScenarioType.ADVERSE, stress_pct, (event(".1", symbol=symbol),)),
    )
    return scenario_engine.evaluate_scenarios(
        assembly_for((source[0], source[2])), config(**config_changes), definitions,
    )


@pytest.fixture
def evidence():
    return scenarios()


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_all_roles_pass_with_actual_estimates_and_decimal_margin(symbol):
    source = scenarios(symbol)
    settings = limits()
    result = decide_scenarios(source, settings)
    assert result.status is Decision.PASS
    assert result.reason_codes == (R.SCENARIO_PASS_ALL_RULES,)
    assert result.scenarios is source and result.config is settings
    assert source[0].net_profit == D("9.0144")
    assert source[1].net_profit == D("25.0272")
    assert source[2].net_profit == D("-2.9952")
    assert result.scenario_margin == D("6.0192")
    assert isinstance(result.scenario_margin, Decimal)
    assert not hasattr(result, "can_trade") and not hasattr(result, "tradable")


@pytest.mark.parametrize("rate", [".01", "0"])
def test_good_convergence_never_overrides_losing_baseline(rate):
    source = scenarios(baseline_rate=rate)
    assert source[1].net_profit > limits().min_convergence_profit
    assert source[0].net_profit < D("0")
    result = decide_scenarios(source, limits())
    assert result.status is Decision.REJECT
    assert R.BASELINE_NON_POSITIVE in result.reason_codes


@pytest.mark.parametrize("changes", [{"profit": "0"}, {"net_return": "0"}])
def test_zero_baseline_profit_or_return_is_always_rejected(evidence, changes):
    base = replace(evidence[0], estimate=final_values(evidence[0].estimate, **changes))
    result = decide_scenarios((base, *evidence[1:]), limits())
    assert result.status is Decision.REJECT
    assert R.BASELINE_NON_POSITIVE in result.reason_codes


def test_positive_funding_and_basis_cannot_override_excess_stress_loss():
    source = scenarios(stress_pct=".20")
    assert source[0].net_profit > D("0")
    assert source[2].funding_total == D("22") and source[2].entry_basis > D("0")
    result = decide_scenarios(source, limits())
    assert result.status is Decision.REJECT
    assert {R.STRESS_LOSS_TOO_LARGE, R.STRESS_RETURN_TOO_LOW} <= set(result.reason_codes)


@pytest.mark.parametrize("changes,reason", [
    ({"decision": thresholds(pass_net_profit=D("10"))},
     DecisionReason.PASS_NET_PROFIT_BELOW_TARGET),
    ({"decision": thresholds(pass_net_return=D(".03"))},
     DecisionReason.PASS_NET_RETURN_BELOW_TARGET),
    ({"min_convergence_profit": D("26")}, R.CONVERGENCE_PROFIT_INSUFFICIENT),
    ({"min_scenario_margin": D("7")}, R.SCENARIO_MARGIN_INSUFFICIENT),
])
def test_valid_safe_but_below_any_higher_target_is_observe(evidence, changes, reason):
    result = decide_scenarios(evidence, limits(**changes))
    assert result.status is Decision.OBSERVE
    assert result.reason_codes == (reason,)


@pytest.mark.parametrize("changes,reason", [
    ({"min_baseline_net_profit": D("10")}, R.BASELINE_BELOW_THRESHOLD),
    ({"min_baseline_net_return": D(".03")}, R.BASELINE_BELOW_THRESHOLD),
    ({"decision": thresholds(min_profit_buffer=D("10"), pass_net_profit=D("11"))},
     DecisionReason.PROFIT_BUFFER_INSUFFICIENT),
    ({"decision": thresholds(min_matched_notional=D("205"))},
     DecisionReason.MATCHED_NOTIONAL_TOO_SMALL),
])
def test_hard_safety_conditions_reject_even_when_convergence_is_profitable(evidence, changes, reason):
    result = decide_scenarios(evidence, limits(**changes))
    assert result.status is Decision.REJECT and reason in result.reason_codes


def test_no_convergence_basis_is_not_itself_a_reason_to_reject(evidence):
    base = evidence[0]
    assert base.entry_mid_basis == base.assumed_exit_mid_basis
    assert base.estimate.mid_basis_pnl == D("0")
    assert base.net_profit > D("0")  # Explicit settlements cover actual execution costs.
    assert decide_scenarios(evidence, limits()).status is Decision.PASS


def test_all_threshold_equalities_are_accepted(evidence):
    base, convergence, stress = evidence
    settings = limits(
        decision=thresholds(pass_net_profit=base.net_profit, pass_net_return=base.net_return),
        min_baseline_net_profit=base.net_profit, min_baseline_net_return=base.net_return,
        max_stress_loss=stress.net_profit.copy_negate(),
        max_stress_loss_return=stress.net_return.copy_negate(),
        min_convergence_profit=convergence.net_profit, min_scenario_margin=D("6.0192"),
    )
    assert decide_scenarios(evidence, settings).status is Decision.PASS


@pytest.mark.parametrize("field,reason", [
    ("max_stress_loss", R.STRESS_LOSS_TOO_LARGE),
    ("max_stress_loss_return", R.STRESS_RETURN_TOO_LOW),
])
def test_amount_and_return_stress_limits_are_independent(evidence, field, reason):
    result = decide_scenarios(evidence, limits(**{field: D("0")}))
    assert result.status is Decision.REJECT
    assert result.reason_codes == (reason,)


def test_profitable_stress_has_zero_loss_magnitude():
    source = scenarios(stress_pct=".12")
    assert source[2].net_profit > D("0") and source[2].net_return > D("0")
    result = decide_scenarios(source, limits(max_stress_loss=D("0"), max_stress_loss_return=D("0")))
    assert result.status is Decision.PASS and result.scenario_margin == source[0].net_profit


@pytest.mark.parametrize("indexes", [(), (1,), (0, 1), (0, 2), (1, 2)])
def test_missing_any_required_role_rejects(evidence, indexes):
    result = decide_scenarios(tuple(evidence[index] for index in indexes), limits())
    assert result.status is Decision.REJECT and R.SCENARIO_DATA_INCOMPLETE in result.reason_codes
    assert result.scenario_margin is None


@pytest.mark.parametrize("problem", ["incomplete", "invalid_quality", "depth"])
def test_failed_source_scenario_blocks_decision_without_available_profit(evidence, problem):
    original = evidence[2]
    assembly = original.estimate.assembly
    definition = original.definition
    kwargs = {}
    if problem == "incomplete":
        assembly = assemble_snapshot(
            assembly.symbol, replace(assembly.sources, spot_depth=None), assembled_at=OPENED,
        )
    elif problem == "invalid_quality":
        kwargs["evaluated_at"] = OPENED + timedelta(seconds=11)
    else:
        exit_book = definition.exit_scenario.perp_exit_book
        definition = replace(definition, exit_scenario=replace(
            definition.exit_scenario,
            perp_exit_book=replace(exit_book, asks=(replace(exit_book.asks[0], quantity=D("1")),)),
        ))
    failed = scenario_engine.evaluate_scenario(assembly, config(), definition, **kwargs)
    assert failed.net_profit is failed.net_return is None
    result = decide_scenarios((*evidence[:2], failed), limits())
    assert result.status is Decision.REJECT and R.SCENARIO_DATA_INCOMPLETE in result.reason_codes
    assert result.scenario_margin is None
    if problem == "depth":
        assert DecisionReason.INSUFFICIENT_DEPTH in result.reason_codes


def test_multiple_roles_report_multiple_stable_reasons_with_source_names():
    source = scenarios(baseline_rate="0", stress_pct=".20")
    result = decide_scenarios(source, limits(min_convergence_profit=D("30")))
    assert result.status is Decision.REJECT
    assert {R.BASELINE_NON_POSITIVE, R.BASELINE_BELOW_THRESHOLD, R.STRESS_LOSS_TOO_LARGE,
            R.STRESS_RETURN_TOO_LOW, R.CONVERGENCE_PROFIT_INSUFFICIENT} <= set(result.reason_codes)
    assert len(result.reason_codes) == len(set(result.reason_codes))
    assert all(isinstance(code, ScenarioDecisionReason | DecisionReason) for code in result.reason_codes)
    assert any(issue.code is R.STRESS_LOSS_TOO_LARGE and issue.scenario_name == source[2].name
               and issue.scenario_type is ScenarioType.ADVERSE for issue in result.issues)


def test_multiple_stress_cases_use_worst_loss_not_first_or_average(evidence):
    worst = scenarios(stress_pct=".20")[2]
    worst = replace(worst, definition=replace(worst.definition, name="Wider stress"))
    for collection in ((*evidence, worst), (worst, *evidence)):
        result = decide_scenarios(collection, limits())
        assert result.status is Decision.REJECT
        assert R.STRESS_LOSS_TOO_LARGE in result.reason_codes
        assert result.scenario_margin == D("-1.9872")


def test_multiple_baselines_and_convergences_are_all_checked(evidence):
    weak = scenarios(baseline_rate=".09")[0]
    weak = replace(weak, definition=replace(weak.definition, name="Lower baseline funding"))
    convergence = replace(evidence[1], definition=replace(evidence[1].definition, name="Lower upside"),
                          estimate=final_values(evidence[1].estimate, profit="19"))
    result = decide_scenarios((*evidence, weak, convergence), limits())
    assert result.status is Decision.OBSERVE
    assert {R.CONVERGENCE_PROFIT_INSUFFICIENT, R.SCENARIO_MARGIN_INSUFFICIENT} <= set(result.reason_codes)
    assert result.scenario_margin == D("3.8192")


@pytest.mark.parametrize("problem", ["duplicate_name", "symbol", "small_stress", "costs"])
def test_incomparable_evidence_cannot_be_used_to_hide_stress_risk(evidence, problem):
    if problem == "duplicate_name":
        stress = replace(evidence[2], definition=replace(evidence[2].definition, name=evidence[0].name))
    elif problem == "symbol":
        stress = scenarios("ETHUSDT")[2]
    elif problem == "small_stress":
        stress = scenarios(target_quantity=D(".1"))[2]
    else:
        stress = scenarios(other_costs=D("0"))[2]
    result = decide_scenarios((*evidence[:2], stress), limits())
    assert result.status is Decision.REJECT
    assert R.SCENARIO_COMPARISON_MISMATCH in result.reason_codes
    assert result.scenario_margin is None
    if problem == "small_stress":
        assert DecisionReason.MATCHED_NOTIONAL_TOO_SMALL in result.reason_codes


def test_excess_residual_delta_remains_hard_failure_even_in_stress(evidence):
    stress = replace(evidence[2], estimate=final_values(evidence[2].estimate, residual="-.02"))
    result = decide_scenarios((*evidence[:2], stress), limits())
    assert result.status is Decision.REJECT
    assert DecisionReason.RESIDUAL_DELTA_TOO_LARGE in result.reason_codes


def test_different_explicit_exit_times_and_funding_hypotheses_are_allowed(evidence):
    source = books()
    definition = target(
        source, ScenarioType.ADVERSE, ".16", (event(".1"),),
        closed_at=CLOSED + timedelta(days=1),
    )
    stress = scenario_engine.evaluate_scenario(evidence[0].estimate.assembly, config(), definition)
    assert stress.estimate.scenario.closed_at != evidence[0].estimate.scenario.closed_at
    assert decide_scenarios((*evidence[:2], stress), limits()).status is Decision.PASS


@pytest.mark.parametrize("field", [
    "min_baseline_net_profit", "min_baseline_net_return", "max_stress_loss",
    "max_stress_loss_return", "min_convergence_profit", "min_scenario_margin",
])
@pytest.mark.parametrize("value,error", [(0.1, TypeError), (D("NaN"), ValueError),
                                         (D("Infinity"), ValueError), (D("-1"), ValueError)])
def test_each_threshold_requires_finite_nonnegative_decimal(field, value, error):
    with pytest.raises(error):
        limits(**{field: value})


def test_decision_never_changes_or_recalculates_scenario_cashflows(evidence, monkeypatch):
    originals = tuple((item.net_profit, item.net_return, item.estimate.calculation,
                       item.estimate.fees, item.estimate.decomposition) for item in evidence)
    forbidden = Mock(side_effect=AssertionError("scenario decision must only consume final results"))
    for module, names in (
        (estimate_engine, ("estimate_once", "size_position", "simulate_fill", "accrue_short_funding",
                           "round_trip_from_fills", "decompose_fills")),
        (scenario_engine, ("evaluate_scenario", "evaluate_scenarios", "estimate_once")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    result = decide_scenarios(evidence, limits())
    assert result.status is Decision.PASS and result.scenarios is evidence
    for item, original in zip(result.scenarios, originals, strict=True):
        assert item.net_profit is original[0] and item.net_return is original[1]
        assert item.estimate.calculation is original[2] and item.estimate.fees is original[3]
        assert item.estimate.decomposition is original[4]
    forbidden.assert_not_called()


def test_decimal_context_is_not_changed_by_scenario_comparison(evidence):
    settings = limits()
    expected = decide_scenarios(evidence, settings)
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        flags = context.flags.copy()
        assert decide_scenarios(evidence, settings) == expected
        assert context.prec == 2 and context.rounding is ROUND_DOWN
        assert context.traps[Inexact] and context.flags == flags


def test_unrepresentable_margin_is_rejected_without_rounding_away_stress(evidence):
    stress = replace(evidence[2], estimate=final_values(evidence[2].estimate, profit="-1e-60"))
    result = decide_scenarios((*evidence[:2], stress), limits())
    assert result.status is Decision.REJECT
    assert result.reason_codes == (R.SCENARIO_MARGIN_UNAVAILABLE,)
    assert result.scenario_margin is None
