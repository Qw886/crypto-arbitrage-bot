"""Offline research decisions over final estimates, without recalculating PnL."""

from dataclasses import replace
from datetime import timedelta
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED, assembly_for, books, config, scenario_for

from crypto_arbitrage_bot.arbitrage import estimate as engine
from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.decision import Decision, DecisionConfig, DecisionReason
from crypto_arbitrage_bot.models.estimate import EstimateStatus

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("decision tests are offline")),
    )


def thresholds(**changes):
    """Deliberately synthetic thresholds; every value is injected explicitly."""
    values = dict(
        min_net_profit=D("1"), min_net_return=D(".001"),
        min_matched_notional=D("100"), max_residual_delta=D(".01"),
        min_profit_buffer=D("2"), pass_net_profit=D("3"), pass_net_return=D(".005"),
    )
    values.update(changes)
    return DecisionConfig(**values)


def research_estimate(symbol="BTCUSDT", rate=".1", **config_changes):
    source_books = books(symbol)
    return engine.estimate_once(
        assembly_for((source_books[0], source_books[2])), config(**config_changes),
        scenario_for(source_books, rate=rate),
    )


@pytest.fixture
def profitable():
    result = research_estimate()
    assert result.complete and result.data_quality.valid
    assert result.net_profit == D("17.004")
    return result


def final_values(estimate, *, profit=None, net_return=None, residual=None):
    """Vary final evidence to isolate thresholds; the decision must not redo math."""
    changes = {}
    if profit is not None:
        changes["calculation"] = replace(estimate.calculation, net_profit=D(profit))
    if net_return is not None:
        changes["net_return"] = D(net_return)
    if residual is not None:
        changes["sizing"] = replace(estimate.sizing, residual_delta=D(residual))
    return replace(estimate, **changes)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_complete_offline_estimate_passes_all_rules(symbol):
    estimate = research_estimate(symbol)
    settings = thresholds()
    result = decide(estimate, settings)
    assert result.status is Decision.PASS
    assert result.reason_codes == (DecisionReason.PASS_ALL_RULES,)
    assert result.estimate is estimate and result.config is settings
    assert result.matched_notional == D("204")
    assert result.matched_notional == min(
        estimate.fills[0].quote_notional, estimate.fills[2].quote_notional,
    )
    assert result.matched_notional != estimate.capital_occupied
    assert isinstance(result.matched_notional, Decimal)
    assert not hasattr(result, "tradable") and not hasattr(result, "can_trade")


def test_invalid_quality_result_rejects_without_profit():
    source_books = books()
    estimate = engine.estimate_once(
        assembly_for((source_books[0], source_books[2])), config(), scenario_for(source_books),
        evaluated_at=OPENED + timedelta(seconds=11),
    )
    assert estimate.status is EstimateStatus.INVALID_MARKET_DATA
    result = decide(estimate, thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.INVALID_MARKET_DATA in result.reason_codes
    assert result.estimate.net_profit is result.estimate.net_return is None


def test_missing_quality_cannot_be_overridden_by_profitable_estimate(profitable):
    result = decide(replace(profitable, data_quality=None), thresholds())
    assert result.status is Decision.REJECT
    assert result.reason_codes == (DecisionReason.INVALID_MARKET_DATA,)


def test_incomplete_snapshot_rejects_without_profit():
    source_books = books()
    complete = assembly_for((source_books[0], source_books[2]))
    incomplete = assemble_snapshot(
        complete.symbol, replace(complete.sources, spot_depth=None), assembled_at=OPENED,
    )
    estimate = engine.estimate_once(incomplete, config(), scenario_for(source_books))
    result = decide(estimate, thresholds())
    assert result.status is Decision.REJECT
    assert {DecisionReason.INVALID_MARKET_DATA, DecisionReason.INCOMPLETE_ESTIMATE} <= set(
        result.reason_codes,
    )
    assert result.estimate.net_profit is result.estimate.net_return is None


@pytest.mark.parametrize(
    "field", ["sizing", "fees", "calculation", "decomposition", "capital_occupied",
              "net_return", "executable_entry_basis"],
)
def test_missing_required_estimate_component_rejects(profitable, field):
    result = decide(replace(profitable, **{field: None}), thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.INCOMPLETE_ESTIMATE in result.reason_codes
    assert DecisionReason.PASS_ALL_RULES not in result.reason_codes


@pytest.mark.parametrize("other_costs,profit", [(".1", "-2.796"), ("17.104", "0")])
def test_non_positive_actual_profit_rejects(other_costs, profit):
    rate = ".01" if profit.startswith("-") else ".1"
    estimate = research_estimate(rate=rate, other_costs=D(other_costs))
    assert estimate.complete and estimate.net_profit == D(profit)
    result = decide(estimate, thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.NON_POSITIVE_NET_PROFIT in result.reason_codes
    assert DecisionReason.NON_POSITIVE_NET_RETURN in result.reason_codes


@pytest.mark.parametrize("net_return", ["-.001", "0"])
def test_non_positive_return_rejects_even_when_profit_positive(profitable, net_return):
    estimate = final_values(profitable, net_return=net_return)
    assert estimate.net_profit > 0
    result = decide(estimate, thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.NON_POSITIVE_NET_RETURN in result.reason_codes


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"min_net_profit": D("18"), "pass_net_profit": D("20")},
         DecisionReason.NET_PROFIT_BELOW_THRESHOLD),
        ({"min_net_return": D(".05"), "pass_net_return": D(".06")},
         DecisionReason.NET_RETURN_BELOW_THRESHOLD),
        ({"min_matched_notional": D("205")}, DecisionReason.MATCHED_NOTIONAL_TOO_SMALL),
        ({"min_profit_buffer": D("18"), "pass_net_profit": D("20")},
         DecisionReason.PROFIT_BUFFER_INSUFFICIENT),
    ],
)
def test_each_hard_minimum_is_reject_not_observe(profitable, changes, reason):
    result = decide(profitable, thresholds(**changes))
    assert result.status is Decision.REJECT
    assert reason in result.reason_codes


@pytest.mark.parametrize("residual", [".0101", "-.0101"])
def test_residual_delta_uses_absolute_base_quantity(profitable, residual):
    result = decide(final_values(profitable, residual=residual), thresholds())
    assert result.status is Decision.REJECT
    assert result.reason_codes == (DecisionReason.RESIDUAL_DELTA_TOO_LARGE,)


@pytest.mark.parametrize(
    "profit,net_return,reasons",
    [
        ("2.5", ".005", (DecisionReason.PASS_NET_PROFIT_BELOW_TARGET,)),
        ("3", ".004", (DecisionReason.PASS_NET_RETURN_BELOW_TARGET,)),
        ("2.5", ".004", (DecisionReason.PASS_NET_PROFIT_BELOW_TARGET,
                            DecisionReason.PASS_NET_RETURN_BELOW_TARGET)),
    ],
)
def test_between_hard_limits_and_pass_targets_is_observe(profitable, profit, net_return, reasons):
    result = decide(final_values(profitable, profit=profit, net_return=net_return), thresholds())
    assert result.status is Decision.OBSERVE
    assert result.reason_codes == reasons


def test_hard_boundaries_are_inclusive_but_do_not_imply_pass(profitable):
    estimate = final_values(profitable, profit="2", net_return=".002", residual=".01")
    result = decide(estimate, thresholds(
        min_net_profit=D("2"), min_net_return=D(".002"), min_matched_notional=D("204"),
    ))
    assert result.status is Decision.OBSERVE
    assert result.reason_codes == (
        DecisionReason.PASS_NET_PROFIT_BELOW_TARGET, DecisionReason.PASS_NET_RETURN_BELOW_TARGET,
    )


@pytest.mark.parametrize("residual", [".01", "-.01"])
def test_pass_targets_and_maximum_residual_are_inclusive(profitable, residual):
    estimate = final_values(profitable, profit="3", net_return=".005", residual=residual)
    result = decide(estimate, thresholds(min_matched_notional=D("204")))
    assert result.status is Decision.PASS
    assert result.reason_codes == (DecisionReason.PASS_ALL_RULES,)


def test_profit_buffer_compares_final_profit_without_adding_or_deducting(profitable):
    estimate = final_values(profitable, profit="4")
    settings = thresholds(min_net_profit=D("3"), min_profit_buffer=D("3"), pass_net_profit=D("4"))
    result = decide(estimate, settings)
    assert result.status is Decision.PASS
    assert result.estimate.net_profit == D("4")


def test_positive_funding_does_not_override_negative_final_profit():
    estimate = research_estimate(rate=".01")
    assert estimate.funding_rate > 0 and estimate.estimated_funding > 0
    assert estimate.net_profit < 0
    assert decide(estimate, thresholds()).status is Decision.REJECT


def test_positive_basis_does_not_override_insufficient_profit(profitable):
    estimate = final_values(profitable, profit=".5")
    assert estimate.executable_entry_basis > 0
    result = decide(estimate, thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.NET_PROFIT_BELOW_THRESHOLD in result.reason_codes


def test_all_failed_rules_are_reported_once(profitable):
    estimate = final_values(profitable, profit="-1", net_return="0", residual=".02")
    result = decide(estimate, thresholds(min_matched_notional=D("205")))
    assert result.status is Decision.REJECT
    assert set(result.reason_codes) == {
        DecisionReason.NON_POSITIVE_NET_PROFIT, DecisionReason.NON_POSITIVE_NET_RETURN,
        DecisionReason.NET_PROFIT_BELOW_THRESHOLD, DecisionReason.NET_RETURN_BELOW_THRESHOLD,
        DecisionReason.MATCHED_NOTIONAL_TOO_SMALL, DecisionReason.RESIDUAL_DELTA_TOO_LARGE,
        DecisionReason.PROFIT_BUFFER_INSUFFICIENT,
    }
    assert len(result.reason_codes) == len(set(result.reason_codes))
    assert all(isinstance(reason, DecisionReason) for reason in result.reason_codes)


@pytest.mark.parametrize("leg_index,side", [(0, "asks"), (1, "bids"), (2, "bids"), (3, "asks")])
def test_any_of_four_legs_with_insufficient_depth_rejects(leg_index, side):
    source_books = list(books())
    source_books[leg_index] = replace(
        source_books[leg_index], **{side: getattr(source_books[leg_index], side)[:1]},
    )
    estimate = engine.estimate_once(
        assembly_for((source_books[0], source_books[2])), config(), scenario_for(source_books, rate=".1"),
    )
    result = decide(estimate, thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.INSUFFICIENT_DEPTH in result.reason_codes
    assert DecisionReason.INCOMPLETE_ESTIMATE in result.reason_codes
    assert estimate.net_profit is estimate.net_return is None
    assert not estimate.fills[leg_index].complete


@pytest.mark.parametrize("change", ["missing_leg", "duplicate_leg", "wrong_symbol"])
def test_execution_evidence_must_have_all_matching_legs(profitable, change):
    fills = profitable.fills
    if change == "missing_leg":
        fills = fills[:3]
        expected = DecisionReason.INCOMPLETE_ESTIMATE
    else:
        modified = fills[0] if change == "duplicate_leg" else replace(fills[1], symbol="ETHUSDT")
        fills = (fills[0], modified, fills[2], fills[3])
        expected = DecisionReason.INCONSISTENT_EXECUTION
    result = decide(replace(profitable, fills=fills), thresholds())
    assert result.status is Decision.REJECT and expected in result.reason_codes


@pytest.mark.parametrize("leg_index", [1, 3])
@pytest.mark.parametrize("change,reason", [
    ({"fills": ()}, DecisionReason.INCOMPLETE_ESTIMATE),
    ({"quote_notional": D("0")}, DecisionReason.INVALID_ESTIMATE_VALUES),
])
def test_filled_exit_leg_requires_actual_execution_evidence(profitable, leg_index, change, reason):
    fills = list(profitable.fills)
    fills[leg_index] = replace(fills[leg_index], **change)
    result = decide(replace(profitable, fills=tuple(fills)), thresholds())
    assert result.status is Decision.REJECT and reason in result.reason_codes


@pytest.mark.parametrize("quantity", [D("NaN"), D("sNaN"), D("Infinity"), D("-Infinity")])
def test_nonfinite_matched_quantity_is_rejected_without_decimal_exception(profitable, quantity):
    estimate = replace(profitable, sizing=replace(profitable.sizing, perp_quantity=quantity))
    with localcontext() as context:
        before = context.flags.copy()
        result = decide(estimate, thresholds())
        assert context.flags == before
    assert result.status is Decision.REJECT
    assert DecisionReason.INVALID_ESTIMATE_VALUES in result.reason_codes


@pytest.mark.parametrize("field,value", [("net_return", D("NaN")),
                                         ("capital_occupied", D("0")),
                                         ("capital_occupied", 100.0)])
def test_invalid_required_values_cannot_pass(profitable, field, value):
    result = decide(replace(profitable, **{field: value}), thresholds())
    assert result.status is Decision.REJECT
    assert DecisionReason.INVALID_ESTIMATE_VALUES in result.reason_codes


@pytest.mark.parametrize("field", ["min_net_profit", "min_net_return", "min_matched_notional",
                                  "max_residual_delta", "min_profit_buffer", "pass_net_profit",
                                  "pass_net_return"])
def test_every_threshold_requires_decimal_not_float(field):
    with pytest.raises(TypeError):
        thresholds(**{field: 0.1})


@pytest.mark.parametrize("value", [D("NaN"), D("Infinity"), D("-1")])
def test_nonfinite_or_negative_threshold_rejected(value):
    with pytest.raises(ValueError):
        thresholds(min_net_profit=value)


@pytest.mark.parametrize("field", ["pass_net_profit", "pass_net_return"])
def test_pass_targets_are_required_decimal_values(field):
    with pytest.raises(TypeError):
        thresholds(**{field: None})
    values = dict(
        min_net_profit=D("1"), min_net_return=D(".001"), min_matched_notional=D("100"),
        max_residual_delta=D(".01"), min_profit_buffer=D("2"),
        pass_net_profit=D("3"), pass_net_return=D(".005"),
    )
    del values[field]
    with pytest.raises(TypeError):
        DecisionConfig(**values)


@pytest.mark.parametrize("changes", [
    {"pass_net_profit": D("2")},
    {"pass_net_profit": D("1.5")},
    {"min_net_profit": D("3"), "pass_net_profit": D("3")},
    {"pass_net_return": D(".001")},
    {"pass_net_return": D(".0005")},
])
def test_pass_targets_must_be_strictly_above_hard_limits(changes):
    with pytest.raises(ValueError):
        thresholds(**changes)


def test_decision_preserves_existing_net_results_and_never_invokes_math(profitable, monkeypatch):
    original_profit, original_return = profitable.net_profit, profitable.net_return
    forbidden = Mock(side_effect=AssertionError("decision must consume the existing estimate"))
    for name in ("estimate_once", "size_position", "simulate_fill", "accrue_short_funding",
                 "round_trip_from_fills", "decompose_fills"):
        monkeypatch.setattr(engine, name, forbidden)
    result = decide(profitable, thresholds())
    assert result.status is Decision.PASS
    assert result.estimate is profitable
    assert result.estimate.net_profit is original_profit
    assert result.estimate.net_return is original_return
    assert result.estimate.fees is profitable.fees
    assert result.estimate.decomposition is profitable.decomposition
    forbidden.assert_not_called()


def test_decision_is_independent_of_ambient_decimal_context(profitable):
    settings = thresholds()
    expected = decide(profitable, settings)
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        before = context.flags.copy()
        result = decide(profitable, settings)
        assert context.prec == 2 and context.rounding == ROUND_DOWN
        assert context.traps[Inexact] and context.flags == before
    assert result == expected
