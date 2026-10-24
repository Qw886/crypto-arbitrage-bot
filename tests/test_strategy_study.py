"""Fixed synthetic cohorts test research review; no real strategy approval."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from unittest.mock import Mock

import pytest
from test_decision import research_estimate, thresholds
from test_forward_validation import later_snapshot, shift_fixture
from test_offline_estimate import OPENED

from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.analytics.strategy_study import assess_strategy_study
from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.models.strategy_study import StrategyStudyProtocol
from crypto_arbitrage_bot.models.strategy_study import StudyReason as R
from crypto_arbitrage_bot.models.strategy_study import StudyStatus as S
from crypto_arbitrage_bot.storage.forward_validation_records import build_forward_validation_record
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity, exact_json

D = Decimal


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("pure study assessment must not perform I/O"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("sqlite3.connect", forbidden)


def entry(day=0, symbol="BTCUSDT", suffix=""):
    delta = timedelta(days=day)
    estimate = shift_fixture(research_estimate(symbol), delta)
    decision = decide(estimate, thresholds())
    assert decision.status.value == "PASS"
    return build_opportunity(estimate, decision, created_at=OPENED+delta,
        opportunity_id=f"synthetic-{symbol}-{day}-{suffix}", formula_version="test-formula",
        config_version="test-fixed-policy")


def forward(record, *, exit_ask="112", late=0):
    opened = record.created_at
    at = opened+timedelta(hours=2, seconds=late)
    result = validate_forward_outcome(record, later_snapshot(record.symbol, perp_ask=exit_ask, at=at),
        validation_time=at, funding_events=())
    assert result.status.value == "VALIDATED"
    return build_forward_validation_record(result, validation_id=record.id+f"-exit-{late}", created_at=at)


def protocol(symbols=("BTCUSDT",), **changes):
    values = dict(id="synthetic-preregistered-study", registered_at=OPENED-timedelta(days=1),
        study_start=OPENED, study_end=OPENED+timedelta(days=3), symbols=symbols,
        formula_version="test-formula", config_version="test-fixed-policy",
        config_fingerprints=tuple((s, entry(symbol=s).config_fingerprint) for s in symbols),
        max_holding_period=timedelta(hours=2), min_entry_separation=timedelta(days=1),
        max_exit_lateness=timedelta(seconds=300), min_entries_per_symbol=2,
        min_entry_days_per_symbol=2, min_mean_net_return=D(".001"),
        min_median_net_return=D(".001"), max_loss_return=D(".1"))
    values.update(changes)
    return StrategyStudyProtocol(**values)


def assess(records, outcomes=None, policy=None, **kwargs):
    return assess_strategy_study(protocol() if policy is None else policy, tuple(records),
        tuple(forward(r) for r in records) if outcomes is None else tuple(outcomes),
        evaluated_at=kwargs.get("at", OPENED+timedelta(days=4)))


def test_complete_positive_cohort_is_only_ready_for_execution_review():
    records = (entry(0), entry(1))
    outcomes = tuple(forward(r) for r in records)
    before = exact_json((records, outcomes))
    result = assess(records, outcomes)
    assert result.status is S.READY_FOR_EXECUTION_REVIEW and result.reason_codes == (R.STUDY_RULES_MET,)
    assert result.symbols[0].profit.mean == D("6.9688")
    assert result.symbols[0].entry_days == 2 and result.symbols[0].net_return.count == 2
    assert isinstance(result.symbols[0].net_return.mean, D)
    assert not hasattr(result, "trade_allowed") and exact_json((records, outcomes)) == before


def test_empty_cohort_is_insufficient_not_zero_profit():
    result = assess(())
    assert result.status is S.INSUFFICIENT_EVIDENCE
    assert R.NO_ELIGIBLE_ENTRY in result.reason_codes and result.symbols[0].profit.mean is None


def test_single_entry_never_satisfies_sample_days():
    result = assess((entry(),))
    assert result.status is S.INSUFFICIENT_EVIDENCE
    assert {R.TOO_FEW_SEPARATED_ENTRIES, R.TOO_FEW_ENTRY_DAYS}.issubset(result.reason_codes)


def test_study_cannot_pass_before_fixed_end_and_exit_allowance():
    result = assess((entry(), entry(1)), at=OPENED+timedelta(days=3))
    assert result.status is S.INSUFFICIENT_EVIDENCE and R.STUDY_NOT_FINISHED in result.reason_codes


def test_negative_results_do_not_pass_even_with_zero_study_thresholds():
    records = (entry(), entry(1))
    result = assess(records, tuple(forward(r, exit_ask="120") for r in records),
                    protocol(min_mean_net_return=D("0"), min_median_net_return=D("0")))
    assert result.status is S.UNSUPPORTED and R.NON_POSITIVE_OUTCOME_SUMMARY in result.reason_codes


def test_positive_results_below_fixed_threshold_are_unsupported():
    result = assess((entry(), entry(1)), policy=protocol(min_mean_net_return=D("1")))
    assert result.status is S.UNSUPPORTED and R.RETURN_BELOW_STUDY_THRESHOLD in result.reason_codes


def test_worst_loss_is_not_hidden_by_profitable_entries():
    records = (entry(), entry(1), entry(2))
    outcomes = (forward(records[0]), forward(records[1]), forward(records[2], exit_ask="120"))
    result = assess(records, outcomes, protocol(max_loss_return=D(".001")))
    assert result.status is S.UNSUPPORTED and R.WORST_LOSS_EXCEEDS_LIMIT in result.reason_codes


def test_earliest_entry_selected_without_cherry_picking_best_outcome():
    first = entry()
    second = shift_fixture(entry(suffix="later"), timedelta(hours=3))
    third = entry(1)
    result = assess((second, third, first), (forward(first, exit_ask="120"), forward(second), forward(third)))
    assert result.symbols[0].eligible_entries == 3
    assert result.symbols[0].selected_entry_ids == (first.id, third.id)
    assert result.status is S.UNSUPPORTED


def test_other_horizon_cannot_replace_bad_planned_exit():
    records = (entry(), entry(1))
    later = forward(records[0], late=3600)
    good = forward(records[1])
    result = assess(records, (later, good))
    assert result.status is S.INSUFFICIENT_EVIDENCE
    assert R.MISSING_PLANNED_EXIT_VALIDATION in result.reason_codes
    assert result.symbols[0].profit.null_count == 1


@pytest.mark.parametrize("late,ok", [(0, True), (300, True), (301, False), (-1, False)])
def test_planned_exit_lateness_is_bounded(late, ok):
    records = (entry(), entry(1))
    result = assess(records, tuple(forward(r, late=late) for r in records))
    assert (result.status is S.READY_FOR_EXECUTION_REVIEW) is ok


def test_missing_or_failed_validation_never_counts_as_zero():
    records = (entry(), entry(1))
    bad = replace(forward(records[1]), status="INCOMPLETE", validated_simulated_net_profit=None,
                  validated_simulated_net_return=None)
    result = assess(records, (forward(records[0]), bad))
    assert result.status is S.INSUFFICIENT_EVIDENCE
    assert result.symbols[0].profit.count == 1 and result.symbols[0].profit.null_count == 1


@pytest.mark.parametrize("field,value", [("symbol", "ETHUSDT"), ("matched_quantity", D("3")),
    ("capital_occupied", D("1")), ("original_estimated_net_profit", D("1")),
    ("original_estimated_net_return", D("1")), ("validated_simulated_net_profit", D("999")),
    ("validated_simulated_net_return", D("999")), ("holding_duration", timedelta(hours=8)),
    ("validated_simulated_net_profit", None)])
def test_mismatched_or_tampered_validation_does_not_support_entry(field, value):
    records = (entry(), entry(1))
    result = assess(records, (replace(forward(records[0]), **{field: value}), forward(records[1])))
    assert result.status is S.INSUFFICIENT_EVIDENCE and R.INVALID_VALIDATION_EVIDENCE in result.reason_codes


def test_duplicate_validation_at_same_planned_exit_is_not_double_counted():
    records = (entry(), entry(1))
    v = forward(records[0])
    result = assess(records, (v, replace(v, id="second-id"), forward(records[1])))
    assert R.AMBIGUOUS_PLANNED_EXIT_VALIDATION in result.reason_codes and result.status is S.INSUFFICIENT_EVIDENCE


def test_orphan_or_duplicate_identity_invalidates_cohort():
    r = entry()
    result = assess((r, r), (forward(r), replace(forward(r), opportunity_id="orphan")))
    assert R.INVALID_COHORT in result.reason_codes


@pytest.mark.parametrize("field,value", [("config_version", "changed"), ("formula_version", "changed"),
                                        ("config_fingerprint", "0"*64), ("config_json", "{}")])
def test_policy_cost_version_binding(field, value):
    records = (entry(), entry(1))
    changed = replace(records[0], **{field: value})
    result = assess((changed, records[1]), tuple(forward(r) for r in records))
    assert result.status is S.INSUFFICIENT_EVIDENCE and R.CONFIG_MISMATCH in result.reason_codes


def test_testnet_sources_cannot_be_production_edge_evidence():
    records = (entry(), entry(1))
    source = records[0].source_timings[0]
    payload = json.loads(source.details_json)
    payload["request"]["actual_url"] = "https://testnet.binance.vision/api/v3/depth"
    changed = replace(records[0], source_timings=(replace(source, details_json=exact_json(payload)), *records[0].source_timings[1:]))
    result = assess((changed, records[1]), tuple(forward(r) for r in records))
    assert R.NON_PRODUCTION_SOURCE in result.reason_codes


def test_capture_before_declared_registration_is_rejected():
    records = (entry(), entry(1))
    result = assess(records, policy=protocol(registered_at=OPENED))
    assert R.NON_PRODUCTION_SOURCE in result.reason_codes and result.status is S.INSUFFICIENT_EVIDENCE


def test_rejected_opportunity_cannot_be_promoted_by_later_profit():
    records = (entry(), entry(1))
    result = assess(tuple(replace(r, decision="REJECT") for r in records), tuple(forward(r) for r in records))
    assert R.NO_ELIGIBLE_ENTRY in result.reason_codes and result.status is S.INSUFFICIENT_EVIDENCE


def test_each_symbol_must_meet_its_own_requirements():
    records = (entry(), entry(1), entry(symbol="ETHUSDT"))
    result = assess(records, policy=protocol(symbols=("BTCUSDT", "ETHUSDT")))
    assert result.status is S.INSUFFICIENT_EVIDENCE
    assert result.symbols[0].net_return.count == 2 and result.symbols[1].net_return.count == 1


def test_low_decimal_context_does_not_change_summary_or_loss_threshold():
    records = (entry(), entry(1))
    outcomes = tuple(forward(r) for r in records)
    expected = assess(records, outcomes)
    with localcontext() as context:
        context.prec = 2
        actual = assess(records, outcomes)
    assert actual == expected


@pytest.mark.parametrize("value", [1.0, "invalid-decimal", "NaN", None])
def test_invalid_serialized_decimal_returns_structured_failure(value):
    records = (entry(), entry(1))
    v = forward(records[0])
    details = json.loads(v.details_json)
    details["calculation"]["net_profit"] = value
    bad = replace(v, details_json=json.dumps(details))
    result = assess(records, (bad, forward(records[1])))
    assert result.status is S.INSUFFICIENT_EVIDENCE and R.INVALID_VALIDATION_EVIDENCE in result.reason_codes


@pytest.mark.parametrize("field,value", [("registered_at", OPENED+timedelta(days=1)),
    ("registered_at", OPENED.replace(tzinfo=None)), ("min_entry_separation", timedelta(minutes=1)),
    ("min_entries_per_symbol", True), ("min_entry_days_per_symbol", 3),
    ("min_mean_net_return", 0.1), ("max_loss_return", D("NaN")), ("max_loss_return", D("-.1"))])
def test_invalid_protocol_cannot_silently_change_study_definition(field, value):
    with pytest.raises((ValueError, TypeError)):
        protocol(**{field: value})


INTEGRITY_CASES = (
    "original_entry_basis", "spot_entry_fee", "perp_entry_fee", "other_costs",
    "price_pnl_projection", "funding_total_projection", "spot_pnl_projection",
    "perp_pnl_projection", "basis_change_projection", "spot_exit_fee_projection",
    "perp_exit_fee_projection", "missing_price_pnl", "negative_exit_fee",
    "forecast_funding", "duplicate_funding_id", "duplicate_settlement",
    "funding_before_entry", "funding_after_exit", "funding_wrong_symbol",
    "funding_float", "funding_nan", "funding_nonpositive_mark",
    "duplicate_status", "duplicate_nested_field", "quality_wrong_container",
    "saved_before_exit",
)


def corrupt_validation(value, case):
    """Change saved evidence, keeping positive reported outcome where possible."""
    details = json.loads(value.details_json)
    if case in ("original_entry_basis", "spot_entry_fee", "perp_entry_fee", "other_costs"):
        value = replace(value, **{case: D("999")})
        if case.endswith("_fee"):
            details["fees"][case[:-4]] = "999"
        elif case == "other_costs":
            details["calculation"][case] = "999"
        else:
            details[case] = "999"
            details["calculation"]["entry_basis"] = "999"
    elif case.endswith("_projection"):
        value = replace(value, **{case.removesuffix("_projection"): D("999")})
    elif case == "missing_price_pnl":
        value = replace(value, price_pnl=None)
        details["calculation"]["price_pnl"] = None
        details["decomposition"]["price_pnl"] = None
    elif case == "negative_exit_fee":
        value = replace(value, spot_exit_fee=D("-1"))
        details["fees"]["spot_exit"] = "-1"
    elif case.startswith("funding_") or case in ("forecast_funding", "duplicate_funding_id", "duplicate_settlement"):
        event = {"event_id": "observed-1", "symbol": value.symbol,
                 "settled_at": value.validation_time.isoformat(),
                 "funding_rate": "0", "settlement_mark_price": "100"}
        events = [event]
        if case == "forecast_funding":
            event["event_id"] = "FORECAST_ONLY:observed-1"
        elif case == "duplicate_funding_id":
            events.append({**event, "settled_at": (value.validation_time-timedelta(seconds=1)).isoformat()})
        elif case == "duplicate_settlement":
            events.append({**event, "event_id": "observed-2"})
        elif case == "funding_before_entry":
            event["settled_at"] = value.entry_time.isoformat()
        elif case == "funding_after_exit":
            event["settled_at"] = (value.validation_time+timedelta(seconds=1)).isoformat()
        elif case == "funding_wrong_symbol":
            event["symbol"] = "ETHUSDT"
        elif case == "funding_float":
            event["funding_rate"] = 0.0
        elif case == "funding_nan":
            event["funding_rate"] = "NaN"
        elif case == "funding_nonpositive_mark":
            event["settlement_mark_price"] = "0"
        details["funding_events"] = events
    elif case == "quality_wrong_container":
        details["exit_data_quality"]["issues"] = {}
    elif case == "saved_before_exit":
        value = replace(value, created_at=value.validation_time-timedelta(seconds=1))
    # Intentionally corrupted on-disk JSON may contain floats; bypass the
    # production writer, which correctly rejects them before persistence.
    body = json.dumps(details)
    if case == "duplicate_status":
        body = '{"status":"INVALID",' + body[1:]
    elif case == "duplicate_nested_field":
        body = body.replace('"net_profit":', '"net_profit":"-999","net_profit":', 1)
    return replace(value, details_json=body)


@pytest.mark.parametrize("case", INTEGRITY_CASES)
def test_positive_summary_requires_complete_untampered_exit_evidence(case):
    records = (entry(), entry(1))
    bad = corrupt_validation(forward(records[0]), case)
    result = assess(records, (bad, forward(records[1])))
    assert result.status is S.INSUFFICIENT_EVIDENCE
    assert R.INVALID_VALIDATION_EVIDENCE in result.reason_codes
    assert result.symbols[0].profit.count == 1
    assert result.symbols[0].profit.null_count == 1
    assert result.symbols[0].selected_validation_ids == (forward(records[1]).id,)
