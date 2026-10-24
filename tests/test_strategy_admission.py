"""Independent research + engineering conjunction; temporary archives only."""

import hashlib
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED
from test_registered_study import Cohort
from test_strategy_study import INTEGRITY_CASES, corrupt_validation

from crypto_arbitrage_bot.analytics.registered_study import StudyReadStatus
from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.config.testnet import StrategyAdmissionSettings
from crypto_arbitrage_bot.execution import strategy_admission as module
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.risk_review import (
    RiskReviewError,
    execution_policy_fingerprint,
    implementation_fingerprint,
)
from crypto_arbitrage_bot.models.strategy_admission import (
    AdmissionReason as R,
)
from crypto_arbitrage_bot.models.strategy_admission import (
    AdmissionStatus,
    ExecutionRiskCase,
    ExecutionRiskReview,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

D = Decimal
AT = OPENED+timedelta(days=4)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("admission must not contact an exchange"))
    monkeypatch.setattr("socket.create_connection", forbidden)


@pytest.fixture
def qualified_inputs(tmp_path, monkeypatch):
    cohort = Cohort(tmp_path)
    cohort.add()
    cohort.add(1)
    current = cohort.add(4, capture=False, validate=False)
    casefile = tmp_path / "synthetic-review.json"
    casefile.write_text(exact_json({"scope": "OFFLINE_QUALIFICATION_FORMAT_FIXTURE"}), encoding="utf-8")
    config = StrategyAdmissionSettings(
        study=StudyRegistrationSettings(path=cohort.study.path, sha256=cohort.study.sha256),
        execution_review=StudyRegistrationSettings(path=casefile, sha256=hashlib.sha256(casefile.read_bytes()).hexdigest()))
    testnet = load_settings(Path("docs/testnet-bot.toml")).testnet.model_copy(update={"admission": config})
    settings = cohort.settings.model_copy(update={"testnet": testnet})
    risk = ExecutionRiskReview(config.execution_review.sha256, execution_policy_fingerprint(settings),
        implementation_fingerprint(), AT-timedelta(minutes=1), AT+timedelta(hours=1),
        (("BTCUSDT", D("2")),), tuple(("BTCUSDT", case, "synthetic-case") for case in ExecutionRiskCase))
    reader = Mock(return_value=risk)
    # Risk reader has separate native-schema tests. This fixture assumes its
    # verified output, and actually reads/assesses the registered SQLite cohort.
    monkeypatch.setattr(module, "read_execution_risk_review", reader)
    return settings, current, risk, reader, cohort


def review(inputs, **changes):
    settings, current, _, _, _ = inputs
    return module.review_automatic_entry(changes.get("settings", settings), changes.get("record", current),
                                         evaluated_at=changes.get("at", AT))


def test_market_pass_without_evidence_defaults_to_denied(qualified_inputs):
    settings, current, _, reader, _ = qualified_inputs
    settings = settings.model_copy(update={"testnet": settings.testnet.model_copy(update={"admission": None})})
    result = module.review_automatic_entry(settings, current, evaluated_at=AT)
    assert result.status is AdmissionStatus.DENIED and result.reason_codes == (R.NOT_CONFIGURED,)
    reader.assert_not_called()


def test_both_verified_evidence_layers_allow_preflight_only_and_no_math_changes(qualified_inputs):
    settings, current, _, reader, _ = qualified_inputs
    before = exact_json(current)
    result = review(qualified_inputs)
    assert result.eligible and result.reason_codes == (R.ALL_EVIDENCE_RULES_MET,)
    assert result.valid_until == AT+settings.research.data_quality.max_data_age
    assert result.quantity == D("2") and exact_json(current) == before
    reader.assert_called_once()
    assert module.refresh_entry_admission(result, settings, current, evaluated_at=AT) == result


@pytest.mark.parametrize("field,value", [("net_profit", D("0")), ("net_profit", D("-1")),
    ("net_return", D("0")), ("net_return", D("-1")), ("matched_quantity", None),
    ("matched_quantity", D("0")), ("matched_quantity", D("NaN")), ("decision", "OBSERVE"),
    ("data_quality_status", "INVALID"), ("snapshot_status", "INCOMPLETE"),
    ("created_at", AT+timedelta(seconds=1))])
def test_current_opportunity_failures_not_overruled_by_good_evidence(qualified_inputs, field, value):
    changed = replace(qualified_inputs[1], **{field: value})
    result = review(qualified_inputs, record=changed)
    assert not result.eligible and R.INVALID_OPPORTUNITY in result.reason_codes


def test_positive_funding_and_basis_not_sufficient(qualified_inputs):
    settings, current, _, _, _ = qualified_inputs
    settings = settings.model_copy(update={"testnet": settings.testnet.model_copy(update={"admission": None})})
    changed = replace(current, estimated_funding=D("100"), executable_entry_basis=D("100"))
    assert not module.review_automatic_entry(settings, changed, evaluated_at=AT).eligible


def test_original_study_record_cannot_be_reused_as_a_new_entry(qualified_inputs):
    record = qualified_inputs[4].repository.get_opportunity_by_id(qualified_inputs[4].read().opportunity_ids[0])
    assert R.INVALID_OPPORTUNITY in review(qualified_inputs, record=record).reason_codes


def test_relabeling_old_entry_creation_time_does_not_make_its_evidence_fresh(qualified_inputs):
    cohort = qualified_inputs[4]
    original = cohort.repository.get_opportunity_by_id(cohort.read().opportunity_ids[0])
    changed = replace(original, id="renamed-stale-entry", created_at=AT)
    assert R.INVALID_OPPORTUNITY in review(qualified_inputs, record=changed).reason_codes


def test_current_record_cost_json_must_match_its_claimed_hash(qualified_inputs):
    changed = replace(qualified_inputs[1], config_json="{}")
    assert R.STUDY_POLICY_MISMATCH in review(qualified_inputs, record=changed).reason_codes


def test_not_supported_study_cannot_be_overruled_by_qualified_execution(qualified_inputs):
    _, _, _, _, cohort = qualified_inputs
    cohort.sql("UPDATE opportunities SET decision='REJECT'")
    result = review(qualified_inputs)
    assert not result.eligible and R.STUDY_NOT_SUPPORTED in result.reason_codes


@pytest.mark.parametrize("case", INTEGRITY_CASES)
def test_corrupt_saved_exit_blocks_registered_study_and_automatic_admission(qualified_inputs, case):
    from crypto_arbitrage_bot.models.strategy_study import StudyReason, StudyStatus
    from crypto_arbitrage_bot.storage.forward_validation_repository import COLUMNS, _encode

    _, current, _, _, cohort = qualified_inputs
    identifier = cohort.read().assessment.symbols[0].selected_validation_ids[0]
    original = cohort.validations.get_forward_validation_by_id(identifier)
    changed = corrupt_validation(original, case)
    cohort.sql("UPDATE forward_validations SET " + ",".join(f"{name}=?" for name in COLUMNS)
               + " WHERE id=?", (*_encode(changed), identifier))
    record_before = exact_json(current)
    evidence = cohort.read()
    assert evidence.status is StudyReadStatus.READ
    assert evidence.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE
    assert StudyReason.INVALID_VALIDATION_EVIDENCE in evidence.assessment.reason_codes
    assert evidence.assessment.symbols[0].profit.null_count == 1
    admission = review(qualified_inputs)
    assert not admission.eligible and R.STUDY_NOT_SUPPORTED in admission.reason_codes
    assert exact_json(current) == record_before


def test_missing_study_and_risk_report_return_multiple_reasons(qualified_inputs):
    _, _, _, reader, cohort = qualified_inputs
    cohort.study.path.unlink()
    reader.side_effect = RiskReviewError(R.RISK_REVIEW_UNAVAILABLE)
    result = review(qualified_inputs)
    assert {R.STUDY_UNAVAILABLE, R.RISK_REVIEW_UNAVAILABLE}.issubset(result.reason_codes)


def test_reader_io_failure_does_not_allow_entry(qualified_inputs, monkeypatch):
    original = qualified_inputs[4].read()
    monkeypatch.setattr(module, "read_registered_study", Mock(return_value=replace(original, status=StudyReadStatus.FAILED, assessment=None)))
    assert R.STUDY_UNAVAILABLE in review(qualified_inputs).reason_codes


def test_full_scenario_policy_changes_invalidate_admission(qualified_inputs):
    settings = qualified_inputs[0]
    scenarios = {"BTCUSDT": settings.research.scenarios["BTCUSDT"].model_copy(update={"adverse_exit_basis_pct": D(".17")})}
    changed = settings.model_copy(update={"research": settings.research.model_copy(update={"scenarios": scenarios})})
    result = review(qualified_inputs, settings=changed)
    assert {R.STUDY_POLICY_MISMATCH, R.RISK_POLICY_MISMATCH}.issubset(result.reason_codes)


@pytest.mark.parametrize("case", list(ExecutionRiskCase))
def test_one_missing_engineering_branch_cannot_be_ignored(qualified_inputs, case):
    _, _, risk, reader, _ = qualified_inputs
    reader.return_value = replace(risk, cases=tuple(row for row in risk.cases if row[1] is not case))
    assert R.RISK_REVIEW_INCOMPLETE in review(qualified_inputs).reason_codes


@pytest.mark.parametrize("change,code", [("expires", R.RISK_REVIEW_EXPIRED), ("policy", R.RISK_POLICY_MISMATCH),
    ("implementation", R.IMPLEMENTATION_MISMATCH), ("quantity", R.QUANTITY_EXCEEDS_QUALIFICATION)])
def test_execution_review_scope_is_enforced(qualified_inputs, change, code):
    _, _, risk, reader, _ = qualified_inputs
    updates = {"expires": {"expires_at": AT}, "policy": {"policy_sha256": "0"*64},
               "implementation": {"implementation_sha256": "0"*64}, "quantity": {"qualified_quantities": (("BTCUSDT", D("1")),)}}
    reader.return_value = replace(risk, **updates[change])
    assert code in review(qualified_inputs).reason_codes


@pytest.mark.parametrize("change,code", [("time", R.RISK_REVIEW_EXPIRED), ("record", R.INVALID_OPPORTUNITY),
    ("quantity", R.INVALID_OPPORTUNITY), ("profit", R.INVALID_OPPORTUNITY), ("config", R.RISK_POLICY_MISMATCH),
    ("file", R.EVIDENCE_CHECK_FAILED)])
def test_preflight_cannot_reuse_stale_changed_or_other_entry_permission(qualified_inputs, change, code):
    settings, current, _, _, cohort = qualified_inputs
    result = review(qualified_inputs)
    at = AT
    if change == "time":
        at = result.valid_until
    elif change == "record":
        current = replace(current, id="other-id")
    elif change == "quantity":
        current = replace(current, matched_quantity=D("1"))
    elif change == "profit":
        current = replace(current, net_profit=D("1000"))
    elif change == "config":
        settings = settings.model_copy(update={"poll_interval_seconds": 30})
    else:
        cohort.study.path.write_text("{}", encoding="utf-8")
    refreshed = module.refresh_entry_admission(result, settings, current, evaluated_at=at)
    assert not refreshed.eligible and code in refreshed.reason_codes


def test_refresh_cannot_turn_a_denial_into_permission(qualified_inputs):
    settings = qualified_inputs[0].model_copy(update={"testnet": None})
    result = review(qualified_inputs, settings=settings)
    assert module.refresh_entry_admission(result, qualified_inputs[0], qualified_inputs[1], evaluated_at=AT) == result


def test_old_demo_default_research_rejects_without_orders(tmp_path):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    cycles = bot.run_once()
    assert all(c.status == "NO_ENTRY" for c in cycles) and transport.orders == {}


def test_synthetic_market_pass_is_blocked_by_real_default_guard(tmp_path, monkeypatch):
    from test_funding_lifecycle import live_clock_bot
    # Construct the same fictional profitable scenario, then remove only its
    # explicit offline admission assumption to exercise the production guard.
    bot, transport, state = live_clock_bot.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.review_automatic_entry", module.review_automatic_entry)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.refresh_entry_admission", module.refresh_entry_admission)
    plan = Mock(side_effect=AssertionError("admission must precede account/quote/order planning"))
    monkeypatch.setattr(bot, "_plan", plan)
    cycles = bot.run_once()
    assert all(c.status == "ENTRY_BLOCKED" and c.admission.reason_codes == (R.NOT_CONFIGURED,) for c in cycles)
    assert all(c.decision_reason_codes == ("SCENARIO_PASS_ALL_RULES",) for c in cycles)
    assert transport.orders == {} and bot.journal.plans() == ()
    plan.assert_not_called()


def test_missing_admission_does_not_block_existing_hedged_exit(tmp_path, monkeypatch):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    plan = bot._plan("ETHUSDT", D(".01"), "already-hedged-without-admission")
    assert bot.coordinator.open_pair(plan).state.value == "HEDGED"
    later = bot.clock()+timedelta(seconds=bot.config.holding_seconds+2)
    transport.clock = bot.clock = bot.collector.clock = bot.research_runner.clock = lambda: later
    guard = Mock(side_effect=AssertionError("admission cannot block a managed exit"))
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.review_automatic_entry", guard)
    bot.run_once()
    assert bot.journal.get(plan.attempt_id).state.value == "CLOSED" and len(transport.orders) == 4
    guard.assert_not_called()
