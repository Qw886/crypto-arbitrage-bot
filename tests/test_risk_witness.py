"""Witness captures explicitly mocked engineering runs; never claims native proof."""

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import risk_witness as module
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.risk_witness import (
    RiskWitness,
    WitnessError,
    WitnessFailure,
    load_witness,
    read_witness_events,
)
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

D = Decimal


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline witness test")))


@pytest.fixture
def example(tmp_path):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path / "bot")
    witness = RiskWitness.register(tmp_path / "witness", bot.settings, bot.spot, bot.perp,
        attempt_id="witness-example", symbol="ETHUSDT", case=ExecutionRiskCase.NORMAL_ENTRY_EXIT,
        duration=timedelta(minutes=10))
    return bot, transport, witness


def plan_for(bot, witness, **changes):
    return bot._plan("ETHUSDT", D(".01"), changes.get("attempt_id", witness.registration.attempt_id))


def test_registration_is_prior_bounded_redacted_and_performs_no_io_to_exchange(example):
    bot, transport, witness = example
    assert transport.requests == []
    assert witness.registration.origin == "OFFLINE_FIXTURE"
    assert load_witness(witness.registration.reference) == witness.registration
    data = witness.registration.reference.path.read_text(encoding="utf-8")
    for value in ("api_key", "api_secret", "signature", "raw_response"):
        assert value not in data
    assert bot.journal.plans() == ()


def test_context_does_not_submit_or_query_by_itself(example):
    bot, transport, witness = example
    with witness.observe(bot.spot, bot.perp):
        pass
    assert transport.requests == [] and transport.orders == {}
    events = read_witness_events(witness.registration)
    assert [e["kind"] for e in events] == ["SESSION_BEGIN", "SESSION_END"]


def test_mock_round_trip_capture_and_immutable_trace_export(example, tmp_path):
    bot, transport, witness = example
    original = bot.spot.place_ioc_limit_order
    with witness.observe(bot.spot, bot.perp):
        plan = plan_for(bot, witness)
        assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
        assert bot.coordinator.close_pair(plan).state is PairExecutionState.CLOSED
    assert bot.spot.place_ioc_limit_order == original
    events = read_witness_events(witness.registration)
    assert sum(e["kind"] == "POST_RESULT" for e in events) == 4
    assert len(transport.orders) == 4
    reference = witness.export_trace(tmp_path / "trace.json")
    data = json.loads(reference.path.read_text(encoding="utf-8"))
    assert data["origin"] == "OFFLINE_FIXTURE" and data["registration"]["sha256"] == witness.registration.reference.sha256
    assert all(isinstance(e["receipt"]["executed_quantity"], str) for e in data["events"] if e["kind"] == "POST_RESULT")
    with pytest.raises(WitnessError):
        witness.export_trace(reference.path)


def test_current_attempt_cannot_be_registered_retrospectively(example, tmp_path):
    bot, _, witness = example
    plan = plan_for(bot, witness)
    assert bot.journal.create(plan)
    with pytest.raises(WitnessError) as error:
        RiskWitness.register(tmp_path / "new", bot.settings, bot.spot, bot.perp,
            attempt_id=plan.attempt_id, symbol=plan.symbol, case=ExecutionRiskCase.NORMAL_ENTRY_EXIT,
            duration=timedelta(minutes=10))
    assert error.value.code is WitnessFailure.ATTEMPT_ALREADY_EXISTS


def test_same_registration_cannot_be_overwritten(example, tmp_path):
    bot, _, witness = example
    path = witness.registration.reference.path
    before = path.read_bytes()
    with pytest.raises(WitnessError) as error:
        RiskWitness.register(path.parent.parent, bot.settings, bot.spot, bot.perp,
            attempt_id=witness.registration.attempt_id, symbol="ETHUSDT", case=ExecutionRiskCase.NORMAL_ENTRY_EXIT,
            duration=timedelta(minutes=10))
    assert error.value.code is WitnessFailure.REGISTRATION_CONFLICT and path.read_bytes() == before


@pytest.mark.parametrize("change", ["edit", "delete", "rename", "partial"])
def test_event_chain_detects_changed_missing_and_partial_records(example, change):
    bot, _, witness = example
    with witness.observe(bot.spot, bot.perp):
        pass
    first = witness.registration.events_directory / "00000.json"
    if change == "edit":
        data = json.loads(first.read_text(encoding="utf-8"))
        data["kind"] = "CHANGED"
        first.write_text(exact_json(data), encoding="utf-8")
    elif change == "delete":
        first.unlink()
    elif change == "rename":
        first.rename(first.with_name("00099.json"))
    else:
        first.write_text('{"partial":', encoding="utf-8")
    with pytest.raises(WitnessError):
        read_witness_events(witness.registration)


def test_unexpected_interruption_does_not_export_complete_evidence(example, tmp_path, monkeypatch):
    bot, _, witness = example
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt), witness.observe(bot.spot, bot.perp):
        bot.coordinator.open_pair(plan_for(bot, witness))
    assert witness.broken
    with pytest.raises(WitnessError):
        witness.export_trace(tmp_path / "invalid.json")


def test_ack_delivery_interruption_is_observed_after_receipt_not_perp_submission(example, monkeypatch):
    bot, transport, witness = example
    monkeypatch.setattr(bot.spot, "get_order", Mock(side_effect=GatewayError(ExecutionErrorKind.NETWORK_ERROR,
                                                                         "controlled query unavailable")))
    def interrupt(receipt):
        raise GatewayError(ExecutionErrorKind.NETWORK_ERROR, "local fake delivery loss")
    with witness.observe(bot.spot, bot.perp, after_spot_ack=interrupt):
        plan = plan_for(bot, witness)
        result = bot.coordinator.open_pair(plan)
    assert result.state is PairExecutionState.RECOVERY_REQUIRED
    kinds = [e["kind"] for e in read_witness_events(witness.registration)]
    assert kinds.index("POST_RESULT") < kinds.index("LOCAL_ACK_DELIVERY_INTERRUPT")
    assert set(transport.orders) == {plan.spot_entry.client_order_id}


def test_resolved_ack_delivery_loss_can_safely_complete_the_hedge(example):
    bot, transport, witness = example
    def interrupt(receipt):
        raise RuntimeError("controlled local callback failure after a real mock receipt")
    with witness.observe(bot.spot, bot.perp, after_spot_ack=interrupt):
        result = bot.coordinator.open_pair(plan_for(bot, witness))
    assert result.state is PairExecutionState.HEDGED and len(transport.orders) == 2
    events = read_witness_events(witness.registration)
    assert any(e["kind"] == "QUERY_RESULT" for e in events)


def test_restart_does_not_erase_prior_incomplete_capture(example, tmp_path, monkeypatch):
    bot, _, witness = example
    with pytest.raises(KeyboardInterrupt), witness.observe(bot.spot, bot.perp):
        raise KeyboardInterrupt
    restored = RiskWitness(load_witness(witness.registration.reference), witness.settings)
    assert restored.broken
    with restored.observe(bot.spot, bot.perp):
        pass
    with pytest.raises(WitnessError):
        restored.export_trace(tmp_path / "cannot-promote.json")


def test_structured_unknown_error_never_leaks_error_message(example, monkeypatch):
    bot, _, witness = example
    secret_message = "signed-url=SECRET and raw body ORDERBOOK DATA"
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", Mock(side_effect=GatewayError(
        ExecutionErrorKind.SUBMISSION_UNKNOWN, secret_message, submission_may_have_reached_exchange=True)))
    with witness.observe(bot.spot, bot.perp):
        result = bot.coordinator.open_pair(plan_for(bot, witness))
    assert result.state is PairExecutionState.RECOVERY_REQUIRED
    all_text = "".join(p.read_text(encoding="utf-8") for p in witness.registration.events_directory.iterdir())
    assert "SECRET" not in all_text and "ORDERBOOK" not in all_text and "SUBMISSION_UNKNOWN" in all_text


def test_trace_failure_before_entry_blocks_gateway_call(example, monkeypatch):
    bot, transport, witness = example
    original = witness._event
    def event(kind, **kwargs):
        if kind == "POST_BEGIN":
            raise OSError("simulated disk failure")
        return original(kind, **kwargs)
    monkeypatch.setattr(witness, "_event", event)
    with witness.observe(bot.spot, bot.perp):
        result = bot.coordinator.open_pair(plan_for(bot, witness))
    assert result.state is PairExecutionState.FAILED and transport.orders == {}
    assert witness.broken


def test_trace_failure_cannot_prevent_safety_exit_or_hide_receipt(example, monkeypatch):
    bot, transport, witness = example
    plan = plan_for(bot, witness)
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    original = witness._event
    def event(kind, **kwargs):
        if kind.startswith("POST_"):
            raise OSError("simulated reducing observation failure")
        return original(kind, **kwargs)
    monkeypatch.setattr(witness, "_event", event)
    with witness.observe(bot.spot, bot.perp):
        result = bot.coordinator.close_pair(plan)
    assert result.state is PairExecutionState.CLOSED and len(transport.orders) == 4
    assert witness.broken


def test_another_attempt_cannot_use_this_entry_registration(example):
    bot, transport, witness = example
    with witness.observe(bot.spot, bot.perp):
        result = bot.coordinator.open_pair(plan_for(bot, witness, attempt_id="other-unregistered-attempt"))
    assert result.state is PairExecutionState.FAILED and transport.orders == {}


def test_changed_policy_expiration_and_registration_pin_fail_closed(example):
    _, _, witness = example
    changed = witness.settings.model_copy(update={"poll_interval_seconds": 30})
    with pytest.raises(WitnessError) as error:
        RiskWitness(witness.registration, changed)
    assert error.value.code is WitnessFailure.POLICY_CHANGED
    with pytest.raises(WitnessError) as error:
        RiskWitness(witness.registration, witness.settings, clock=lambda: witness.registration.expires_at)
    assert error.value.code is WitnessFailure.EXPIRED
    witness.registration.reference.path.write_text("{}", encoding="utf-8")
    with pytest.raises(WitnessError):
        load_witness(witness.registration.reference)


def test_finite_event_limit_is_enforced(example, monkeypatch):
    bot, _, witness = example
    monkeypatch.setattr(module, "MAX_EVENTS", 2)
    with witness.observe(bot.spot, bot.perp):
        pass
    with pytest.raises(WitnessError) as error:
        witness._event("ANOTHER_SESSION")
    assert error.value.code is WitnessFailure.EVENT_LIMIT


def test_restarting_loads_the_original_registry_and_hash_chain(example):
    bot, _, witness = example
    with witness.observe(bot.spot, bot.perp):
        pass
    resumed = RiskWitness(load_witness(witness.registration.reference), witness.settings)
    with resumed.observe(bot.spot, bot.perp):
        pass
    events = read_witness_events(witness.registration)
    assert len(events) == 4 and [e["sequence"] for e in events] == list(range(4))


@pytest.mark.parametrize("duration", [timedelta(0), timedelta(seconds=-1), timedelta(days=2)])
def test_unbounded_registration_is_rejected(example, tmp_path, duration):
    bot, _, _ = example
    with pytest.raises(WitnessError) as error:
        RiskWitness.register(tmp_path / "bad", bot.settings, bot.spot, bot.perp, attempt_id="bad-duration",
            symbol="ETHUSDT", case=ExecutionRiskCase.NORMAL_ENTRY_EXIT, duration=duration)
    assert error.value.code is WitnessFailure.INVALID_REGISTRATION
