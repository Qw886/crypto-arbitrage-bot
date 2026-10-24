"""Bounded preflight recovery never resubmits an existing execution intent."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError


def ready(fixture, tmp_path):
    bot, transport, state = fixture
    acceptance = FundingAcceptance(bot, tmp_path / "acceptance.json", max_preflight_attempts=3)
    assert acceptance.tick() == "WAITING"
    state["now"] = state["settlement"] - timedelta(seconds=180)
    return bot, transport, state, acceptance


def fail_tick(acceptance):
    try:
        return acceptance.tick()
    except Exception as error:
        return acceptance._handle_failure(error)


@pytest.mark.parametrize("kind", [ExecutionErrorKind.NETWORK_ERROR, ExecutionErrorKind.RATE_LIMITED])
def test_transient_preflight_waits_then_enters_once(live_clock_bot, tmp_path, monkeypatch, kind):
    bot, transport, state, acceptance = ready(live_clock_bot, tmp_path)
    real_plan = bot._plan
    plan = Mock(side_effect=GatewayError(kind, "safe network failure"))
    monkeypatch.setattr(bot, "_plan", plan)
    assert fail_tick(acceptance) == "WAITING"
    attempt = acceptance.state["attempt_id"]
    assert acceptance.state["last_preflight_failure"]["reason"] == kind.value
    requests = len(transport.requests)
    for _ in range(10):
        assert acceptance.tick() == "WAITING"
    assert plan.call_count == 1 and len(transport.requests) == requests and transport.orders == {}
    state["now"] += timedelta(seconds=15)
    monkeypatch.setattr(bot, "_plan", real_plan)
    assert acceptance.tick() == "HOLDING"
    assert acceptance.state["attempt_id"] == attempt
    assert len(transport.orders) == 2 and acceptance.state["preflight_attempts"] == 2
    assert acceptance.tick() == "HOLDING" and len(transport.orders) == 2


def test_repeated_network_failure_is_bounded(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = ready(live_clock_bot, tmp_path)
    plan = Mock(side_effect=GatewayError(ExecutionErrorKind.NETWORK_ERROR, "timeout"))
    monkeypatch.setattr(bot, "_plan", plan)
    for expected in ("WAITING", "WAITING", "ABORTED"):
        assert fail_tick(acceptance) == expected
        state["now"] += timedelta(seconds=15)
    assert plan.call_count == 3 and transport.orders == {} and bot.journal.plans() == ()
    assert acceptance.state["reason"] == "NETWORK_ERROR"
    assert acceptance.tick() == "ABORTED" and plan.call_count == 3


def test_restart_retains_attempt_and_retry_budget(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = ready(live_clock_bot, tmp_path)
    plan = Mock(side_effect=GatewayError(ExecutionErrorKind.NETWORK_ERROR, "timeout"))
    monkeypatch.setattr(bot, "_plan", plan)
    assert fail_tick(acceptance) == "WAITING"
    restarted = FundingAcceptance(bot, acceptance.report_path, max_preflight_attempts=3)
    assert restarted.state["attempt_id"] == acceptance.state["attempt_id"]
    assert restarted.tick() == "WAITING" and plan.call_count == 1
    for expected in ("WAITING", "ABORTED"):
        state["now"] += timedelta(seconds=15)
        assert fail_tick(restarted) == expected
    assert plan.call_count == 3 and transport.orders == {}


def test_restart_cannot_reset_retry_policy(live_clock_bot, tmp_path):
    bot, _, _, acceptance = ready(live_clock_bot, tmp_path)
    with pytest.raises(ValueError, match="policy mismatch"):
        FundingAcceptance(bot, acceptance.report_path, max_preflight_attempts=5)


@pytest.mark.parametrize("reason", ["ACCOUNT_NOT_READY", "INSUFFICIENT_TESTNET_BALANCE",
                                    "UNRESOLVED_EXECUTION_BLOCKS_NEW_ENTRY"])
def test_hard_guard_never_retries(live_clock_bot, tmp_path, monkeypatch, reason):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=ExecutionGuardError(reason)))
    assert fail_tick(acceptance) == "ABORTED"
    assert acceptance.state["reason"] == reason and transport.orders == {}


@pytest.mark.parametrize("codes,expected", [(["SAMPLING_SKEW_TOO_LARGE"], "WAITING"),
    (["STALE_SPOT_DATA", "STALE_PERP_DATA"], "WAITING"),
    (["CROSSED_SPOT_BOOK"], "ABORTED"), (["INVALID_PRICE"], "ABORTED"),
    (["MISSING_FUNDING"], "ABORTED"), ([], "ABORTED")])
def test_quality_retry_requires_specific_transient_evidence(live_clock_bot, tmp_path, monkeypatch, codes, expected):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    error = ExecutionGuardError("INVALID_TESTNET_MARKET_DATA", diagnostics={"quality_reason_codes": codes})
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=error))
    assert fail_tick(acceptance) == expected and transport.orders == {}


def test_retry_cannot_cross_entry_deadline(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = ready(live_clock_bot, tmp_path)
    state["now"] = state["settlement"] - timedelta(seconds=70)
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=GatewayError(ExecutionErrorKind.NETWORK_ERROR, "timeout")))
    assert fail_tick(acceptance) == "ABORTED" and transport.orders == {}


@pytest.mark.parametrize("kind,expected", [("NETWORK_ERROR", "WAITING"), ("RATE_LIMITED", "WAITING"),
                                          ("INVALID_RESPONSE", "ABORTED"), (None, "ABORTED")])
def test_missing_payload_requires_network_source_evidence(live_clock_bot, tmp_path, monkeypatch, kind, expected):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    error = ExecutionGuardError("INVALID_TESTNET_MARKET_DATA", diagnostics={
        "quality_reason_codes": ["INCOMPLETE_SNAPSHOT", "SOURCE_ERROR", "MISSING_FUNDING"],
        "sources": {"premium_index": {"error_kind": kind}},
    })
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=error))
    assert fail_tick(acceptance) == expected and transport.orders == {}


def test_submission_uncertainty_never_retries(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, _, acceptance = ready(live_clock_bot, tmp_path)
    error = GatewayError(ExecutionErrorKind.NETWORK_ERROR, "ambiguous", submission_may_have_reached_exchange=True)
    monkeypatch.setattr(bot, "_plan", Mock(side_effect=error))
    assert fail_tick(acceptance) == "ABORTED" and transport.orders == {}


def test_claim_before_crash_prevents_another_preflight(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = ready(live_clock_bot, tmp_path)
    assert acceptance.tick() == "HOLDING"
    acceptance._save("WAITING", next_preflight_at=state["now"] + timedelta(seconds=15))
    plan = Mock(side_effect=AssertionError("must not send entry again"))
    monkeypatch.setattr(bot, "_plan", plan)
    assert acceptance.tick() == "HOLDING"
    plan.assert_not_called()
    assert len(transport.orders) == 2


@pytest.mark.parametrize("kwargs", [{"max_preflight_attempts": 0}, {"max_preflight_attempts": 6},
    {"max_preflight_attempts": True}, {"preflight_retry_seconds": 0}, {"preflight_retry_seconds": 61},
    {"max_wait_seconds": 28801}, {"max_wait_seconds": True}])
def test_policy_has_strict_bounds(live_clock_bot, tmp_path, kwargs):
    bot, _, _ = live_clock_bot
    with pytest.raises(ValueError):
        FundingAcceptance(bot, tmp_path / "report.json", **kwargs)


def test_six_hour_wait_is_bounded_without_polling_network(live_clock_bot, tmp_path):
    bot, transport, state = live_clock_bot
    state["settlement"] = state["now"] + timedelta(hours=6)
    acceptance = FundingAcceptance(bot, tmp_path / "report.json", max_wait_seconds=21600)
    assert acceptance.tick() == "WAITING"
    count = len(transport.requests)
    assert acceptance.tick() == "WAITING" and len(transport.requests) == count
    assert transport.orders == {}
