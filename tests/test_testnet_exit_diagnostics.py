"""Offline exit diagnostics, failure isolation and uncertain-order safety."""

import json
from datetime import timedelta
from decimal import Decimal
from io import BytesIO
from ssl import SSLEOFError
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot
from test_testnet_exit_recovery import failed_exit as failed_exit

from crypto_arbitrage_bot.execution.demo import DemoResponse
from crypto_arbitrage_bot.execution.exit_recovery import recover_perp_exit
from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.models.calculation import TradeLeg


def opened(fixture, symbol="ETHUSDT", attempt="exit-diagnostic"):
    bot, transport, state = fixture
    quantity = Decimal(".01") if symbol == "ETHUSDT" else Decimal(".001")
    plan = bot._plan(symbol, quantity, attempt)
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    return bot, transport, state, plan


def events(caplog, status):
    return [json.loads(record.message) for record in caplog.records
            if record.name == "crypto_arbitrage_bot.testnet" and record.message.startswith("{")
            and json.loads(record.message).get("status") == status]


@pytest.mark.parametrize("failure,cause", [(TimeoutError("secret timeout"), "TimeoutError"),
    (URLError(ConnectionRefusedError("secret address")), "ConnectionRefusedError"),
    (URLError(SSLEOFError("secret tunnel")), "SSLEOFError")])
def test_exit_open_failure_has_endpoint_cause_and_no_order(live_clock_bot, monkeypatch, caplog, failure, cause):
    bot, transport, state, plan = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    def broken(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/account":
            raise failure
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", broken)
    bot._managed_exits()
    event = events(caplog, "EXIT_BLOCKED")[-1]
    assert event["reason"] == "NETWORK_ERROR" and event["attempt_id"] == plan.attempt_id
    assert event["stage"] == "EXIT_PREFLIGHT"
    details = event["diagnostics"]
    assert details["endpoint"] == "/api/v3/account" and details["operation"] == "account"
    assert details["phase"] == "OPEN" and details["transport_error_type"] == cause
    assert details["method"] == "GET" and details["elapsed_ms"] >= 0
    assert details["timeout_seconds"] == str(bot.spot.policy.timeout_seconds)
    assert "secret" not in caplog.text and "signature=" not in caplog.text
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).state is State.HEDGED


def test_exit_read_timeout_is_distinct_from_open(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, _ = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    class SlowResponse(DemoResponse):
        def read(self, limit):
            raise TimeoutError("body not available")
    def slow(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/account":
            return SlowResponse({})
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", slow)
    bot._managed_exits()
    assert events(caplog, "EXIT_BLOCKED")[-1]["diagnostics"]["phase"] == "READ"
    assert len(transport.orders) == 2


@pytest.mark.parametrize("http_status,api_code,kind", [(429, -1003, "RATE_LIMITED"),
    (503, -1000, "HTTP_REJECTED"), (400, -1021, "HTTP_REJECTED")])
def test_http_and_api_codes_retained_without_response_body(live_clock_bot, monkeypatch, caplog, http_status, api_code, kind):
    bot, transport, state, _ = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    def rejected(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/account":
            raise HTTPError(request.full_url, http_status, "bad", {},
                BytesIO(json.dumps({"code": api_code, "msg": "SIGNED_PRIVATE_RESPONSE"}).encode()))
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", rejected)
    bot._managed_exits()
    event = events(caplog, "EXIT_BLOCKED")[-1]
    assert event["reason"] == kind
    assert event["diagnostics"]["http_status"] == http_status
    assert event["diagnostics"]["api_code"] == api_code
    assert "SIGNED_PRIVATE_RESPONSE" not in caplog.text and "signature=" not in caplog.text
    assert len(event["diagnostics"]) < 20 and len(transport.orders) == 2


def test_time_sync_failure_is_not_mislabeled_as_account_send(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, _ = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    def failed_time(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/time":
            raise TimeoutError("time request failed")
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", failed_time)
    bot._managed_exits()
    details = events(caplog, "EXIT_BLOCKED")[-1]["diagnostics"]
    assert details["operation"] == "time" and details["endpoint"] == "/api/v3/time"
    assert details["parent_operation"] == "account" and details["phase"] == "OPEN"
    assert len(transport.orders) == 2


def test_invalid_json_has_decode_phase(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, _ = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    def invalid(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/account":
            response = DemoResponse({})
            response.body = b"PRIVATE_INVALID_JSON"
            return response
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", invalid)
    bot._managed_exits()
    event = events(caplog, "EXIT_BLOCKED")[-1]
    assert event["reason"] == "INVALID_RESPONSE" and event["diagnostics"]["phase"] == "DECODE"
    assert "PRIVATE_INVALID_JSON" not in caplog.text and len(transport.orders) == 2


def test_invalid_time_response_keeps_time_operation(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, _ = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    def invalid_time(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/time":
            return DemoResponse({"serverTime": "invalid"})
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", invalid_time)
    bot._managed_exits()
    details = events(caplog, "EXIT_BLOCKED")[-1]["diagnostics"]
    assert details["operation"] == "time" and details["phase"] == "DECODE"
    assert details["parent_operation"] == "account" and len(transport.orders) == 2


def test_http_error_body_timeout_keeps_endpoint_and_http_status(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, _ = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    class SlowErrorBody(BytesIO):
        def read(self, *args):
            raise TimeoutError("PRIVATE_HTTP_BODY")
    def failed(request, **kwargs):
        if urlsplit(request.full_url).path == "/api/v3/account":
            raise HTTPError(request.full_url, 503, "bad", {}, SlowErrorBody())
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", failed)
    bot._managed_exits()
    details = events(caplog, "EXIT_BLOCKED")[-1]["diagnostics"]
    assert details["endpoint"] == "/api/v3/account" and details["http_status"] == 503
    assert details["phase"] == "READ" and details["transport_error_type"] == "TimeoutError"
    assert "PRIVATE_HTTP_BODY" not in caplog.text and len(transport.orders) == 2


def test_five_transient_failures_then_close_without_duplicate_orders(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, plan = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    failures = 0
    def flaky(request, **kwargs):
        nonlocal failures
        if urlsplit(request.full_url).path == "/api/v3/account" and failures < 5:
            failures += 1
            raise TimeoutError("mock transient failure")
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", flaky)
    for _ in range(5):
        bot._managed_exits()
        assert bot.journal.get(plan.attempt_id).state is State.HEDGED and len(transport.orders) == 2
        state["now"] += timedelta(seconds=5)
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED and len(transport.orders) == 4
    bot._managed_exits()
    assert len(transport.orders) == 4 and len(events(caplog, "EXIT_BLOCKED")) == 5
    assert transport.positions[plan.symbol] == transport.base[plan.symbol] == 0


@pytest.mark.parametrize("stage", ["journal", "reconcile", "close"])
def test_one_btc_failure_does_not_block_eth_exit(live_clock_bot, monkeypatch, caplog, stage):
    bot, transport, state, btc = opened(live_clock_bot, "BTCUSDT", "btc-exit")
    _, _, _, eth = opened(live_clock_bot, "ETHUSDT", "eth-exit")
    state["now"] += timedelta(seconds=65)
    owner, attribute = (bot.journal, "get") if stage == "journal" else (
        (bot.coordinator, "reconcile") if stage == "reconcile" else (bot, "_close"))
    original = getattr(owner, attribute)
    def failed_btc(value):
        attempt = value if isinstance(value, str) else value.attempt_id
        if attempt == btc.attempt_id:
            raise ExecutionJournalError("private db path")
        return original(value)
    monkeypatch.setattr(owner, attribute, failed_btc)
    bot._managed_exits()
    assert bot.journal.get(eth.attempt_id).state is State.CLOSED
    assert len(transport.orders) == 6 and transport.positions["ETHUSDT"] == 0
    event = events(caplog, "EXIT_BLOCKED")[-1]
    assert event["attempt_id"] == btc.attempt_id and "private db path" not in caplog.text


def test_quality_error_retains_reasons_but_not_large_orderbook(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, plan = opened(live_clock_bot)
    assembly = bot.collector.collect_exit_depth(plan.symbol)
    state["now"] += timedelta(seconds=65)
    monkeypatch.setattr(bot.collector, "collect_exit_depth", Mock(return_value=assembly))
    bot._managed_exits()
    event = events(caplog, "EXIT_BLOCKED")[-1]
    assert event["reason"] == "INVALID_TESTNET_MARKET_DATA"
    assert "STALE_SPOT_DATA" in event["diagnostics"]["quality_reason_codes"]
    assert "raw_response" not in caplog.text and "bids" not in caplog.text
    assert len(json.dumps(event)) < 6000 and len(transport.orders) == 2


def test_recovery_unknown_submission_keeps_intent_and_logs_context(failed_exit, monkeypatch, caplog):
    bot, transport, plan = failed_exit
    error = GatewayError(ExecutionErrorKind.SUBMISSION_UNKNOWN, "secret signed body",
                         submission_may_have_reached_exchange=True)
    error.endpoint, error.method, error.operation, error.phase = "/fapi/v1/order", "POST", "order", "READ"
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", Mock(side_effect=error))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    event = events(caplog, "RECOVERY_SUBMISSION_UNCONFIRMED")[-1]
    assert event["stage"] == "RECOVERY_SUBMISSION" and event["reason"] == "SUBMISSION_UNKNOWN"
    assert event["diagnostics"]["submission_may_have_reached_exchange"]
    assert "secret signed body" not in caplog.text and len(transport.orders) == 4
    # The durable intent is queried before another send.
    with pytest.raises(KeyError):
        recover_perp_exit(bot, plan)
    assert bot.perp.place_ioc_limit_order.call_count == 1


def test_recovery_invalid_quote_is_logged_without_submission(failed_exit, monkeypatch, caplog):
    bot, transport, plan = failed_exit
    assembly = bot.collector.collect_exit_depth(plan.symbol, markets=(TradeLeg.PERP_EXIT.market,))
    monkeypatch.setattr(bot.collector, "collect_exit_depth", Mock(return_value=assembly))
    now = bot.clock()
    bot.clock = lambda: now + timedelta(minutes=1)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert events(caplog, "RECOVERY_BLOCKED")[-1]["stage"] == "RECOVERY_QUOTE"
    assert len(transport.orders) == 4


def test_guard_reason_is_retained_without_changing_exposure(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, plan = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    monkeypatch.setattr(bot, "_close", Mock(side_effect=ExecutionGuardError("EXIT_ACCOUNT_NOT_READY")))
    bot._managed_exits()
    assert events(caplog, "EXIT_BLOCKED")[-1]["reason"] == "EXIT_ACCOUNT_NOT_READY"
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).state is State.HEDGED


def test_recovery_exception_for_btc_still_allows_eth_exit(live_clock_bot, monkeypatch, caplog):
    bot, transport, state, btc = opened(live_clock_bot, "BTCUSDT", "btc-recovery")
    _, _, _, eth = opened(live_clock_bot, "ETHUSDT", "eth-normal-exit")
    bot.journal.update(btc.attempt_id, state=State.RECOVERY_REQUIRED)
    state["now"] += timedelta(seconds=65)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.recover_perp_exit",
                        Mock(side_effect=GatewayError(ExecutionErrorKind.NETWORK_ERROR, "mock failure")))
    bot._managed_exits()
    assert bot.journal.get(eth.attempt_id).state is State.CLOSED
    assert bot.journal.get(btc.attempt_id).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 6
    assert events(caplog, "RECOVERY_BLOCKED")[-1]["attempt_id"] == btc.attempt_id


def test_continuous_exit_failures_use_existing_poll_wait(live_clock_bot, monkeypatch):
    bot, transport, state, plan = opened(live_clock_bot)
    state["now"] += timedelta(seconds=65)
    original = transport.open
    failures, waits = 0, []
    def flaky(request, **kwargs):
        nonlocal failures
        if urlsplit(request.full_url).path == "/api/v3/account" and failures < 5:
            failures += 1
            raise TimeoutError("temporary read-only failure")
        return original(request, **kwargs)
    def wait(seconds):
        assert seconds == bot.settings.poll_interval_seconds and seconds > 0
        waits.append(seconds)
        state["now"] += timedelta(seconds=seconds)
        if len(waits) == 6:
            bot.stop.set()
            return True
        return False
    monkeypatch.setattr(transport, "open", flaky)
    monkeypatch.setattr(bot.research_runner, "run_once", Mock(return_value=()))
    monkeypatch.setattr(bot.stop, "wait", wait)
    bot.run_forever()
    assert len(waits) == 6 and failures == 5
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED and len(transport.orders) == 4
