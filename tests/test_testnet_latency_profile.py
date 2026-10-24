"""Timing observer cannot change payloads, errors or routes, or log secrets."""

import importlib.util
import json
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.runner import ExecutionGuardError


@pytest.fixture
def observer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("work").resolve()))
    spec = importlib.util.spec_from_file_location("latency_profile", Path("work/testnet_latency_profile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response(BytesIO):
    status = 200
    headers = {"Content-Encoding": "identity"}


def test_metadata_has_no_queries_headers_or_response_body(observer):
    payload = b'{"not_for_logs":"response-secret"}'
    original = SimpleNamespace(open=lambda *args, **kwargs: Response(payload))
    profile = observer.Profile()
    wrapped = observer.TimedOpener(original, profile)
    request = Request("https://demo-fapi.binance.com/fapi/v2/account?signature=signed-secret&timestamp=1",
                      headers={"X-MBX-APIKEY": "private-key"})
    with wrapped.open(request, timeout=5) as response:
        assert response.status == 200 and response.read() == payload
    text = json.dumps(profile.requests, default=str)
    assert all(value not in text for value in ("signed-secret", "private-key", "response-secret", "timestamp=1"))
    assert profile.requests[0]["body_bytes"] == len(payload)
    assert profile.requests[0]["path"] == "/fapi/v2/account"
    assert "open_ms" in profile.requests[0] and "read_ms" in profile.requests[0]


@pytest.mark.parametrize("url,method", [
    ("https://fapi.binance.com/fapi/v1/order", "POST"),
    ("https://demo-fapi.binance.com/fapi/v1/unknown", "GET"),
    ("http://demo-fapi.binance.com/fapi/v1/time", "GET"),
    ("https://demo-fapi.binance.com/fapi/v1/order", "DELETE"),
])
def test_observer_refuses_unknown_or_production_routes(observer, url, method):
    called = []
    opener = observer.TimedOpener(SimpleNamespace(open=lambda *args, **kwargs: called.append(args)), observer.Profile())
    with pytest.raises(ExecutionGuardError):
        opener.open(Request(url, method=method), timeout=5)
    assert called == []


def test_read_only_measurement_cannot_submit_order(observer):
    opener = observer.TimedOpener(None, observer.Profile(read_only=True))
    with pytest.raises(ExecutionGuardError):
        opener.open(Request("https://demo-fapi.binance.com/fapi/v1/order", method="POST"), timeout=5)


def test_open_error_unchanged_and_arbitrary_message_not_logged(observer):
    error = TimeoutError("signed-secret")
    def fail(*args, **kwargs):
        raise error
    profile = observer.Profile()
    opener = observer.TimedOpener(SimpleNamespace(open=fail), profile)
    with pytest.raises(TimeoutError) as caught:
        opener.open(Request("https://demo-fapi.binance.com/fapi/v1/time"), timeout=5)
    assert caught.value is error and "signed-secret" not in json.dumps(profile.requests, default=str)
    assert profile.requests[0]["error_type"] == "TimeoutError"


def test_read_error_unchanged_and_response_closed(observer):
    error = TimeoutError("body-secret")
    class FailingResponse(Response):
        def read(self, *args):
            raise error
    raw = FailingResponse()
    profile = observer.Profile()
    opener = observer.TimedOpener(SimpleNamespace(open=lambda *args, **kwargs: raw), profile)
    with pytest.raises(TimeoutError) as caught:
        with opener.open(Request("https://demo-fapi.binance.com/fapi/v1/time"), timeout=5) as response:
            response.read()
    assert caught.value is error and raw.closed
    assert profile.requests[0]["read_error_type"] == "TimeoutError"
    assert "body-secret" not in json.dumps(profile.requests, default=str)


def test_scopes_restore_phase_even_after_failure(observer):
    profile = observer.Profile()
    with pytest.raises(ValueError):
        with profile.scope("EXIT_PREFLIGHT"):
            with profile.scope("EXIT_SUBMISSION"):
                raise ValueError("private-message")
    assert profile.phase == "OTHER" and len(profile.stages) == 2
    assert "private-message" not in json.dumps(profile.stages, default=str)


def test_read_only_preflight_records_real_modules_and_submits_no_orders(observer, live_clock_bot):
    bot, transport, _ = live_clock_bot
    profile = observer.Profile(read_only=True)
    observer.attach(bot, profile)
    result = observer.preflight(bot, profile)
    assert result["orders_submitted"] is False and transport.orders == {}
    assert [s["stage"] for s in profile.stages] == ["EXIT_ACCOUNTS", "EXIT_RULES", "EXIT_MARKET_DATA"]
    assert all(r["method"] == "GET" for r in profile.requests)


def test_profiled_virtual_entry_exit_math_and_order_count_unchanged(observer, live_clock_bot):
    bot, transport, _ = live_clock_bot
    profile = observer.Profile()
    observer.attach(bot, profile)
    quantity = bot.research.sizing["ETHUSDT"].target_quantity
    pair = bot.round_trip("ETHUSDT", quantity)
    assert pair.state.value == "CLOSED" and len(transport.orders) == 4
    assert pair.spot_entry.average_price == Decimal("2000")
    assert pair.perp_entry.average_price == Decimal("2009")
    assert {s["stage"] for s in profile.stages} == {"ENTRY_PREFLIGHT", "ENTRY_SUBMISSION", "EXIT_PREFLIGHT", "EXIT_SUBMISSION"}
    assert sum(r["method"] == "POST" for r in profile.requests) == 4
