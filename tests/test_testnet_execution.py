import hashlib
import hmac
import json
from decimal import Decimal
from io import BytesIO
from urllib.parse import parse_qs
from urllib.request import ProxyHandler

import pytest

from crypto_arbitrage_bot.execution.testnet_client import (
    _BASE_URLS,
    ExecutionErrorKind,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetCredentials as Credentials,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionClient as Client,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionError as ExecutionError,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionPolicy as ExecutionPolicy,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder, OrderState
from crypto_arbitrage_bot.models.market import Side


class Response:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = json.dumps(payload).encode()

    def read(self, limit=-1):
        return self._body if limit < 0 else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Opener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def client(market, opener, *, clock_ms=lambda: 10_000):
    return Client(
        market,
        Credentials("test-api-key", "test-secret-do-not-print"),
        ExecutionPolicy(Decimal("1000")),
        opener=opener,
        clock_ms=clock_ms,
    )


def exchange_info(symbol="BTCUSDT"):
    return {"symbols": [{"symbol": symbol, "filters": [
        {"filterType": "PRICE_FILTER", "minPrice": "0.00", "maxPrice": "1000000.00", "tickSize": "0.10"},
        {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "10", "stepSize": "0.001"},
        {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
    ]}]}


def order_reply(symbol="BTCUSDT", cid="test-order-1", status="FILLED"):
    return {"symbol": symbol, "orderId": 123, "clientOrderId": cid, "status": status,
            "origQty": "0.001", "executedQty": "0.001", "cumQuote": "50",
            "avgPrice": "50000", "price": "50000.10", "side": "BUY",
            "type": "LIMIT", "timeInForce": "IOC"}


def test_from_env_requires_explicit_testnet_mode_and_separate_credentials():
    policy = ExecutionPolicy(Decimal("100"))
    env = {
        "CAB_EXECUTION_MODE": "TESTNET",
        "CAB_TESTNET_SPOT_API_KEY": "spot-test-key",
        "CAB_TESTNET_SPOT_API_SECRET": "spot-test-secret",
        "CAB_TESTNET_USDTM_API_KEY": "um-test-key",
        "CAB_TESTNET_USDTM_API_SECRET": "um-test-secret",
    }
    spot = Client.from_env(ExecutionMarket.SPOT, policy, environ=env, opener=Opener([]))
    um = Client.from_env(ExecutionMarket.USDTM, policy, environ=env, opener=Opener([]))
    assert spot._credentials.api_key == "spot-test-key"
    assert um._credentials.api_key == "um-test-key"
    assert "test-secret" not in repr(spot._credentials)
    with pytest.raises(ExecutionError) as error:
        Client.from_env(ExecutionMarket.SPOT, policy, environ={}, opener=Opener([]))
    assert error.value.kind is ExecutionErrorKind.DISABLED


def test_testnet_hosts_are_fixed_and_have_no_production_route():
    assert _BASE_URLS == {
        ExecutionMarket.SPOT: "https://testnet.binance.vision",
        ExecutionMarket.USDTM: "https://demo-fapi.binance.com",
    }
    assert all("binance.com" not in url or "demo-fapi.binance.com" == url.removeprefix("https://")
               for url in _BASE_URLS.values())


@pytest.mark.parametrize("market,expected_url", [
    (ExecutionMarket.SPOT, "https://testnet.binance.vision/api/v3/order"),
    (ExecutionMarket.USDTM, "https://demo-fapi.binance.com/fapi/v1/order"),
])
def test_ioc_order_uses_signed_testnet_url_and_returns_decimal_receipt(market, expected_url):
    opener = Opener([Response(exchange_info()), Response({"serverTime": 10_050}), Response(order_reply())])
    if market is ExecutionMarket.USDTM:
        opener.outcomes[-1] = Response(dict(order_reply(), reduceOnly=False, positionSide='BOTH'))
    api = client(market, opener)
    order = IocLimitOrder(market, "BTCUSDT", Side.BUY, Decimal("0.001"), Decimal("50000.10"), "test-order-1")
    receipt = api.place_ioc_limit_order(order)
    assert receipt.status is OrderState.FILLED
    assert receipt.executed_quantity == Decimal("0.001")
    assert receipt.cumulative_quote_quantity == Decimal("50")
    assert receipt.average_price == Decimal("50000")
    request, timeout = opener.requests[-1]
    assert request.full_url == expected_url
    assert request.get_method() == "POST"
    assert request.get_header("X-mbx-apikey") == "test-api-key"
    assert timeout == 5.0
    info_request = opener.requests[0][0]
    if market is ExecutionMarket.USDTM:
        assert info_request.full_url == "https://demo-fapi.binance.com/fapi/v1/exchangeInfo"
    else:
        assert info_request.full_url.endswith("/api/v3/exchangeInfo?symbol=BTCUSDT")
    body = request.data.decode()
    params = parse_qs(body)
    assert params["type"] == ["LIMIT"]
    assert params["timeInForce"] == ["IOC"]
    assert params["quantity"] == ["0.001"]
    assert params["price"] == ["50000.10"]
    signed_fields = body.rsplit("&signature=", 1)[0]
    expected_signature = hmac.new(b"test-secret-do-not-print", signed_fields.encode(), hashlib.sha256).hexdigest()
    assert params["signature"] == [expected_signature]


@pytest.mark.parametrize("quantity,price", [("0.0011", "50000.1"), ("0.001", "50000.11")])
def test_filter_misalignment_is_rejected_before_any_order_submission(quantity, price):
    opener = Opener([Response(exchange_info())])
    api = client(ExecutionMarket.SPOT, opener)
    request = IocLimitOrder(ExecutionMarket.SPOT, "BTCUSDT", Side.BUY,
                            Decimal(quantity), Decimal(price), "test-order-1")
    with pytest.raises(ExecutionError) as error:
        api.place_ioc_limit_order(request)
    assert error.value.kind is ExecutionErrorKind.ORDER_GUARD_REJECTED
    assert len(opener.requests) == 1


def test_testnet_notional_cap_is_checked_before_order_submission():
    opener = Opener([Response(exchange_info())])
    api = Client(
        ExecutionMarket.SPOT, Credentials("key", "secret"),
        ExecutionPolicy(Decimal("25")), opener=opener, clock_ms=lambda: 10_000,
    )
    order = IocLimitOrder(ExecutionMarket.SPOT, "BTCUSDT", Side.BUY,
                          Decimal("0.001"), Decimal("50000"), "test-order-1")
    with pytest.raises(ExecutionError, match="notional cap"):
        api.place_ioc_limit_order(order)
    assert len(opener.requests) == 1


def test_submission_timeout_is_marked_unknown_and_never_retried():
    opener = Opener([Response(exchange_info()), Response({"serverTime": 10_000}), TimeoutError()])
    api = client(ExecutionMarket.USDTM, opener)
    order = IocLimitOrder(ExecutionMarket.USDTM, "BTCUSDT", Side.SELL,
                          Decimal("0.001"), Decimal("49999.9"), "um-order-1")
    with pytest.raises(ExecutionError) as error:
        api.place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN
    assert error.value.submission_may_have_reached_exchange
    assert len(opener.requests) == 3


def test_private_order_query_is_scoped_to_symbol_and_client_order_id():
    opener = Opener([Response({"serverTime": 10_000}), Response(order_reply(cid="test-order-1")),
                     Response([{"symbol": "BTCUSDT", "orderId": 123, "qty": "0.001",
                                "commission": "0.000001", "commissionAsset": "BTC"}])])
    api = client(ExecutionMarket.SPOT, opener)
    receipt = api.get_order("BTCUSDT", "test-order-1")
    assert receipt.order_id == "123"
    assert receipt.base_commission == Decimal("0.000001")
    request, _ = opener.requests[-2]
    assert request.full_url.startswith("https://testnet.binance.vision/api/v3/order?")
    params = parse_qs(request.full_url.split("?", 1)[1])
    assert params["symbol"] == ["BTCUSDT"]
    assert params["origClientOrderId"] == ["test-order-1"]
    assert "signature" in params


def test_timeout_error_never_exposes_secret_or_signed_query():
    opener = Opener([TimeoutError("https://host?signature=leak")])
    api = client(ExecutionMarket.SPOT, opener)
    with pytest.raises(ExecutionError) as error:
        api._request("time")
    assert error.value.kind is ExecutionErrorKind.NETWORK_ERROR
    assert "leak" not in str(error.value)
    assert "test-secret-do-not-print" not in str(error.value)


def test_default_transport_inherits_https_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8888")
    monkeypatch.setenv("https_proxy", "http://proxy.example:8888")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    api = Client(ExecutionMarket.USDTM, Credentials("key", "secret"),
                 ExecutionPolicy(Decimal("100")))
    proxy = next(handler for handler in api._opener.handlers if isinstance(handler, ProxyHandler))
    assert proxy.proxies["https"] == "http://proxy.example:8888"


def test_rate_limit_is_structured_and_never_retried():
    from urllib.error import HTTPError

    error_response = HTTPError("https://testnet.invalid", 429, "rate", {}, BytesIO(b'{"code":-1003,"msg":"rate limit"}'))
    opener = Opener([error_response])
    api = client(ExecutionMarket.SPOT, opener)
    with pytest.raises(ExecutionError) as error:
        api._request("time")
    assert error.value.kind is ExecutionErrorKind.RATE_LIMITED
    assert len(opener.requests) == 1


def test_ioc_order_rejects_unsupported_symbols_and_float_amounts():
    with pytest.raises(ValueError, match="BTCUSDT and ETHUSDT"):
        IocLimitOrder(ExecutionMarket.SPOT, "BNBUSDT", Side.BUY,
                      Decimal("0.001"), Decimal("10"), "x")
    with pytest.raises(TypeError):
        IocLimitOrder(ExecutionMarket.SPOT, "BTCUSDT", Side.BUY, 0.001,
                      Decimal("10"), "x")
