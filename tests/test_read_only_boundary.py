import ast
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import pytest

from crypto_arbitrage_bot.exchanges import endpoints, http_client
from crypto_arbitrage_bot.exchanges.endpoints import ENDPOINT_POLICIES, PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import (
    PublicGetRequest,
    prepare_get,
    validate_public_request,
)


@pytest.mark.parametrize("endpoint", list(PublicEndpoint))
def test_every_registered_public_endpoint_prepares_offline(endpoint):
    params = {"symbol": "BTCUSDT"} if ENDPOINT_POLICIES[endpoint].required_parameters else {}
    request = prepare_get(endpoint, params)
    assert request.method == "GET"
    assert request.url == endpoint.value
    assert request == validate_public_request("GET", endpoint.value, params)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_only_supported_symbols_pass(symbol):
    request = prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": symbol, "limit": 100})
    assert dict(request.params) == {"symbol": symbol, "limit": 100}


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "get", " GET", "GET\r\nPOST", ""])
def test_methods_other_than_exact_get_rejected(method):
    with pytest.raises(ValueError, match="GET"):
        validate_public_request(method, PublicEndpoint.SPOT_TIME.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.binance.com/api/v3/account",
        "https://api.binance.com/api/v3/order",
        "https://fapi.binance.com/fapi/v2/account",
        "https://fapi.binance.com/fapi/v1/order",
        "https://fapi.binance.com/fapi/v1/leverage",
        "https://api.binance.com/api/v3/depth",  # Different host from this route's whitelist.
        "http://api.binance.com/api/v3/time",
        "https://evil.example/api/v3/time",
        "https://api.binance.com.evil.example/api/v3/time",
        "https://api.binance.com@evil.example/api/v3/time",
        "https://user:password@api.binance.com/api/v3/time",
        "https://api.binance.com:443/api/v3/time",
        "https://api.binance.com:8443/api/v3/time",
        "https://api.binance.com/api/v3/time?symbol=BTCUSDT",
        "https://api.binance.com/api/v3/time?",
        "https://api.binance.com/api/v3/time#fragment",
        "https://api.binance.com/api/v3/time#",
        "https://api.binance.com/api/v3/time/../account",
        "https://api.binance.com/api/v3/%74ime",
        "https://api.binance.com/api/v3/time/",
        "https://api.binance.com/api/v3/time\r\nHost: evil.example",
        "/api/v3/time",
    ],
)
def test_nonwhitelisted_or_decorated_urls_rejected(url):
    with pytest.raises(ValueError, match="whitelisted"):
        validate_public_request("GET", url)


@pytest.mark.parametrize("symbol", ["BNBUSDT", "BTCUSDC", "btcusdt", "BTCUSDT ", "BTCUSDT&limit=5000", "BTCUSDT?x=y", None, 1, ["BTCUSDT"]])
def test_symbol_whitelist_and_query_injection_rejected(symbol):
    with pytest.raises(ValueError, match="symbol"):
        prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": symbol})


@pytest.mark.parametrize("endpoint", [e for e in PublicEndpoint if ENDPOINT_POLICIES[e].required_parameters])
def test_market_endpoints_require_explicit_supported_symbol(endpoint):
    with pytest.raises(ValueError, match="missing"):
        prepare_get(endpoint)


@pytest.mark.parametrize("name", ["apiKey", "secret", "signature", "timestamp", "recvWindow", "redirect", "headers", "url", "symbols", "symbol&limit"])
def test_unknown_or_credential_parameters_rejected(name):
    with pytest.raises(ValueError, match="not allowed"):
        prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "BTCUSDT", name: "x"})


@pytest.mark.parametrize("keyword", ["headers", "auth", "follow_redirects", "data", "json", "method", "base_url"])
def test_transport_options_and_request_bodies_are_not_part_of_api(keyword):
    with pytest.raises(TypeError):
        prepare_get(PublicEndpoint.SPOT_TIME, **{keyword: "unsupported"})


@pytest.mark.parametrize("limit", [1, 100, 5000])
def test_spot_depth_integer_limits(limit):
    assert dict(prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "BTCUSDT", "limit": limit}).params)["limit"] == limit


@pytest.mark.parametrize("limit", [0, -1, 5001])
def test_spot_depth_limit_out_of_range_rejected(limit):
    with pytest.raises(ValueError, match="limit"):
        prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "BTCUSDT", "limit": limit})


@pytest.mark.parametrize("limit", [5, 10, 20, 50, 100, 500, 1000])
def test_perp_depth_supported_limits(limit):
    assert dict(prepare_get(PublicEndpoint.PERP_DEPTH, {"symbol": "ETHUSDT", "limit": limit}).params)["limit"] == limit


@pytest.mark.parametrize("limit", [1, 101, 5000])
def test_perp_depth_unsupported_limits_rejected(limit):
    with pytest.raises(ValueError, match="limit"):
        prepare_get(PublicEndpoint.PERP_DEPTH, {"symbol": "BTCUSDT", "limit": limit})


@pytest.mark.parametrize("limit", [True, False, 100.0, "100", Decimal("100"), None])
def test_limit_requires_actual_integer(limit):
    with pytest.raises(TypeError, match="integer"):
        prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "BTCUSDT", "limit": limit})


def test_funding_history_parameters_are_validated():
    request = prepare_get(
        PublicEndpoint.PERP_FUNDING_RATE,
        {"symbol": "BTCUSDT", "limit": 1000, "startTime": 0, "endTime": 1000},
    )
    assert dict(request.params)["endTime"] == 1000


@pytest.mark.parametrize(
    "extra",
    [{"limit": 1001}, {"startTime": -1}, {"endTime": 2**63}, {"startTime": 2, "endTime": 1}],
)
def test_invalid_funding_history_ranges_rejected(extra):
    with pytest.raises(ValueError):
        prepare_get(PublicEndpoint.PERP_FUNDING_RATE, {"symbol": "BTCUSDT", **extra})


@pytest.mark.parametrize("value", [True, "1000", 1000.0])
def test_history_timestamps_are_integer_milliseconds(value):
    with pytest.raises(TypeError):
        prepare_get(PublicEndpoint.PERP_FUNDING_RATE, {"symbol": "BTCUSDT", "startTime": value})


def test_parameter_policies_are_specific_to_each_endpoint():
    with pytest.raises(ValueError, match="not allowed"):
        prepare_get(PublicEndpoint.PERP_PREMIUM_INDEX, {"symbol": "BTCUSDT", "limit": 100})
    with pytest.raises(ValueError, match="not allowed"):
        prepare_get(PublicEndpoint.PERP_EXCHANGE_INFO, {"symbol": "BTCUSDT"})
    assert prepare_get(PublicEndpoint.SPOT_EXCHANGE_INFO, {"symbol": "ETHUSDT"}).method == "GET"


def test_params_copied_so_caller_mutation_cannot_change_request():
    supplied = {"symbol": "BTCUSDT", "limit": 100}
    request = prepare_get(PublicEndpoint.SPOT_DEPTH, supplied)
    supplied["symbol"] = "BNBUSDT"
    supplied["limit"] = 5000
    assert dict(request.params) == {"symbol": "BTCUSDT", "limit": 100}


def test_request_and_policy_are_immutable():
    request = prepare_get(PublicEndpoint.SPOT_TIME)
    with pytest.raises(FrozenInstanceError):
        request.params = (("symbol", "BNBUSDT"),)
    with pytest.raises(TypeError):
        ENDPOINT_POLICIES[PublicEndpoint.SPOT_TIME] = ENDPOINT_POLICIES[PublicEndpoint.SPOT_DEPTH]
    with pytest.raises(FrozenInstanceError):
        ENDPOINT_POLICIES[PublicEndpoint.SPOT_TIME].allowed_parameters = frozenset({"url"})


def test_direct_request_constructor_cannot_bypass_validation():
    with pytest.raises(TypeError, match="PublicEndpoint"):
        PublicGetRequest("https://api.binance.com/api/v3/account")
    with pytest.raises(ValueError, match="symbol"):
        PublicGetRequest(PublicEndpoint.SPOT_DEPTH, (("symbol", "BNBUSDT"),))
    with pytest.raises(ValueError, match="duplicate"):
        PublicGetRequest(PublicEndpoint.SPOT_DEPTH, (("symbol", "BTCUSDT"), ("symbol", "ETHUSDT")))
    with pytest.raises(TypeError, match="immutable"):
        PublicGetRequest(PublicEndpoint.SPOT_TIME, [])


def test_prepare_get_requires_enum_and_mapping():
    with pytest.raises(TypeError, match="PublicEndpoint"):
        prepare_get(PublicEndpoint.SPOT_TIME.value)
    with pytest.raises(TypeError, match="mapping"):
        prepare_get(PublicEndpoint.SPOT_DEPTH, [("symbol", "BTCUSDT")])


def test_no_request_execution_or_mutating_method_entry_points():
    request = prepare_get(PublicEndpoint.SPOT_TIME)
    for name in ("post", "put", "delete", "patch", "send", "execute", "request", "session"):
        assert not hasattr(http_client, name)
        assert not hasattr(request, name)


def test_boundary_modules_import_no_network_or_authentication_libraries():
    banned = {"httpx", "requests", "aiohttp", "urllib", "http", "socket", "ssl", "hmac", "hashlib"}
    for module in (endpoints, http_client):
        assert module.__file__ is not None
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert not imports.intersection(banned)
