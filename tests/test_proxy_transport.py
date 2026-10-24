"""Offline environment proxy selection for each public HTTPS adapter."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.exchanges import binance_spot, binance_usdtm, funding_history
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.proxy_transport import _proxy_value
from crypto_arbitrage_bot.models.spot import SpotErrorKind
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind

FIXTURES = Path(__file__).parent / "fixtures"


def connection(body: bytes, status: int = 200):
    response = Mock(status=status, read=Mock(return_value=body), getheaders=Mock(return_value=[]))
    result = Mock()
    result.getresponse.return_value = response
    return result


def spot_body():
    return (FIXTURES / "spot_depth_btc.json").read_bytes()


def usdtm_body(name: str):
    return (FIXTURES / f"usdtm_{name}_btc.json").read_bytes()


@pytest.mark.parametrize("spelling", ["HTTPS_PROXY", "https_proxy"])
def test_spot_uses_environment_proxy_and_https_tunnel(monkeypatch, spelling):
    monkeypatch.setenv(spelling, "http://proxy.example:7890")
    transport = Mock(return_value=connection(spot_body()))
    monkeypatch.setattr(binance_spot, "HTTPSConnection", transport)

    result = BinanceSpotClient(timeout_seconds=7).fetch_order_book("BTCUSDT")

    assert result.ok and result.error is None
    transport.assert_called_once_with("proxy.example", port=7890, timeout=7)
    transport.return_value.set_tunnel.assert_called_once_with("data-api.binance.vision", port=443)
    args, kwargs = transport.return_value.request.call_args
    assert args[0] == "GET" and args[1].startswith("/api/v3/depth?")
    assert kwargs["headers"] == {"Accept": "application/json", "User-Agent": "crypto-arbitrage-bot/0.1"}


@pytest.mark.parametrize("method,fixture", [
    ("fetch_order_book", "depth"),
    ("fetch_premium_index", "premium"),
    ("fetch_open_interest", "open_interest"),
])
@pytest.mark.parametrize("spelling", ["HTTPS_PROXY", "https_proxy"])
def test_every_usdtm_get_uses_same_environment_proxy(monkeypatch, method, fixture, spelling):
    monkeypatch.setenv(spelling, "http://proxy.example:7890/")
    transport = Mock(return_value=connection(usdtm_body(fixture)))
    monkeypatch.setattr(binance_usdtm, "HTTPSConnection", transport)

    result = getattr(BinanceUsdtmClient(timeout_seconds=9), method)("BTCUSDT")

    assert result.ok and result.error is None
    transport.assert_called_once_with("proxy.example", port=7890, timeout=9)
    transport.return_value.set_tunnel.assert_called_once_with("fapi.binance.com", port=443)
    args = transport.return_value.request.call_args.args
    assert args[0] == "GET" and args[1].startswith("/fapi/v1/")


def test_funding_history_uses_the_same_proxy_transport(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.example:7890")
    raw = json.dumps([{"symbol": "BTCUSDT", "fundingRate": "0.01", "fundingTime": 1767254400000,
                       "markPrice": "90000"}]).encode()
    transport = Mock(return_value=connection(raw))
    monkeypatch.setattr(funding_history, "HTTPSConnection", transport)
    from datetime import UTC, datetime, timedelta
    opened = datetime(2026, 1, 1, tzinfo=UTC)
    window = funding_history.BinanceFundingHistoryClient(timeout_seconds=4, page_limit=100, max_pages=2).fetch_window(
        "BTCUSDT", opened, opened + timedelta(days=1))
    assert window.failure is None and len(window.events) == 1
    transport.assert_called_once_with("proxy.example", port=7890, timeout=4)
    transport.return_value.set_tunnel.assert_called_once_with("fapi.binance.com", port=443)


@pytest.mark.parametrize("adapter", ["spot", "perp"])
def test_no_proxy_environment_preserves_direct_https(monkeypatch, adapter):
    if adapter == "spot":
        transport = Mock(return_value=connection(spot_body()))
        monkeypatch.setattr(binance_spot, "HTTPSConnection", transport)
        result = BinanceSpotClient().fetch_order_book("BTCUSDT")
        host = "data-api.binance.vision"
    else:
        transport = Mock(return_value=connection(usdtm_body("premium")))
        monkeypatch.setattr(binance_usdtm, "HTTPSConnection", transport)
        result = BinanceUsdtmClient().fetch_premium_index("BTCUSDT")
        host = "fapi.binance.com"
    assert result.ok
    transport.assert_called_once_with(host, timeout=5)
    transport.return_value.set_tunnel.assert_not_called()


@pytest.mark.parametrize("adapter", ["spot", "perp"])
def test_malformed_proxy_returns_structured_network_error(monkeypatch, adapter):
    monkeypatch.setenv("HTTPS_PROXY", "this-is-not-a-proxy-url")
    transport = Mock()
    if adapter == "spot":
        monkeypatch.setattr(binance_spot, "HTTPSConnection", transport)
        result = BinanceSpotClient().fetch_order_book("ETHUSDT")
        assert result.error.kind is SpotErrorKind.NETWORK_ERROR
    else:
        monkeypatch.setattr(binance_usdtm, "HTTPSConnection", transport)
        result = BinanceUsdtmClient().fetch_open_interest("ETHUSDT")
        assert result.error.kind is UsdtmErrorKind.NETWORK_ERROR
    assert not result.ok and result.data is None if adapter == "perp" else not result.ok and result.order_book is None
    assert "proxy" in result.error.message.lower()
    assert result.error.phase.value == "REQUEST"
    assert transport.call_count == 0


@pytest.mark.parametrize("scheme,upper,lower", [
    ("https", "HTTPS_PROXY", "https_proxy"),
    ("http", "HTTP_PROXY", "http_proxy"),
])
def test_standard_scheme_mapping_lowercase_precedence_and_disable(scheme, upper, lower, monkeypatch):
    monkeypatch.setenv(upper, "http://uppercase.example:8080")
    monkeypatch.setenv(lower, "http://lowercase.example:8081")
    assert _proxy_value(scheme, __import__("os").environ) == "http://lowercase.example:8081"
    monkeypatch.setenv(lower, "")
    assert _proxy_value(scheme, __import__("os").environ) is None
