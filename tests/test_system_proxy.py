"""Offline regression for a Windows system proxy with no proxy environment."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.exchanges import (
    binance_spot,
    binance_usdtm,
    funding_history,
    proxy_transport,
)

FIXTURES = Path(__file__).parent / "fixtures"


def connection(body):
    response = Mock(status=200, read=Mock(return_value=body), getheaders=Mock(return_value=[]))
    result = Mock()
    result.getresponse.return_value = response
    return result


@pytest.mark.parametrize("source", ["spot", "depth", "premium", "oi", "funding"])
def test_every_research_adapter_inherits_system_proxy_without_env(monkeypatch, source):
    discovery = Mock(return_value={"https": "http://system-proxy.example:8123"})
    monkeypatch.setattr(proxy_transport, "getproxies", discovery)
    monkeypatch.setattr(proxy_transport, "proxy_bypass", lambda _: False)
    if source == "spot":
        module, host = binance_spot, "data-api.binance.vision"
        body = (FIXTURES / "spot_depth_btc.json").read_bytes()
    else:
        module, host = (funding_history if source == "funding" else binance_usdtm), "fapi.binance.com"
        fixture = {"depth": "depth", "premium": "premium", "oi": "open_interest"}
        body = b"[]" if source == "funding" else (FIXTURES / f"usdtm_{fixture[source]}_btc.json").read_bytes()
    factory = Mock(return_value=connection(body))
    monkeypatch.setattr(module, "HTTPSConnection", factory)
    if source == "spot":
        result = binance_spot.BinanceSpotClient(3).fetch_order_book("BTCUSDT")
        assert result.ok
    elif source == "funding":
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = funding_history.BinanceFundingHistoryClient(timeout_seconds=3, page_limit=100, max_pages=1).fetch_window(
            "BTCUSDT", now, now + timedelta(hours=1))
        assert result.failure is None
    else:
        method = {"depth": "fetch_order_book", "premium": "fetch_premium_index", "oi": "fetch_open_interest"}[source]
        assert getattr(binance_usdtm.BinanceUsdtmClient(3), method)("BTCUSDT").ok
    discovery.assert_called_once()
    factory.assert_called_once_with("system-proxy.example", port=8123, timeout=3)
    factory.return_value.set_tunnel.assert_called_once_with(host, port=443)


@pytest.mark.parametrize("name", ["HTTPS_PROXY", "https_proxy"])
@pytest.mark.parametrize("value", ["http://explicit.example:8081", ""])
def test_explicit_environment_overrides_or_disables_system_proxy(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    discovery = Mock(side_effect=AssertionError("system discovery must not override explicit env"))
    monkeypatch.setattr(proxy_transport, "getproxies", discovery)
    monkeypatch.setattr(proxy_transport, "proxy_bypass", lambda _: False)
    factory = Mock()
    proxy_transport.open_public_connection("https://fapi.binance.com/fapi/v1/time", timeout=3,
                                           connection_factory=factory)
    if value:
        factory.assert_called_once_with("explicit.example", port=8081, timeout=3)
    else:
        factory.assert_called_once_with("fapi.binance.com", timeout=3)
    discovery.assert_not_called()


def test_no_system_or_environment_proxy_preserves_direct_connection(monkeypatch):
    monkeypatch.setattr(proxy_transport, "getproxies", lambda: {})
    factory = Mock()
    proxy_transport.open_public_connection("https://fapi.binance.com/fapi/v1/time", timeout=3,
                                           connection_factory=factory)
    factory.assert_called_once_with("fapi.binance.com", timeout=3)
    factory.return_value.set_tunnel.assert_not_called()


def test_no_proxy_bypasses_discovered_system_proxy(monkeypatch):
    monkeypatch.setattr(proxy_transport, "getproxies", lambda: {"https": "http://proxy.example:8123"})
    monkeypatch.setattr(proxy_transport, "proxy_bypass", lambda host: host == "fapi.binance.com")
    factory = Mock()
    proxy_transport.open_public_connection("https://fapi.binance.com/fapi/v1/time", timeout=3,
                                           connection_factory=factory)
    factory.assert_called_once_with("fapi.binance.com", timeout=3)


def test_injected_environment_never_reads_system_settings(monkeypatch):
    monkeypatch.setattr(proxy_transport, "getproxies", Mock(side_effect=AssertionError("no OS settings")))
    monkeypatch.setattr(proxy_transport, "proxy_bypass", Mock(side_effect=AssertionError("no OS settings")))
    factory = Mock()
    proxy_transport.open_public_connection("https://fapi.binance.com/fapi/v1/time", timeout=3,
                                           connection_factory=factory, environ={})
    factory.assert_called_once_with("fapi.binance.com", timeout=3)
