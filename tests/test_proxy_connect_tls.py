"""Exercise real stdlib CONNECT/HTTP code; replace only TCP and TLS boundaries.

No external or loopback networking. The TLS socket is simulated: these tests
verify TLS context/SNI selection and wrapping order, not cryptography.
"""

import io
import socket
import ssl
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.funding_history import BinanceFundingHistoryClient
from crypto_arbitrage_bot.exchanges.proxy_transport import open_public_connection

FIXTURES = Path(__file__).parent / "fixtures"
CASES = (
    ("spot", "data-api.binance.vision", "spot_depth_btc.json"),
    ("depth", "fapi.binance.com", "usdtm_depth_btc.json"),
    ("premium", "fapi.binance.com", "usdtm_premium_btc.json"),
    ("oi", "fapi.binance.com", "usdtm_open_interest_btc.json"),
    ("funding", "fapi.binance.com", None),
)


class TimedOutStream(io.BytesIO):
    def readline(self, size=-1):
        raise TimeoutError("synthetic protocol read timeout")


class WireSocket:
    def __init__(self, label, response, events, failure):
        self.label = label
        self.response = response
        self.events = events
        self.failure = failure
        self.sent = []
        self.closed = 0

    def setsockopt(self, *args):
        pass

    def sendall(self, data):
        self.events.append((self.label, "send"))
        self.sent.append(bytes(data))
        if self.label == "tls" and self.failure == "tls_write_eof":
            raise ssl.SSLEOFError("synthetic TLS write EOF")

    def makefile(self, mode):
        assert mode == "rb"
        self.events.append((self.label, "read"))
        if (self.label, self.failure) in (("tcp", "connect_timeout"), ("tls", "response_timeout")):
            return TimedOutStream(self.response)
        return io.BytesIO(self.response)

    def close(self):
        self.closed += 1


def install_wire(monkeypatch, body, *, failure=None, connect_status=200):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8123")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    events = []
    raw = WireSocket("tcp", f"HTTP/1.1 {connect_status} Tunnel\r\n\r\n".encode(), events, failure)
    tls = WireSocket("tls", b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                     + b"\r\n\r\n" + body, events, failure)

    def tcp_connect(address, timeout, source_address):
        events.append(("connect", address, timeout))
        if failure == "tcp_timeout":
            raise TimeoutError("synthetic TCP timeout")
        return raw

    def tls_wrap(context, sock, *, server_hostname):
        assert sock is raw and raw.closed == 0
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        assert events[-1] == ("tcp", "read")  # CONNECT response consumed before TLS.
        events.append(("wrap", server_hostname))
        if failure == "tls_eof":
            raise ssl.SSLEOFError("synthetic handshake EOF")
        if failure == "tls_timeout":
            raise TimeoutError("synthetic handshake timeout")
        return tls

    monkeypatch.setattr(socket, "create_connection", tcp_connect)
    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", tls_wrap)
    return events, raw, tls


def fetch(case):
    if case == "spot":
        return BinanceSpotClient(3).fetch_order_book("BTCUSDT")
    perp = BinanceUsdtmClient(3)
    if case == "depth":
        return perp.fetch_order_book("BTCUSDT")
    if case == "premium":
        return perp.fetch_premium_index("BTCUSDT")
    if case == "oi":
        return perp.fetch_open_interest("BTCUSDT")
    opened = datetime(2026, 1, 1, tzinfo=UTC)
    return BinanceFundingHistoryClient(timeout_seconds=3, page_limit=100, max_pages=1).fetch_window(
        "BTCUSDT", opened, opened + timedelta(hours=1))


@pytest.mark.parametrize("case,host,fixture", CASES)
def test_real_stdlib_connect_then_one_tls_wrap_target_sni_and_host(monkeypatch, case, host, fixture):
    body = b"[]" if fixture is None else (FIXTURES / fixture).read_bytes()
    events, raw, tls = install_wire(monkeypatch, body)
    result = fetch(case)
    assert result.failure is None if case == "funding" else result.ok
    assert events[0] == ("connect", ("proxy.example", 8123), 3)
    assert len([event for event in events if event[0] == "connect"]) == 1
    assert [event for event in events if event[0] == "wrap"] == [("wrap", host)]
    connect = b"".join(raw.sent)
    assert connect.startswith(f"CONNECT {host}:443 HTTP/1.1\r\n".encode())
    assert f"Host: {host}:443\r\n".encode() in connect
    request = b"".join(tls.sent)
    assert request.startswith(b"GET /")
    assert f"Host: {host}\r\n".encode() in request
    assert b"proxy.example" not in request
    assert b"Proxy-Authorization" not in request
    assert tls.closed == 1


@pytest.mark.parametrize("host", ["fapi.binance.com", "data-api.binance.vision"])
def test_project_matches_urllib_connect_wire_and_sni_at_same_timeout(monkeypatch, host):
    url = f"https://{host}/fapi/v1/time" if host == "fapi.binance.com" else f"https://{host}/api/v3/time"
    bodies = []
    for mode in ("project", "urllib"):
        events, raw, tls = install_wire(monkeypatch, b'{"serverTime":123}')
        if mode == "project":
            conn = open_public_connection(url, timeout=3)
            try:
                conn.request("GET", url.split(host)[1])
                response = conn.getresponse()
                assert response.read() == b'{"serverTime":123}'
            finally:
                conn.close()
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://proxy.example:8123"}))
            with opener.open(url, timeout=3) as response:
                assert response.read() == b'{"serverTime":123}'
        bodies.append((raw.sent, [item for item in events if item[0] in ("connect", "wrap")]))
        assert f"Host: {host}\r\n".encode() in b"".join(tls.sent)
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("case", ["spot", "depth", "funding"])
@pytest.mark.parametrize("failure,kind", [
    ("tcp_timeout", "TIMEOUT"), ("connect_timeout", "TIMEOUT"),
    ("tls_timeout", "TIMEOUT"), ("tls_eof", "NETWORK_ERROR"),
    ("tls_write_eof", "NETWORK_ERROR"), ("response_timeout", "TIMEOUT"),
])
def test_transport_failures_preserve_structured_semantics_without_retry(monkeypatch, case, failure, kind):
    events, raw, tls = install_wire(monkeypatch, b"[]", failure=failure)
    result = fetch(case)
    if case == "funding":
        assert result.failure.value == kind and not result.events
    else:
        assert not result.ok and result.error.kind.value == kind
        assert (result.order_book if case == "spot" else result.data) is None
        assert result.error.phase.value == ("RESPONSE_HEADERS" if failure == "response_timeout" else "REQUEST")
    assert len([item for item in events if item[0] == "connect"]) == 1
    if failure in ("connect_timeout", "tls_timeout", "tls_eof"):
        assert raw.closed >= 1 and not tls.sent
    if failure in ("tls_write_eof", "response_timeout"):
        assert tls.closed >= 1


def test_rejected_connect_never_wraps_tls_or_sends_binance_get(monkeypatch):
    events, raw, tls = install_wire(monkeypatch, b"[]", connect_status=502)
    result = fetch("depth")
    assert not result.ok and result.error.kind.value == "NETWORK_ERROR"
    assert not any(item[0] == "wrap" for item in events)
    assert raw.closed >= 1 and not tls.sent
