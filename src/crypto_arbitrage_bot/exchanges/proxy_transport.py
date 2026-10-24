"""Environment-aware HTTPS connection creation for the fixed public GET clients.

``http.client.HTTPSConnection`` deliberately has no environment-proxy support.
This small boundary reads explicit proxy variables, then the standard OS proxy
resolver (including Windows Internet Settings) when variables are absent, and
establishes an HTTPS CONNECT tunnel. It never adds proxy credentials or changes
the whitelisted destination/request target.
"""

import os
from collections.abc import Callable, Mapping
from http.client import HTTPSConnection
from urllib import request as urllib_request
from urllib.parse import SplitResult, urlsplit
from urllib.request import getproxies, proxy_bypass


class ProxyConfigurationError(OSError):
    """Invalid proxy configuration, safely classed as a network failure."""


def _proxy_value(scheme: str, environ: Mapping[str, str]) -> str | None:
    upper, lower = f"{scheme.upper()}_PROXY", f"{scheme.lower()}_proxy"
    # Matches urllib's environment convention: lowercase takes precedence when
    # both spellings exist; an empty lowercase value explicitly disables it.
    if lower in environ:
        value = environ[lower]
    elif upper in environ:
        value = environ[upper]
    else:
        value = ""
    return value.strip() or None


def _parse_proxy(value: str) -> SplitResult:
    try:
        parsed = urlsplit(value)
        # Require an explicit supported proxy transport and forbid embedded
        # credentials/path/query so no secret or unexpected route is sent.
        if (parsed.scheme not in {"http", "https"} or parsed.hostname is None
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        return parsed
    except ValueError as exc:
        # Do not echo environment content; it may contain proxy credentials.
        raise ProxyConfigurationError("invalid HTTP(S) proxy URL in environment") from exc


def open_public_connection(
    destination: str,
    *,
    timeout: int,
    connection_factory: Callable[..., HTTPSConnection] = HTTPSConnection,
    environ: Mapping[str, str] | None = None,
) -> HTTPSConnection:
    """Create a direct or proxy-tunneled connection to an approved HTTPS URL.

    Callers must pass the URL from ``PublicGetRequest``. HTTP proxy variables
    are scheme-specific; for the Binance V1 adapters all destinations are HTTPS,
    so ``HTTPS_PROXY``/``https_proxy`` apply. Otherwise urllib discovers the OS
    proxy. No configured proxy means direct HTTPS. Explicit empty variables
    disable the proxy; an injected environ never reads workstation settings.
    """
    target = urlsplit(destination)
    if target.scheme != "https" or target.hostname is None or target.port not in (None, 443):
        raise ValueError("only approved HTTPS destination URLs are supported")
    variables = os.environ if environ is None else environ
    value = _proxy_value(target.scheme, variables)
    explicit = any(name in variables for name in ("HTTPS_PROXY", "https_proxy"))
    if environ is None and not explicit:
        value = getproxies().get(target.scheme)
    if environ is None:
        bypass = proxy_bypass(target.hostname)
    else:
        # Runtime stdlib helper is absent from the bundled Windows type stubs.
        bypass = urllib_request.proxy_bypass_environment(  # type: ignore[attr-defined]
            target.hostname, {"no": _proxy_value("no", variables) or ""})
    if bypass:
        value = None
    if value is None:
        return connection_factory(target.hostname, timeout=timeout)

    proxy = _parse_proxy(value)
    proxy_port = proxy.port or (443 if proxy.scheme == "https" else 80)
    connection = connection_factory(proxy.hostname, port=proxy_port, timeout=timeout)
    try:
        connection.set_tunnel(target.hostname, port=target.port or 443)
    except Exception:
        connection.close()
        raise
    return connection
