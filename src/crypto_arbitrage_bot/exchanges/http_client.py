"""Offline request preparation for a future strictly public, GET-only transport.

There is intentionally no HTTP client, session, socket, redirect handling,
authentication, or request execution in this module. Only immutable request
descriptors are constructed. A later transport must consume this boundary.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from crypto_arbitrage_bot.exchanges.endpoints import (
    ENDPOINT_POLICIES,
    SUPPORTED_SYMBOLS,
    PublicEndpoint,
)
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument

_RESEARCH_ROUTES = frozenset((PublicEndpoint.SPOT_DEPTH, PublicEndpoint.PERP_DEPTH,
                             PublicEndpoint.PERP_PREMIUM_INDEX, PublicEndpoint.PERP_OPEN_INTEREST,
                             PublicEndpoint.PERP_FUNDING_RATE))

ParameterValue: TypeAlias = str | int
RequestParameters: TypeAlias = tuple[tuple[str, ParameterValue], ...]


class RequestPhase(StrEnum):
    """Diagnostic stages only; this module still performs no network IO."""

    REQUEST = "REQUEST"  # Includes DNS, connection/TLS and sending the GET.
    RESPONSE_HEADERS = "RESPONSE_HEADERS"
    RESPONSE_BODY = "RESPONSE_BODY"
    PARSE = "PARSE"
    CLOSE = "CLOSE"


def _validate_parameters(
    endpoint: PublicEndpoint, pairs: tuple[tuple[str, object], ...],
    *, research_instrument: ResearchInstrument | None = None,
) -> RequestParameters:
    if not isinstance(endpoint, PublicEndpoint):
        raise TypeError("endpoint must be a PublicEndpoint enum member")
    if research_instrument is not None and (not isinstance(research_instrument, ResearchInstrument)
                                           or endpoint not in _RESEARCH_ROUTES):
        raise ValueError("research instrument is valid only for the bounded public research routes")
    policy = ENDPOINT_POLICIES[endpoint]
    values: dict[str, ParameterValue] = {}
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError("each parameter must be a (name, value) tuple")
        name, value = pair
        if type(name) is not str:
            raise TypeError("parameter names must be strings")
        if name in values:
            raise ValueError(f"duplicate parameter: {name}")
        if name not in policy.allowed_parameters:
            raise ValueError(f"parameter is not allowed for this endpoint: {name}")
        if name == "symbol":
            if research_instrument is not None:
                if type(value) is not str:
                    raise ValueError("research symbol must be an exact string")
                research_instrument.require_symbol(value)
            elif type(value) is not str or value not in SUPPORTED_SYMBOLS:
                raise ValueError("symbol must be BTCUSDT or ETHUSDT")
            values[name] = value
        else:
            # bool is an int subclass; require a genuine integer deliberately.
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if name == "limit":
                if policy.limit_range is not None:
                    lower, upper = policy.limit_range
                    if not lower <= value <= upper:
                        raise ValueError(f"limit must be between {lower} and {upper}")
                elif policy.limit_choices is not None:
                    if value not in policy.limit_choices:
                        raise ValueError("limit is not a supported order-book depth")
                else:
                    raise ValueError("limit is not allowed for this endpoint")
            elif not 0 <= value <= 2**63 - 1:
                raise ValueError(f"{name} must be a nonnegative 64-bit Unix millisecond timestamp")
            values[name] = value
    missing = policy.required_parameters.difference(values)
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(sorted(missing))}")
    start = values.get("startTime")
    end = values.get("endTime")
    if isinstance(start, int) and isinstance(end, int) and start > end:
        raise ValueError("startTime must not exceed endTime")
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class PublicGetRequest:
    """A validated, immutable request description with no send capability."""

    endpoint: PublicEndpoint
    params: RequestParameters = ()
    research_instrument: ResearchInstrument | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.params) is not tuple:
            raise TypeError("params must be an immutable tuple of parameter pairs")
        object.__setattr__(self, "params", _validate_parameters(self.endpoint, self.params,
                           research_instrument=self.research_instrument))

    @property
    def method(self) -> str:
        return "GET"

    @property
    def url(self) -> str:
        return self.endpoint.value


@dataclass(frozen=True, slots=True)
class PremiumCatalogGetRequest(PublicGetRequest):
    """Explicit public all-market premium read; no single-symbol authorization."""

    def __post_init__(self) -> None:
        if (self.endpoint is not PublicEndpoint.PERP_PREMIUM_INDEX
                or type(self.params) is not tuple or self.params
                or self.research_instrument is not None):
            raise ValueError("premium catalog permits only the exact public premium GET without parameters")


def prepare_premium_catalog_get() -> PremiumCatalogGetRequest:
    return PremiumCatalogGetRequest(PublicEndpoint.PERP_PREMIUM_INDEX)


def prepare_get(
    endpoint: PublicEndpoint, params: Mapping[str, object] | None = None
) -> PublicGetRequest:
    """Validate an enum-selected public endpoint and return an offline descriptor.

    Values are copied and reduced to immutable primitive values. Unknown
    keyword arguments, headers, redirect settings, and bodies are unsupported.
    """
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("params must be a mapping or None")
    pairs = () if params is None else tuple(params.items())
    return PublicGetRequest(endpoint, _validate_parameters(endpoint, pairs))


def prepare_research_get(
    endpoint: PublicEndpoint, instrument: ResearchInstrument, params: Mapping[str, object],
) -> PublicGetRequest:
    """Explicit research identity; legacy prepare_get retains BTC/ETH-only behavior."""
    if not isinstance(params, Mapping) or not isinstance(instrument, ResearchInstrument):
        raise TypeError("research request requires typed instrument and mapping")
    pairs = tuple(params.items())
    validated = _validate_parameters(endpoint, pairs, research_instrument=instrument)
    return PublicGetRequest(endpoint, validated, research_instrument=instrument)


def validate_public_request(
    method: str, url: str, params: Mapping[str, object] | None = None
) -> PublicGetRequest:
    """Validate a raw request description without executing it.

    The URL must exactly equal a registered HTTPS URL. No URL normalization is
    performed: inline queries, fragments, credentials, explicit ports, encoded
    path components, alternate hosts, and private GET routes are all rejected.
    """
    if type(method) is not str or method != "GET":
        raise ValueError("only the GET method is permitted")
    if type(url) is not str:
        raise TypeError("url must be a string")
    try:
        endpoint = PublicEndpoint(url)
    except ValueError as error:
        raise ValueError("URL is not an exact whitelisted public endpoint") from error
    return prepare_get(endpoint, params)
