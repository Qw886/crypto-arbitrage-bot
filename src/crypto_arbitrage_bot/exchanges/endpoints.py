"""Immutable registry of the public Binance GET routes permitted by V1.

This registry describes a future transport boundary only. It makes no requests.
The deliberately narrow parameter subset supports BTCUSDT and ETHUSDT research.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


class PublicEndpoint(str, Enum):
    SPOT_DEPTH = "https://data-api.binance.vision/api/v3/depth"
    SPOT_BOOK_TICKER = "https://data-api.binance.vision/api/v3/ticker/bookTicker"
    SPOT_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
    SPOT_TIME = "https://api.binance.com/api/v3/time"
    PERP_DEPTH = "https://fapi.binance.com/fapi/v1/depth"
    PERP_BOOK_TICKER = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
    PERP_PREMIUM_INDEX = "https://fapi.binance.com/fapi/v1/premiumIndex"
    PERP_OPEN_INTEREST = "https://fapi.binance.com/fapi/v1/openInterest"
    PERP_FUNDING_RATE = "https://fapi.binance.com/fapi/v1/fundingRate"
    PERP_FUNDING_INFO = "https://fapi.binance.com/fapi/v1/fundingInfo"
    PERP_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    PERP_TIME = "https://fapi.binance.com/fapi/v1/time"


SUPPORTED_SYMBOLS: Final = frozenset({"BTCUSDT", "ETHUSDT"})


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    required_parameters: frozenset[str] = frozenset()
    allowed_parameters: frozenset[str] = frozenset()
    limit_range: tuple[int, int] | None = None
    limit_choices: frozenset[int] | None = None


_SYMBOL: Final = frozenset({"symbol"})
_SYMBOL_LIMIT: Final = frozenset({"symbol", "limit"})
_SYMBOL_ONLY: Final = EndpointPolicy(_SYMBOL, _SYMBOL)

ENDPOINT_POLICIES: Final[Mapping[PublicEndpoint, EndpointPolicy]] = MappingProxyType(
    {
        PublicEndpoint.SPOT_DEPTH: EndpointPolicy(_SYMBOL, _SYMBOL_LIMIT, (1, 5000)),
        PublicEndpoint.SPOT_BOOK_TICKER: _SYMBOL_ONLY,
        PublicEndpoint.SPOT_EXCHANGE_INFO: EndpointPolicy(allowed_parameters=_SYMBOL),
        PublicEndpoint.SPOT_TIME: EndpointPolicy(),
        PublicEndpoint.PERP_DEPTH: EndpointPolicy(
            _SYMBOL, _SYMBOL_LIMIT, limit_choices=frozenset({5, 10, 20, 50, 100, 500, 1000})
        ),
        PublicEndpoint.PERP_BOOK_TICKER: _SYMBOL_ONLY,
        PublicEndpoint.PERP_PREMIUM_INDEX: _SYMBOL_ONLY,
        PublicEndpoint.PERP_OPEN_INTEREST: _SYMBOL_ONLY,
        PublicEndpoint.PERP_FUNDING_RATE: EndpointPolicy(
            _SYMBOL, frozenset({"symbol", "limit", "startTime", "endTime"}), (1, 1000)
        ),
        PublicEndpoint.PERP_FUNDING_INFO: EndpointPolicy(),
        PublicEndpoint.PERP_EXCHANGE_INFO: EndpointPolicy(),
        PublicEndpoint.PERP_TIME: EndpointPolicy(),
    }
)
