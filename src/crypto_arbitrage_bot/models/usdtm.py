"""Independent USDT-M public observations, without a combined market snapshot."""

from dataclasses import dataclass, field
from enum import StrEnum

from crypto_arbitrage_bot.exchanges.http_client import PublicGetRequest, RequestPhase
from crypto_arbitrage_bot.models.market import (
    FundingRate,
    Market,
    OrderBook,
    Price,
    Quantity,
    Timestamp,
    require_market_symbol,
)
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class UsdtmErrorKind(StrEnum):
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"


@dataclass(frozen=True)
class UsdtmFetchError:
    kind: UsdtmErrorKind
    message: str
    api_code: int | None = None
    phase: RequestPhase | None = None


@dataclass(frozen=True)
class PerpDepth:
    """Depth E is order_book.event_time; T is transaction_time, both UTC."""

    order_book: OrderBook
    last_update_id: int
    transaction_time: Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.order_book, OrderBook) or self.order_book.market is not Market.USDTM:
            raise ValueError("perpetual depth requires a USDT-M OrderBook")
        if self.order_book.event_time is None:
            raise ValueError("perpetual depth requires the exchange message time")
        if type(self.last_update_id) is not int or self.last_update_id < 0:
            raise ValueError("last_update_id must be a nonnegative integer")
        object.__setattr__(self, "transaction_time", utc_timestamp(self.transaction_time))


@dataclass(frozen=True)
class PremiumIndex:
    """Prices in USDT/base asset and a signed rate, not a funding settlement."""

    symbol: str
    mark_price: Price
    index_price: Price
    last_funding_rate: FundingRate
    next_funding_time: Timestamp
    event_time: Timestamp
    observed_at: Timestamp
    research_instrument: ResearchInstrument | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        require_market_symbol(self.symbol, self.research_instrument)
        require_decimal(self.mark_price, "mark_price", positive=True)
        require_decimal(self.index_price, "index_price", positive=True)
        require_decimal(self.last_funding_rate, "last_funding_rate")
        for name in ("next_funding_time", "event_time", "observed_at"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name)))


@dataclass(frozen=True)
class OpenInterest:
    """Open-interest quantity for the requested contract, not a USDT notional.

    event_time retains the official `time` field verbatim (transaction time).
    observed_at is the local response observation, not a replacement for it.
    """

    symbol: str
    open_interest: Quantity
    event_time: Timestamp
    observed_at: Timestamp
    research_instrument: ResearchInstrument | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        require_market_symbol(self.symbol, self.research_instrument)
        require_decimal(self.open_interest, "open_interest", nonnegative=True)
        for name in ("event_time", "observed_at"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name)))


@dataclass(frozen=True)
class UsdtmFetchResult[T]:
    """One GET attempt with immutable response provenance, including failures.

    received_at records a completely read response body, not exchange time.
    Oversized bodies retain only a prefix with body_truncated=True. Network
    failures may retain partial bytes, but never a successfully parsed data item.
    elapsed_ms is monotonic; wall-clock times are UTC.
    """

    request: PublicGetRequest
    started_at: Timestamp
    finished_at: Timestamp
    received_at: Timestamp | None
    elapsed_ms: int
    status_code: int | None
    response_headers: tuple[tuple[str, str], ...]
    raw_response: bytes | None
    body_truncated: bool
    data: T | None
    error: UsdtmFetchError | None
    cleanup_error: UsdtmFetchError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None
