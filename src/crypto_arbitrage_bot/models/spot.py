"""Spot-only collection results with raw response provenance and typed failures."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from crypto_arbitrage_bot.exchanges.http_client import PublicGetRequest, RequestPhase
from crypto_arbitrage_bot.models.market import OrderBook


class SpotErrorKind(StrEnum):
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"


@dataclass(frozen=True)
class SpotFetchError:
    kind: SpotErrorKind
    message: str
    api_code: int | None = None
    phase: RequestPhase | None = None


@dataclass(frozen=True)
class SpotDepthResult:
    """One GET attempt, including failed attempts; no data is silently reused.

    raw_response is the unchanged response body (bytes). Only an oversized
    response is truncated, explicitly indicated by body_truncated. received_at
    means the complete body was read, not an exchange event timestamp.
    elapsed_ms uses a monotonic clock; wall-clock timestamps are aware UTC.
    """

    request: PublicGetRequest
    started_at: datetime
    finished_at: datetime
    received_at: datetime | None
    elapsed_ms: int
    status_code: int | None
    response_headers: tuple[tuple[str, str], ...]
    raw_response: bytes | None
    body_truncated: bool
    order_book: OrderBook | None
    last_update_id: int | None
    error: SpotFetchError | None
    cleanup_error: SpotFetchError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.order_book is not None and self.last_update_id is not None
