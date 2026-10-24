"""Bounded official public funding history reads for observed exit validation.

GET /fapi/v1/fundingRate supplies settlement time, rate and settlement mark.
Indicative premium-index rates and scenario assumptions are never substituted.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from http.client import HTTPException, HTTPSConnection
from urllib.parse import urlencode, urlsplit

from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.exchanges.binance_usdtm import (
    MAX_BODY_BYTES,
    _decimal,
    _milliseconds,
    _reject_nonfinite_json,
    _unique_json_object,
)
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import (
    PublicGetRequest,
    prepare_get,
    prepare_research_get,
)
from crypto_arbitrage_bot.exchanges.proxy_transport import open_public_connection
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class FundingFailure(StrEnum):
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    HTTP_ERROR = "HTTP_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    PAGE_LIMIT = "PAGE_LIMIT"


@dataclass(frozen=True)
class FundingPage:
    requested_at: datetime
    received_at: datetime
    status_code: int
    raw_response: bytes


@dataclass(frozen=True)
class FundingWindow:
    symbol: str
    opened_at: datetime
    closed_at: datetime
    events: tuple[FundingEvent, ...] = ()
    pages: tuple[FundingPage, ...] = ()
    failure: FundingFailure | None = None
    message: str | None = None


def _epoch_ms(at: datetime) -> int:
    delta = utc_timestamp(at) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


class BinanceFundingHistoryClient:
    def __init__(self, *, timeout_seconds: int, page_limit: int, max_pages: int,
                 research_instrument: ResearchInstrument | None = None) -> None:
        for value, maximum in ((timeout_seconds, 60), (page_limit, 1000), (max_pages, 100)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("funding transport limits must be bounded positive integers")
        if research_instrument is not None and not isinstance(research_instrument, ResearchInstrument):
            raise TypeError("research_instrument must be ResearchInstrument")
        self.research_instrument = research_instrument
        self.timeout_seconds = timeout_seconds
        self.page_limit = page_limit
        self.max_pages = max_pages

    def _prepare(self, params: Mapping[str, object]) -> PublicGetRequest:
        if self.research_instrument is None:
            return prepare_get(PublicEndpoint.PERP_FUNDING_RATE, params)
        return prepare_research_get(PublicEndpoint.PERP_FUNDING_RATE, self.research_instrument, params)

    def fetch_window(self, symbol: str, opened_at: datetime, closed_at: datetime) -> FundingWindow:
        self._prepare({"symbol": symbol})
        opened, closed = utc_timestamp(opened_at), utc_timestamp(closed_at)
        if opened >= closed or _epoch_ms(opened) < 0:
            raise ValueError("funding window must follow nonnegative entry time")
        cursor, end = _epoch_ms(opened) + 1, _epoch_ms(closed)
        events: list[FundingEvent] = []
        pages: list[FundingPage] = []

        def result(failure: FundingFailure | None = None, message: str | None = None) -> FundingWindow:
            # An incomplete history never exposes partial events for accrual.
            return FundingWindow(symbol, opened, closed, tuple(events) if failure is None else (),
                                 tuple(pages), failure, message)

        for _ in range(self.max_pages):
            if cursor > end:
                return result()
            request = self._prepare({"symbol": symbol, "startTime": cursor, "endTime": end, "limit": self.page_limit})
            url = urlsplit(request.url)
            connection = None
            try:
                assert url.hostname is not None
                started = utc_now()
                connection = open_public_connection(
                    request.url, timeout=self.timeout_seconds, connection_factory=HTTPSConnection,
                )
                connection.request("GET", f"{url.path}?{urlencode(request.params)}",
                                   headers={"Accept": "application/json", "User-Agent": "crypto-arbitrage-bot/0.1"})
                response = connection.getresponse()
                body = response.read(MAX_BODY_BYTES + 1)
                pages.append(FundingPage(started, utc_now(), response.status, body[:MAX_BODY_BYTES]))
                if response.status == 429:
                    return result(FundingFailure.RATE_LIMIT, "HTTP 429; no automatic retry")
                if response.status != 200:
                    return result(FundingFailure.HTTP_ERROR, f"HTTP {response.status}")
                if len(body) > MAX_BODY_BYTES:
                    raise ValueError("funding response exceeds body limit")
                payload = json.loads(body.decode("utf-8"), parse_float=Decimal,
                                     parse_constant=_reject_nonfinite_json, object_pairs_hook=_unique_json_object)
                if not isinstance(payload, list) or len(payload) > self.page_limit:
                    raise ValueError("funding history must be a bounded array")
                previous = cursor - 1
                for row in payload:
                    if not isinstance(row, dict) or row.get("symbol") != symbol:
                        raise ValueError("funding row symbol mismatch or invalid payload")
                    settled = _milliseconds(row.get("fundingTime"), "fundingTime")
                    millis = _epoch_ms(settled)
                    if not previous < millis <= end:
                        raise ValueError("funding times must be unique, ascending and inside requested window")
                    events.append(FundingEvent(f"{symbol}:{millis}", settled,
                                               _decimal(row.get("markPrice"), "markPrice"),
                                               _decimal(row.get("fundingRate"), "fundingRate"), symbol))
                    previous = millis
                if len(payload) < self.page_limit or previous == end:
                    return result()
                cursor = previous + 1
            except TimeoutError as exc:
                return result(FundingFailure.TIMEOUT, str(exc))
            except (OSError, HTTPException) as exc:
                return result(FundingFailure.NETWORK_ERROR, str(exc))
            except (ValueError, TypeError, ArithmeticError) as exc:
                return result(FundingFailure.INVALID_RESPONSE, str(exc))
            finally:
                if connection is not None:
                    connection.close()
        return result(FundingFailure.PAGE_LIMIT, "history coverage incomplete: configured page limit reached")
