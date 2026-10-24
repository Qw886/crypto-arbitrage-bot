"""One-shot, public Binance USDT-M depth, premium-index and open-interest GETs.

Uses the existing endpoint/parameter whitelist and domain types. No Spot calls,
credentials, redirects, retries, private API, or background work are supported.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from http.client import HTTPException, HTTPSConnection, IncompleteRead
from time import monotonic_ns
from urllib.parse import urlencode, urlsplit

from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.funding_info import decode_funding_info, parse_funding_info
from crypto_arbitrage_bot.exchanges.http_client import (
    PremiumCatalogGetRequest,
    PublicGetRequest,
    RequestPhase,
    prepare_get,
    prepare_premium_catalog_get,
    prepare_research_get,
)
from crypto_arbitrage_bot.exchanges.premium_catalog import decode_catalog, parse_catalog
from crypto_arbitrage_bot.exchanges.proxy_transport import open_public_connection
from crypto_arbitrage_bot.models.funding_info import FundingAdjustment
from crypto_arbitrage_bot.models.market import Market, OrderBook, OrderBookLevel
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.usdtm import (
    OpenInterest,
    PerpDepth,
    PremiumIndex,
    UsdtmErrorKind,
    UsdtmFetchError,
    UsdtmFetchResult,
)
from crypto_arbitrage_bot.utils.clock import utc_now

MAX_BODY_BYTES = 2_000_000
_DECIMAL_STRING = re.compile(r"-?[0-9]+(?:\.[0-9]+)?\Z")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MAX_TIMESTAMP_MS = 253402300799999  # Last millisecond representable by datetime.
_ENDPOINTS = frozenset({
    PublicEndpoint.PERP_DEPTH,
    PublicEndpoint.PERP_PREMIUM_INDEX,
    PublicEndpoint.PERP_OPEN_INTEREST,
    PublicEndpoint.PERP_FUNDING_INFO,
})


@dataclass(frozen=True)
class PublicTimeResult:
    request: PublicGetRequest
    started_at: datetime
    finished_at: datetime
    elapsed_ns: int
    server_time: datetime | None
    http_status: int | None
    failure_code: str | None


def fetch_public_time(request: PublicGetRequest, timeout_seconds: int) -> PublicTimeResult:
    """One bounded time GET through the existing public transport boundary."""
    if type(request) is not PublicGetRequest or request.endpoint not in (
        PublicEndpoint.SPOT_TIME, PublicEndpoint.PERP_TIME
    ) or request.params or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 10:
        raise ValueError("BOUNDED_PUBLIC_TIME_GET_REQUIRED")
    started, tick = utc_now(), monotonic_ns()
    connection = None
    server = None
    status = None
    failure = None
    try:
        connection = open_public_connection(request.url, timeout=timeout_seconds)
        connection.request("GET", urlsplit(request.url).path,
                           headers={"Accept": "application/json", "Cache-Control": "no-cache"})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            failure = "PUBLIC_TIME_HTTP_ERROR"
        else:
            length = response.getheader("Content-Length")
            if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > 4096):
                raise ValueError("BOUNDED_CLOCK_BODY_REQUIRED")
            body = response.read(4097)
            if len(body) > 4096:
                raise ValueError("BOUNDED_CLOCK_BODY_REQUIRED")
            server = _milliseconds(_decode_body(body).get("serverTime"), "serverTime")
    except TimeoutError:
        failure = "PUBLIC_TIME_TIMEOUT"
    except (OSError, HTTPException):
        failure = "PUBLIC_TIME_IO_ERROR"
    except (ValueError, TypeError, OverflowError):
        failure = "PUBLIC_TIME_RESPONSE_INVALID"
    finally:
        if connection is not None:
            try:
                connection.close()
            except (OSError, HTTPException):
                server, failure = None, "PUBLIC_TIME_IO_ERROR"
    return PublicTimeResult(request, started, utc_now(), monotonic_ns()-tick, server, status, failure)


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _decode_body(body: bytes) -> dict[str, object]:
    value = json.loads(
        body.decode("utf-8"),
        parse_float=Decimal,
        parse_constant=_reject_nonfinite_json,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str) or not _DECIMAL_STRING.fullmatch(value):
        raise ValueError(f"{name} must be a decimal string")
    return Decimal(value)


def _milliseconds(value: object, name: str) -> datetime:
    if type(value) is not int or not 0 <= value <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"{name} must be representable nonnegative Unix milliseconds")
    # Integer arithmetic preserves all milliseconds without a float conversion.
    return _EPOCH + timedelta(milliseconds=value)


def _levels(value: object, side: str) -> tuple[OrderBookLevel, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{side} must be a nonempty array")
    levels: list[OrderBookLevel] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{side} levels must contain exactly price and quantity")
        levels.append(OrderBookLevel(_decimal(row[0], "price"), _decimal(row[1], "quantity")))
    for left, right in zip(levels, levels[1:], strict=False):
        if side == "bids" and left.price <= right.price:
            raise ValueError("bids must have unique prices in descending order")
        if side == "asks" and left.price >= right.price:
            raise ValueError("asks must have unique prices in ascending order")
    return tuple(levels)


def _parse_depth(
    payload: dict[str, object], symbol: str, limit: int, observed_at: datetime,
    *, research_instrument: ResearchInstrument | None = None,
) -> PerpDepth:
    if "symbol" in payload and payload["symbol"] != symbol:
        raise ValueError("response symbol does not match the requested symbol")
    update_id = payload.get("lastUpdateId")
    if type(update_id) is not int or update_id < 0:
        raise ValueError("lastUpdateId must be a nonnegative integer")
    event_time = _milliseconds(payload.get("E"), "E")
    transaction_time = _milliseconds(payload.get("T"), "T")
    bids = _levels(payload.get("bids"), "bids")
    asks = _levels(payload.get("asks"), "asks")
    if len(bids) > limit or len(asks) > limit:
        raise ValueError("response contains more levels than requested")
    if bids[0].price > asks[0].price:
        raise ValueError("best bid exceeds best ask")
    return PerpDepth(
        order_book=OrderBook(
            symbol=symbol, market=Market.USDTM, bids=bids, asks=asks,
            observed_at=observed_at, event_time=event_time, research_instrument=research_instrument,
        ),
        last_update_id=update_id,
        transaction_time=transaction_time,
    )


def _parse_premium(
    payload: dict[str, object], symbol: str, observed_at: datetime,
    *, research_instrument: ResearchInstrument | None = None,
) -> PremiumIndex:
    if payload.get("symbol") != symbol:
        raise ValueError("response symbol does not match the requested symbol")
    return PremiumIndex(
        symbol=symbol,
        mark_price=_decimal(payload.get("markPrice"), "markPrice"),
        index_price=_decimal(payload.get("indexPrice"), "indexPrice"),
        last_funding_rate=_decimal(payload.get("lastFundingRate"), "lastFundingRate"),
        next_funding_time=_milliseconds(payload.get("nextFundingTime"), "nextFundingTime"),
        event_time=_milliseconds(payload.get("time"), "time"),
        observed_at=observed_at, research_instrument=research_instrument,
    )


def _parse_open_interest(
    payload: dict[str, object], symbol: str, observed_at: datetime,
    *, research_instrument: ResearchInstrument | None = None,
) -> OpenInterest:
    if payload.get("symbol") != symbol:
        raise ValueError("response symbol does not match the requested symbol")
    return OpenInterest(
        symbol=symbol,
        open_interest=_decimal(payload.get("openInterest"), "openInterest"),
        event_time=_milliseconds(payload.get("time"), "time"),
        observed_at=observed_at, research_instrument=research_instrument,
    )


class BinanceUsdtmClient:
    """Three public reads; new assets require explicit research identity.

    Construction is offline. Invalid caller arguments raise before IO; expected
    HTTP/network/payload failures return a typed result with data=None. The
    timeout applies to socket operations, not a strict total-request deadline.
    """

    def __init__(self, timeout_seconds: int = 5, *, research_instrument: ResearchInstrument | None = None) -> None:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be an integer from 1 to 60")
        if research_instrument is not None and not isinstance(research_instrument, ResearchInstrument):
            raise TypeError("research_instrument must be ResearchInstrument")
        self.research_instrument = research_instrument
        self._timeout_seconds = timeout_seconds

    def _prepare(self, endpoint: PublicEndpoint, params: Mapping[str, object]) -> PublicGetRequest:
        if self.research_instrument is None:
            return prepare_get(endpoint, params)
        return prepare_research_get(endpoint, self.research_instrument, params)

    def fetch_order_book(self, symbol: str, limit: int = 100) -> UsdtmFetchResult[PerpDepth]:
        request = self._prepare(PublicEndpoint.PERP_DEPTH, {"symbol": symbol, "limit": limit})
        return self._fetch(request, lambda payload, at: _parse_depth(payload, symbol, limit, at, research_instrument=self.research_instrument))

    def fetch_premium_index(self, symbol: str) -> UsdtmFetchResult[PremiumIndex]:
        request = self._prepare(PublicEndpoint.PERP_PREMIUM_INDEX, {"symbol": symbol})
        return self._fetch(request, lambda payload, at: _parse_premium(payload, symbol, at, research_instrument=self.research_instrument))

    def fetch_open_interest(self, symbol: str) -> UsdtmFetchResult[OpenInterest]:
        request = self._prepare(PublicEndpoint.PERP_OPEN_INTEREST, {"symbol": symbol})
        return self._fetch(request, lambda payload, at: _parse_open_interest(payload, symbol, at, research_instrument=self.research_instrument))

    def fetch_premium_catalog(self) -> UsdtmFetchResult[tuple[dict[str, object], ...]]:
        """One official all-market GET; not depth, eligibility or a trading signal."""
        if self.research_instrument is not None:
            raise ValueError("bulk catalog has no single research instrument")
        return self._fetch(prepare_premium_catalog_get(), parse_catalog, decoder=decode_catalog)

    def fetch_funding_info(self) -> UsdtmFetchResult[tuple[FundingAdjustment, ...]]:
        """One public adjustment catalog GET; absent symbols stay unknown."""
        if self.research_instrument is not None:
            raise ValueError("funding adjustments are a bulk catalog, not a single-symbol request")
        return self._fetch(prepare_get(PublicEndpoint.PERP_FUNDING_INFO), parse_funding_info,
                           decoder=decode_funding_info)

    def _fetch[T](
        self, request: PublicGetRequest, parser: Callable[[dict[str, object], datetime], T],
        *, decoder: Callable[[bytes], dict[str, object]] = _decode_body,
    ) -> UsdtmFetchResult[T]:
        # Revalidate even this private helper; execute only supported public routes.
        if isinstance(request, PremiumCatalogGetRequest):
            request = PremiumCatalogGetRequest(request.endpoint, request.params, research_instrument=request.research_instrument)
        else:
            request = PublicGetRequest(request.endpoint, request.params, research_instrument=request.research_instrument)
        if request.research_instrument != self.research_instrument:
            raise ValueError("request research identity differs from adapter")
        if request.endpoint not in _ENDPOINTS:
            raise ValueError("endpoint is not supported by the USDT-M adapter")
        endpoint = urlsplit(request.url)
        assert endpoint.hostname is not None
        started_at = utc_now()
        started_ns = monotonic_ns()
        received_at = None
        status_code = None
        headers: tuple[tuple[str, str], ...] = ()
        body: bytes | None = None
        body_truncated = False
        data: T | None = None
        error = None
        cleanup_error = None
        phase = RequestPhase.REQUEST
        connection: HTTPSConnection | None = None
        try:
            connection = open_public_connection(
                request.url, timeout=self._timeout_seconds, connection_factory=HTTPSConnection,
            )
            connection.request(
                "GET", f"{endpoint.path}?{urlencode(request.params)}",
                headers={"Accept": "application/json", "User-Agent": "crypto-arbitrage-bot/0.1"},
            )
            phase = RequestPhase.RESPONSE_HEADERS
            response = connection.getresponse()
            status_code = response.status
            headers = tuple(response.getheaders())
            phase = RequestPhase.RESPONSE_BODY
            body = response.read(MAX_BODY_BYTES + 1)
            if len(body) > MAX_BODY_BYTES:
                body = body[:MAX_BODY_BYTES]
                body_truncated = True
                error = UsdtmFetchError(UsdtmErrorKind.INVALID_RESPONSE, "response body exceeds limit")
            elif any(
                not value.isascii() or not value.isdigit()
                for name, value in headers if name.lower() == "content-length"
            ):
                error = UsdtmFetchError(UsdtmErrorKind.INVALID_RESPONSE, "invalid Content-Length header")
            elif any(
                int(value) != len(body)
                for name, value in headers if name.lower() == "content-length"
            ):
                error = UsdtmFetchError(UsdtmErrorKind.NETWORK_ERROR, "incomplete HTTP response body")
            else:
                received_at = utc_now()
                if status_code != 200:
                    kind = (
                        UsdtmErrorKind.RATE_LIMITED if status_code in (418, 429)
                        else UsdtmErrorKind.HTTP_ERROR
                    )
                    error = UsdtmFetchError(kind, f"HTTP {status_code}; no redirect or retry attempted")
                else:
                    phase = RequestPhase.PARSE
                    payload = decoder(body)
                    code = payload.get("code")
                    if type(code) is int and code < 0:
                        error = UsdtmFetchError(
                            UsdtmErrorKind.API_ERROR,
                            str(payload.get("msg", "Binance API error")), api_code=code,
                        )
                    else:
                        data = parser(payload, received_at)
        except TimeoutError:
            error = UsdtmFetchError(UsdtmErrorKind.TIMEOUT, "USDT-M HTTP request timed out")
        except IncompleteRead as failure:
            body = failure.partial[:MAX_BODY_BYTES]
            body_truncated = len(failure.partial) > MAX_BODY_BYTES
            error = UsdtmFetchError(UsdtmErrorKind.NETWORK_ERROR, "incomplete HTTP response body")
        except (OSError, HTTPException) as failure:
            error = UsdtmFetchError(
                UsdtmErrorKind.NETWORK_ERROR, f"network failure: {type(failure).__name__}",
            )
        except (ValueError, InvalidOperation, RecursionError):
            error = UsdtmFetchError(UsdtmErrorKind.INVALID_RESPONSE, "malformed USDT-M response")
        finally:
            if connection is not None:
                try:
                    connection.close()
                except (OSError, HTTPException) as failure:
                    cleanup_error = UsdtmFetchError(
                        UsdtmErrorKind.NETWORK_ERROR,
                        f"connection cleanup failed: {type(failure).__name__}",
                        phase=RequestPhase.CLOSE,
                    )
        if error is not None:
            error = replace(error, phase=phase)
        # Once rate limiting is known, a broken body must not hide that status.
        if status_code in (418, 429) and (
            error is None or error.kind is not UsdtmErrorKind.RATE_LIMITED
        ):
            detail = "" if error is None else f"; {error.kind.value}: {error.message}"
            error = UsdtmFetchError(
                UsdtmErrorKind.RATE_LIMITED, f"HTTP {status_code}{detail}", phase=phase,
            )
        if error is None and cleanup_error is not None:
            error = cleanup_error
        if error is not None:
            data = None
        return UsdtmFetchResult(
            request=request, started_at=started_at, finished_at=utc_now(),
            received_at=received_at, elapsed_ms=(monotonic_ns() - started_ns) // 1_000_000,
            status_code=status_code, response_headers=headers, raw_response=body,
            body_truncated=body_truncated, data=data, error=error, cleanup_error=cleanup_error,
        )
