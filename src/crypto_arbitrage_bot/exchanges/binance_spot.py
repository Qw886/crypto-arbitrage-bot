"""Binance Spot public depth collection, with no account or order operations.

    GET https://data-api.binance.vision/api/v3/depth

The existing offline request boundary validates the symbol and depth before
any connection. This adapter adds only that single Spot route. HTTPSConnection
performs certificate verification by default and never follows redirects.
Standard HTTPS proxy environment variables are read by proxy_transport.
"""

import json
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from http.client import HTTPException, HTTPSConnection, IncompleteRead
from time import monotonic_ns
from urllib.parse import urlencode, urlsplit

from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import (
    RequestPhase,
    prepare_get,
    prepare_research_get,
)
from crypto_arbitrage_bot.exchanges.proxy_transport import open_public_connection
from crypto_arbitrage_bot.models.market import Market, OrderBook, OrderBookLevel
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.spot import SpotDepthResult, SpotErrorKind, SpotFetchError
from crypto_arbitrage_bot.utils.clock import utc_now

MAX_BODY_BYTES = 2_000_000
_DECIMAL_STRING = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")


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
        raise ValueError("depth response must be a JSON object")
    return value


def _levels(value: object, side: str) -> tuple[OrderBookLevel, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{side} must be a nonempty array")
    levels: list[OrderBookLevel] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{side} levels must contain exactly price and quantity")
        if any(type(item) is not str or not _DECIMAL_STRING.fullmatch(item) for item in row):
            raise ValueError(f"{side} price and quantity must be decimal strings")
        levels.append(OrderBookLevel(price=Decimal(row[0]), quantity=Decimal(row[1])))
    for left, right in zip(levels, levels[1:], strict=False):
        if side == "bids" and left.price <= right.price:
            raise ValueError("bids must have unique prices in descending order")
        if side == "asks" and left.price >= right.price:
            raise ValueError("asks must have unique prices in ascending order")
    return tuple(levels)


class BinanceSpotClient:
    """One-shot Spot depth reads; new assets require explicit research identity.

    No network activity occurs on construction. Invalid caller arguments raise
    before IO; expected HTTP/network/payload failures return SpotDepthResult
    with error and no OrderBook. The timeout applies to socket operations, not
    an overall deadline. There is no retry loop or background scanner.
    """

    def __init__(self, timeout_seconds: int = 5, *, research_instrument: ResearchInstrument | None = None) -> None:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be an integer from 1 to 60")
        if research_instrument is not None and not isinstance(research_instrument, ResearchInstrument):
            raise TypeError("research_instrument must be ResearchInstrument")
        self.research_instrument = research_instrument
        self._timeout_seconds = timeout_seconds

    def fetch_order_book(self, symbol: str, limit: int = 100) -> SpotDepthResult:
        request = (prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": symbol, "limit": limit})
                   if self.research_instrument is None else prepare_research_get(
                       PublicEndpoint.SPOT_DEPTH, self.research_instrument, {"symbol": symbol, "limit": limit}))
        # Derive the host/path only from the existing exact endpoint whitelist.
        endpoint = urlsplit(request.url)
        started_at = utc_now()
        started_ns = monotonic_ns()
        received_at = None
        status_code = None
        headers: tuple[tuple[str, str], ...] = ()
        body: bytes | None = None
        body_truncated = False
        order_book = None
        update_id = None
        error = None
        cleanup_error = None
        phase = RequestPhase.REQUEST
        connection: HTTPSConnection | None = None
        try:
            connection = open_public_connection(
                request.url, timeout=self._timeout_seconds, connection_factory=HTTPSConnection,
            )
            connection.request(
                "GET",
                f"{endpoint.path}?{urlencode(request.params)}",
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
                error = SpotFetchError(SpotErrorKind.INVALID_RESPONSE, "response body exceeds limit")
            elif any(
                not value.isascii() or not value.isdigit()
                for name, value in headers if name.lower() == "content-length"
            ):
                error = SpotFetchError(SpotErrorKind.INVALID_RESPONSE, "invalid Content-Length header")
            elif any(
                int(value) != len(body)
                for name, value in headers if name.lower() == "content-length"
            ):
                error = SpotFetchError(SpotErrorKind.NETWORK_ERROR, "incomplete HTTP response body")
            else:
                received_at = utc_now()
                if status_code != 200:
                    kind = (
                        SpotErrorKind.RATE_LIMITED if status_code in (418, 429)
                        else SpotErrorKind.HTTP_ERROR
                    )
                    error = SpotFetchError(kind, f"HTTP {status_code}; no redirect or retry attempted")
                else:
                    phase = RequestPhase.PARSE
                    payload = _decode_body(body)
                    if "code" in payload and type(payload["code"]) is int and payload["code"] < 0:
                        error = SpotFetchError(
                            SpotErrorKind.API_ERROR,
                            str(payload.get("msg", "Binance API error")),
                            api_code=payload["code"],
                        )
                    else:
                        if "symbol" in payload and payload["symbol"] != symbol:
                            raise ValueError("response symbol does not match the requested symbol")
                        last_update_id = payload.get("lastUpdateId")
                        if type(last_update_id) is not int or last_update_id < 0:
                            raise ValueError("lastUpdateId must be a nonnegative integer")
                        bids = _levels(payload.get("bids"), "bids")
                        asks = _levels(payload.get("asks"), "asks")
                        if len(bids) > limit or len(asks) > limit:
                            raise ValueError("response contains more levels than requested")
                        if bids[0].price > asks[0].price:
                            raise ValueError("best bid exceeds best ask")
                        order_book = OrderBook(
                            symbol=symbol, market=Market.SPOT, bids=bids, asks=asks,
                            observed_at=received_at, event_time=None, research_instrument=self.research_instrument,
                        )
                        update_id = last_update_id
        except TimeoutError:
            error = SpotFetchError(SpotErrorKind.TIMEOUT, "Spot depth HTTP request timed out")
        except IncompleteRead as failure:
            body = failure.partial[:MAX_BODY_BYTES]
            body_truncated = len(failure.partial) > MAX_BODY_BYTES
            error = SpotFetchError(SpotErrorKind.NETWORK_ERROR, "incomplete HTTP response body")
        except (OSError, HTTPException) as failure:
            error = SpotFetchError(SpotErrorKind.NETWORK_ERROR, f"network failure: {type(failure).__name__}")
        except (ValueError, InvalidOperation, RecursionError):
            error = SpotFetchError(SpotErrorKind.INVALID_RESPONSE, "malformed Spot depth response")
        finally:
            if connection is not None:
                try:
                    connection.close()
                except (OSError, HTTPException) as failure:
                    cleanup_error = SpotFetchError(
                        SpotErrorKind.NETWORK_ERROR,
                        f"connection cleanup failed: {type(failure).__name__}",
                        phase=RequestPhase.CLOSE,
                    )
        if error is not None:
            error = replace(error, phase=phase)
        # Once rate limiting is known, a broken body must not hide that status.
        if status_code in (418, 429) and (
            error is None or error.kind is not SpotErrorKind.RATE_LIMITED
        ):
            detail = "" if error is None else f"; {error.kind.value}: {error.message}"
            error = SpotFetchError(
                SpotErrorKind.RATE_LIMITED, f"HTTP {status_code}{detail}", phase=phase,
            )
        if error is None and cleanup_error is not None:
            error = cleanup_error
        if error is not None:
            order_book = None
            update_id = None
        return SpotDepthResult(
            request=request,
            started_at=started_at,
            finished_at=utc_now(),
            received_at=received_at,
            elapsed_ms=(monotonic_ns() - started_ns) // 1_000_000,
            status_code=status_code,
            response_headers=headers,
            raw_response=body,
            body_truncated=body_truncated,
            order_book=order_book,
            last_update_id=update_id,
            error=error,
            cleanup_error=cleanup_error,
        )
