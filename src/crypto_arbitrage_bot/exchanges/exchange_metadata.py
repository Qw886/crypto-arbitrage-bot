"""Bounded official public exchangeInfo GETs. No symbol or account authority."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from http.client import HTTPException, HTTPSConnection
from time import monotonic_ns
from urllib.parse import urlencode, urlsplit

from crypto_arbitrage_bot.exchanges.binance_spot import _decode_body
from crypto_arbitrage_bot.exchanges.binance_usdtm import _milliseconds
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import RequestPhase, prepare_get
from crypto_arbitrage_bot.exchanges.proxy_transport import open_public_connection
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

METADATA_ENDPOINTS = frozenset((PublicEndpoint.SPOT_EXCHANGE_INFO, PublicEndpoint.PERP_EXCHANGE_INFO))
# Spot's full exchangeInfo exceeds 16 MB as of the live 2026-10-05 capture.
# This remains a finite transport bound, separate from small depth payloads.
MAX_METADATA_BYTES = 64_000_000


class MetadataErrorKind(StrEnum):
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    HTTP_ERROR = "HTTP_ERROR"
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"


@dataclass(frozen=True)
class MetadataError:
    kind: MetadataErrorKind
    phase: RequestPhase
    api_code: int | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class MetadataSource:
    endpoint: PublicEndpoint
    requested_at: datetime
    received_at: datetime | None
    finished_at: datetime
    official_time: datetime | None
    elapsed_ms: int
    status_code: int | None
    raw_response: bytes | None
    body_truncated: bool
    error: MetadataError | None
    research_instrument: ResearchInstrument | None = None

    def __post_init__(self) -> None:
        if self.research_instrument is not None and (
                not isinstance(self.research_instrument, ResearchInstrument)
                or self.endpoint is not PublicEndpoint.SPOT_EXCHANGE_INFO):
            raise ValueError("scoped metadata requires an explicit Spot research identity")
        if not isinstance(self.endpoint, PublicEndpoint) or self.endpoint not in METADATA_ENDPOINTS:
            raise ValueError("metadata source endpoint must be public exchangeInfo")
        for name in ("requested_at", "received_at", "finished_at", "official_time"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc_timestamp(value))
        if self.finished_at < self.requested_at or self.received_at is not None and not (
                self.requested_at <= self.received_at <= self.finished_at):
            raise ValueError("metadata request/receive times are inconsistent")


class BinanceMetadataClient:
    """No construction IO, redirects, retries, credentials or arbitrary URLs."""

    def __init__(self, *, timeout_seconds: int) -> None:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ValueError("metadata timeout must be 1..60 integer seconds")
        self.timeout_seconds = timeout_seconds

    def fetch(self, endpoint: PublicEndpoint, *,
              research_instrument: ResearchInstrument | None = None) -> MetadataSource:
        if endpoint not in METADATA_ENDPOINTS or not isinstance(endpoint, PublicEndpoint):
            raise ValueError("only public exchangeInfo endpoints are supported")
        if research_instrument is not None and (
                not isinstance(research_instrument, ResearchInstrument)
                or endpoint is not PublicEndpoint.SPOT_EXCHANGE_INFO):
            raise ValueError("scoped metadata requires an explicit Spot research identity")
        request = prepare_get(endpoint)
        parsed = urlsplit(request.url)
        # Fixed official public parameters; no caller-supplied URL/headers/body.
        route = parsed.path if research_instrument is None else parsed.path + "?" + urlencode(
            (("symbol", research_instrument.symbol), ("showPermissionSets", "false")))
        started, started_ns = utc_now(), monotonic_ns()
        received = official = status = body = error = None
        truncated = False
        phase = RequestPhase.REQUEST
        connection = None
        try:
            connection = open_public_connection(request.url, timeout=self.timeout_seconds,
                                                connection_factory=HTTPSConnection)
            connection.request("GET", route, headers={"Accept": "application/json",
                                                           "User-Agent": "crypto-arbitrage-bot/0.1"})
            phase = RequestPhase.RESPONSE_HEADERS
            response = connection.getresponse()
            status = response.status
            phase = RequestPhase.RESPONSE_BODY
            raw = response.read(MAX_METADATA_BYTES + 1)
            received = utc_now()
            truncated, body = len(raw) > MAX_METADATA_BYTES, raw[:MAX_METADATA_BYTES]
            if status == 429:
                error = MetadataError(MetadataErrorKind.RATE_LIMIT, phase)
            elif status != 200:
                error = MetadataError(MetadataErrorKind.HTTP_ERROR, phase)
            elif truncated:
                error = MetadataError(MetadataErrorKind.INVALID_RESPONSE, phase)
            else:
                phase = RequestPhase.PARSE
                payload = _decode_body(body)
                if "code" in payload:
                    code = payload["code"]
                    if type(code) is not int:
                        raise ValueError("invalid API code")
                    error = MetadataError(MetadataErrorKind.API_ERROR, phase, api_code=code)
                elif not isinstance(payload.get("symbols"), list):
                    raise ValueError("missing symbols array")
                else:
                    if research_instrument is not None:
                        rows = payload["symbols"]
                        if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
                                or rows[0].get("symbol") != research_instrument.symbol):
                            raise ValueError("scoped Spot response does not match the requested instrument")
                    if "serverTime" in payload:
                        official = _milliseconds(payload["serverTime"], "serverTime")
        except TimeoutError:
            error = MetadataError(MetadataErrorKind.TIMEOUT, phase)
        except (OSError, HTTPException) as exc:
            error = MetadataError(MetadataErrorKind.NETWORK_ERROR, phase, error_type=type(exc).__name__)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            error = MetadataError(MetadataErrorKind.INVALID_RESPONSE, phase, error_type=type(exc).__name__)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except (OSError, HTTPException) as exc:
                    if error is None:
                        error = MetadataError(MetadataErrorKind.NETWORK_ERROR, RequestPhase.CLOSE,
                                              error_type=type(exc).__name__)
        return MetadataSource(endpoint, started, received, utc_now(), official,
                              (monotonic_ns() - started_ns) // 1_000_000, status, body, truncated, error,
                              research_instrument)
