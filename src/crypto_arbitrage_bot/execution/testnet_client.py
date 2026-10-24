"""A narrowly scoped signed Binance testnet client.

The client has no production URL and no generic method/path escape hatch. Its
only order operation is an IOC limit order, and it never retries submissions.
"""

import gzip
import hashlib
import hmac
import json
import os
import re
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from io import BytesIO
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from crypto_arbitrage_bot.execution.types import (
    AccountState,
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
    SymbolTradingRules,
)
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS, Side
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

_BASE_URLS = {
    ExecutionMarket.SPOT: "https://testnet.binance.vision",
    ExecutionMarket.USDTM: "https://demo-fapi.binance.com",
}
_ROUTES = {
    (ExecutionMarket.SPOT, "time"): ("GET", "/api/v3/time"),
    (ExecutionMarket.USDTM, "time"): ("GET", "/fapi/v1/time"),
    (ExecutionMarket.SPOT, "exchange_info"): ("GET", "/api/v3/exchangeInfo"),
    (ExecutionMarket.USDTM, "exchange_info"): ("GET", "/fapi/v1/exchangeInfo"),
    (ExecutionMarket.SPOT, "order"): ("POST", "/api/v3/order"),
    (ExecutionMarket.USDTM, "order"): ("POST", "/fapi/v1/order"),
    (ExecutionMarket.SPOT, "query_order"): ("GET", "/api/v3/order"),
    (ExecutionMarket.USDTM, "query_order"): ("GET", "/fapi/v1/order"),
    (ExecutionMarket.SPOT, "depth"): ("GET", "/api/v3/depth"),
    (ExecutionMarket.USDTM, "depth"): ("GET", "/fapi/v1/depth"),
    (ExecutionMarket.USDTM, "premium"): ("GET", "/fapi/v1/premiumIndex"),
    (ExecutionMarket.USDTM, "interest"): ("GET", "/fapi/v1/openInterest"),
    (ExecutionMarket.SPOT, "account"): ("GET", "/api/v3/account"),
    (ExecutionMarket.USDTM, "account"): ("GET", "/fapi/v2/account"),
    (ExecutionMarket.SPOT, "open_orders"): ("GET", "/api/v3/openOrders"),
    (ExecutionMarket.USDTM, "open_orders"): ("GET", "/fapi/v1/openOrders"),
    (ExecutionMarket.USDTM, "positions"): ("GET", "/fapi/v2/positionRisk"),
    (ExecutionMarket.USDTM, "position_mode"): ("GET", "/fapi/v1/positionSide/dual"),
    (ExecutionMarket.USDTM, "prepare_one_x"): ("POST", "/fapi/v1/leverage"),
    (ExecutionMarket.SPOT, "trades"): ("GET", "/api/v3/myTrades"),
    (ExecutionMarket.USDTM, "trades"): ("GET", "/fapi/v1/userTrades"),
    (ExecutionMarket.USDTM, "funding_payments"): ("GET", "/fapi/v1/income"),
}
_ALLOWED_SYMBOLS = frozenset(SUPPORTED_SYMBOLS)
_ERROR_TEXT_LIMIT = 240


class ExecutionErrorKind(StrEnum):
    DISABLED = "DISABLED"
    CREDENTIALS_MISSING = "CREDENTIALS_MISSING"
    INVALID_RULES = "INVALID_RULES"
    ORDER_GUARD_REJECTED = "ORDER_GUARD_REJECTED"
    HTTP_REJECTED = "HTTP_REJECTED"
    RATE_LIMITED = "RATE_LIMITED"
    NETWORK_ERROR = "NETWORK_ERROR"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    REDIRECT_BLOCKED = "REDIRECT_BLOCKED"


class TestnetExecutionError(RuntimeError):
    """Structured safe error; never includes the signed URL or request secret."""

    def __init__(
        self,
        kind: ExecutionErrorKind,
        message: str,
        *,
        market: ExecutionMarket | None = None,
        http_status: int | None = None,
        api_code: int | None = None,
        submission_may_have_reached_exchange: bool = False,
    ) -> None:
        self.kind = kind
        self.message = message[:_ERROR_TEXT_LIMIT]
        self.market = market
        self.http_status = http_status
        self.api_code = api_code
        self.submission_may_have_reached_exchange = submission_may_have_reached_exchange
        self.operation: str | None = None
        self.parent_operation: str | None = None
        self.endpoint: str | None = None
        self.method: str | None = None
        self.phase: str | None = None
        self.elapsed_ms: int | None = None
        self.timeout_seconds: float | None = None
        self.transport_error_type: str | None = None
        super().__init__(f"{kind}: {self.message}")

    def diagnostic_fields(self) -> dict[str, object]:
        """Fixed metadata only: never include signed URLs, bodies or API messages."""
        return {"kind": self.kind.value, "market": self.market, "http_status": self.http_status,
                "api_code": self.api_code, "operation": self.operation,
                "parent_operation": self.parent_operation, "endpoint": self.endpoint,
                "method": self.method, "phase": self.phase, "elapsed_ms": self.elapsed_ms,
                "timeout_seconds": None if self.timeout_seconds is None else str(self.timeout_seconds),
                "transport_error_type": self.transport_error_type,
                "submission_may_have_reached_exchange": self.submission_may_have_reached_exchange}


@dataclass(frozen=True, slots=True)
class TestnetCredentials:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.api_key) is not str or not self.api_key.strip():
            raise ValueError("testnet API key is required")
        if type(self.api_secret) is not str or not self.api_secret.strip():
            raise ValueError("testnet API secret is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", self.api_key, re.ASCII):
            raise ValueError("testnet API key contains unsupported header characters")
        if "\r" in self.api_secret or "\n" in self.api_secret:
            raise ValueError("testnet API secret contains unsupported characters")

    @classmethod
    def from_env(cls, market: ExecutionMarket, environ: Mapping[str, str] | None = None) -> "TestnetCredentials":
        env = os.environ if environ is None else environ
        prefix = "CAB_TESTNET_SPOT" if market is ExecutionMarket.SPOT else "CAB_TESTNET_USDTM"
        key = env.get(f"{prefix}_API_KEY", "")
        secret = env.get(f"{prefix}_API_SECRET", "")
        if not key or not secret:
            raise TestnetExecutionError(
                ExecutionErrorKind.CREDENTIALS_MISSING,
                f"{prefix}_API_KEY and {prefix}_API_SECRET are required",
                market=market,
            )
        return cls(key, secret)


@dataclass(frozen=True, slots=True)
class TestnetExecutionPolicy:
    max_order_notional: Decimal
    timeout_seconds: float = 5.0
    recv_window_ms: int = 5000

    def __post_init__(self) -> None:
        if not isinstance(self.max_order_notional, Decimal) or not self.max_order_notional.is_finite():
            raise TypeError("max_order_notional must be a finite Decimal")
        if self.max_order_notional <= 0:
            raise ValueError("max_order_notional must be positive")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not 0 < self.timeout_seconds < float("inf")):
            raise ValueError("timeout_seconds must be positive")
        if type(self.recv_window_ms) is not int or not 1 <= self.recv_window_ms <= 60000:
            raise ValueError("recv_window_ms must be between 1 and 60000")


class _Opener(Protocol):
    def open(self, fullurl: Request, data: bytes | None = None, timeout: float = ...) -> Any: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _decimal(value: Any, name: str) -> Decimal:
    if type(value) not in (str, int, Decimal):
        raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, f"{name} is not a decimal string")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, f"{name} is invalid") from error
    if not parsed.is_finite():
        raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, f"{name} is not finite")
    return parsed


class TestnetExecutionClient:
    """Calls fixed testnet hosts only; instantiate explicitly from separate test keys."""

    def __init__(
        self,
        market: ExecutionMarket,
        credentials: TestnetCredentials,
        policy: TestnetExecutionPolicy,
        *,
        opener: _Opener | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(market, ExecutionMarket):
            raise TypeError("market must be an ExecutionMarket")
        self.market = market
        self._credentials = credentials
        self.policy = policy
        self.uses_default_transport = opener is None
        self._opener = opener or build_opener(ProxyHandler(), _NoRedirect())
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._server_offset_ms: int | None = None
        self._last_time_sync_ms: int | None = None
        self._latest_symbol_rules: dict[str, tuple[SymbolTradingRules, int]] = {}
        self._preflight_rules: tuple[SymbolTradingRules, int] | None = None

    @classmethod
    def from_env(
        cls,
        market: ExecutionMarket,
        policy: TestnetExecutionPolicy,
        *,
        environ: Mapping[str, str] | None = None,
        opener: _Opener | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> "TestnetExecutionClient":
        env = os.environ if environ is None else environ
        if env.get("CAB_EXECUTION_MODE") != "TESTNET":
            raise TestnetExecutionError(
                ExecutionErrorKind.DISABLED,
                "set CAB_EXECUTION_MODE=TESTNET to construct the isolated testnet client",
                market=market,
            )
        return cls(market, TestnetCredentials.from_env(market, env), policy, opener=opener, clock_ms=clock_ms)

    def _route(self, action: str) -> tuple[str, str]:
        try:
            return _ROUTES[(self.market, action)]
        except KeyError as error:
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "unsupported internal route",
                                        market=self.market) from error

    def _request(self, action: str, params: tuple[tuple[str, str | int], ...] = (), *,
                signed: bool = False, capture: Callable[[bytes], None] | None = None,
                submit_before: datetime | None = None) -> Any:
        started = time.monotonic_ns()
        method, path = self._route(action)
        try:
            return self._perform_request(action, params, signed=signed, capture=capture, submit_before=submit_before)
        except TestnetExecutionError as error:
            if error.operation is None:
                error.operation, error.endpoint, error.method = action, path, method
                error.phase = error.phase or "PREFLIGHT"
                error.elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
                error.timeout_seconds = self.policy.timeout_seconds
            elif error.operation != action:
                # A failed /time is a distinct operation, not a failed account/order send.
                error.parent_operation = action
            raise

    def _perform_request(self, action: str, params: tuple[tuple[str, str | int], ...] = (), *,
                         signed: bool = False, capture: Callable[[bytes], None] | None = None,
                         submit_before: datetime | None = None) -> Any:
        method, path = self._route(action)
        payload = list(params)
        headers = {"Accept": "application/json", "User-Agent": "crypto-arbitrage-bot-testnet/0.1"}
        # Only large read-only payloads negotiate compression. Order request
        # bodies, signing, deadlines and submission ambiguity are unchanged.
        if action in ("account", "exchange_info"):
            headers["Accept-Encoding"] = "gzip"
        if signed:
            now_ms = self._clock_ms()
            if action != "time" and (self._last_time_sync_ms is None or
                                      not 0 <= now_ms - self._last_time_sync_ms < 30000):
                self._sync_time()
            payload.extend((("recvWindow", self.policy.recv_window_ms),
                            ("timestamp", self._clock_ms() + (self._server_offset_ms or 0))))
            encoded = urlencode(payload)
            signature = hmac.new(self._credentials.api_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
            payload.append(("signature", signature))
            headers["X-MBX-APIKEY"] = self._credentials.api_key
        if method == "GET":
            url = f"{_BASE_URLS[self.market]}{path}"
            if payload:
                url = f"{url}?{urlencode(payload)}"
            request = Request(url, headers=headers, method="GET")
        else:
            url = f"{_BASE_URLS[self.market]}{path}"
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            request = Request(url, data=urlencode(payload).encode("ascii"), headers=headers, method="POST")
        phase = "SEND_GUARD"
        try:
            response_limit = 8_000_000 if action == "exchange_info" else 1_000_000
            if submit_before is not None:
                deadline = utc_timestamp(submit_before)
                epoch = datetime(1970, 1, 1, tzinfo=UTC)
                deadline_ms = (deadline - epoch) // timedelta(milliseconds=1)
                # Run AFTER exchange-info/time-sync network calls, immediately
                # before the send. Reject locally: no POST reached the exchange.
                local_ms = self._clock_ms()
                # A delayed /time reply must not move an already expired local
                # deadline back into the future. Either clock expires the entry.
                if max(local_ms, local_ms + (self._server_offset_ms or 0)) >= deadline_ms:
                    raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                                "ENTRY_SUBMISSION_DEADLINE_PASSED", market=self.market)
            phase = "OPEN"
            response = self._opener.open(request, timeout=self.policy.timeout_seconds)
            with response:
                status = getattr(response, "status", 200)
                phase = "READ"
                body = response.read(response_limit + 1)
                encoding = (getattr(response, "headers", None) or {}).get("Content-Encoding", "identity")
            if status < 200 or status >= 300:
                raise TestnetExecutionError(ExecutionErrorKind.HTTP_REJECTED, f"HTTP {status}",
                                            market=self.market, http_status=status,
                                            submission_may_have_reached_exchange=method == "POST")
        except HTTPError as error:
            try:
                body = error.read(4097)
            except (TimeoutError, URLError, OSError) as read_error:
                failure = TestnetExecutionError(
                    ExecutionErrorKind.SUBMISSION_UNKNOWN if method == "POST" else ExecutionErrorKind.NETWORK_ERROR,
                    type(read_error).__name__, market=self.market, http_status=error.code,
                    submission_may_have_reached_exchange=method == "POST")
                failure.phase = "READ"
                cause = (read_error.reason if isinstance(read_error, URLError)
                         and isinstance(read_error.reason, BaseException) else read_error)
                failure.transport_error_type = type(cause).__name__
                raise failure from None
            try:
                error_encoding = error.headers.get("Content-Encoding", "identity") if error.headers is not None else "identity"
                body = self._decode_body(body, error_encoding, 4096)
            except ValueError:
                # Preserve HTTP/rate-limit/ambiguous-POST safety semantics even
                # when an error payload is oversized or cannot be decoded.
                body = b""
            parsed = self._safe_error_payload(body)
            message = parsed.get("msg", f"HTTP {error.code}")
            for credential in (self._credentials.api_key, self._credentials.api_secret):
                message = message.replace(credential, "[redacted]")
            kind = (ExecutionErrorKind.REDIRECT_BLOCKED if 300 <= error.code < 400 else
                    ExecutionErrorKind.RATE_LIMITED if error.code == 429 else
                    ExecutionErrorKind.SUBMISSION_UNKNOWN if method == "POST" and
                    (error.code >= 500 or parsed.get("code") in (-1006, -1007)) else
                    ExecutionErrorKind.HTTP_REJECTED)
            failure = TestnetExecutionError(
                kind, message, market=self.market,
                http_status=error.code, api_code=parsed.get("code"),
                submission_may_have_reached_exchange=method == "POST" and
                (error.code >= 500 or parsed.get("code") in (-1006, -1007)),
            )
            failure.phase = phase
            raise failure from None
        except (TimeoutError, URLError, OSError) as error:
            failure = TestnetExecutionError(
                ExecutionErrorKind.SUBMISSION_UNKNOWN if method == "POST" else ExecutionErrorKind.NETWORK_ERROR,
                type(error).__name__, market=self.market,
                submission_may_have_reached_exchange=method == "POST",
            )
            failure.phase = phase
            cause = error.reason if isinstance(error, URLError) and isinstance(error.reason, BaseException) else error
            failure.transport_error_type = type(cause).__name__
            raise failure from None
        except TestnetExecutionError as error:
            error.phase = error.phase or phase
            raise
        response_limit = 8_000_000 if action == "exchange_info" else 1_000_000
        if len(body) > response_limit:
            invalid = TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "response exceeds configured size limit",
                                         market=self.market, submission_may_have_reached_exchange=method == "POST")
            invalid.phase = "DECODE"
            raise invalid
        try:
            body = self._decode_body(body, encoding, response_limit)
        except ValueError:
            invalid = TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE,
                "compressed response is invalid or exceeds size limit", market=self.market,
                submission_may_have_reached_exchange=method == "POST")
            invalid.phase = "DECODE"
            raise invalid from None
        try:
            result = json.loads(body, parse_float=Decimal, parse_int=int)
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid = TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "response is not valid JSON",
                                         market=self.market, submission_may_have_reached_exchange=method == "POST")
            invalid.phase = "DECODE"
            raise invalid from None
        expected = list if action in ("positions", "open_orders", "trades", "funding_payments") else dict
        if not isinstance(result, expected):
            invalid = TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "response must be a JSON object",
                                         market=self.market, submission_may_have_reached_exchange=method == "POST")
            invalid.phase = "DECODE"
            raise invalid
        if isinstance(result, dict) and type(result.get("code")) is int and result["code"] < 0:
            invalid = TestnetExecutionError(
                ExecutionErrorKind.SUBMISSION_UNKNOWN if method == "POST" else ExecutionErrorKind.HTTP_REJECTED,
                "API returned an error code", market=self.market, api_code=result["code"],
                submission_may_have_reached_exchange=method == "POST",
            )
            invalid.phase = "DECODE"
            raise invalid
        if capture is not None:
            capture(body)
        return result

    @staticmethod
    def _decode_body(body: bytes, encoding: str, limit: int) -> bytes:
        """Bound both wire and decoded sizes; never expand a gzip bomb fully."""
        if len(body) > limit or not isinstance(encoding, str):
            raise ValueError("response exceeds limit or has invalid encoding")
        kind = encoding.strip().lower()
        if kind in ("", "identity"):
            return body
        if kind != "gzip":
            raise ValueError("unsupported content encoding")
        try:
            with gzip.GzipFile(fileobj=BytesIO(body), mode="rb") as compressed:
                decoded = compressed.read(limit + 1)
        except (OSError, EOFError, zlib.error):
            raise ValueError("invalid gzip response") from None
        if len(decoded) > limit:
            raise ValueError("decoded response exceeds limit")
        return decoded

    def public_payload(self, action: str, symbol: str, *, capture: Callable[[bytes], None]) -> dict[str, Any]:
        if symbol not in _ALLOWED_SYMBOLS or action not in ("depth", "premium", "interest"):
            raise ValueError("unsupported testnet public market request")
        params: tuple[tuple[str, str | int], ...] = (("symbol", symbol),)
        if action == "depth":
            params += (("limit", 100),)
        result = self._request(action, params, capture=capture)
        return dict(result)

    def get_account_state(self, symbol: str) -> AccountState:
        if symbol not in _ALLOWED_SYMBOLS:
            raise ValueError("unsupported symbol")
        account = self._request("account", signed=True)
        orders = self._request("open_orders", (("symbol", symbol),), signed=True)
        try:
            if type(account["canTrade"]) is not bool:
                raise ValueError("canTrade must be boolean")
            if self.market is ExecutionMarket.SPOT:
                balances = account["balances"]
                if not isinstance(balances, list):
                    raise ValueError("balances must be a list")
                assets = {row["asset"]: _decimal(row["free"], "free") for row in balances}
                if len(assets) != len(balances) or "USDT" not in assets or symbol[:-4] not in assets:
                    raise ValueError("required balances are absent or duplicated")
                return AccountState(self.market, symbol, assets["USDT"], assets[symbol[:-4]],
                                    Decimal("0"), account["canTrade"], True, True, Decimal("1"), len(orders))
            positions = self._request("positions", (("symbol", symbol),), signed=True)
            matching = [row for row in positions if row.get("symbol") == symbol]
            mode = self._request("position_mode", signed=True)
            if len(matching) != 1 or type(mode["dualSidePosition"]) is not bool:
                raise ValueError("position identity/mode is ambiguous")
            position = matching[0]
            if position["positionSide"] != "BOTH" or type(account["multiAssetsMargin"]) is not bool:
                raise ValueError("only one-way single-asset positions are supported")
            return AccountState(
                self.market, symbol, _decimal(account["availableBalance"], "availableBalance"), Decimal("0"),
                _decimal(position["positionAmt"], "positionAmt"), account["canTrade"],
                not mode["dualSidePosition"], not account["multiAssetsMargin"],
                _decimal(position["leverage"], "leverage"), len(orders),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "invalid testnet account state",
                                        market=self.market) from error

    def get_funding_payments(self, symbol: str, opened_at: datetime,
                             closed_at: datetime) -> tuple[FundingPayment, ...]:
        """Bounded Demo-only signed GET of actual USDT funding ledger rows."""
        if self.market is not ExecutionMarket.USDTM or symbol not in _ALLOWED_SYMBOLS:
            raise ValueError("funding ledger requires supported USDT-M Demo symbol")
        opened, closed = utc_timestamp(opened_at), utc_timestamp(closed_at)
        if closed <= opened or closed - opened > timedelta(days=7):
            raise ValueError("funding ledger window must be positive and at most 7 days")
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        start_ms = (opened - epoch) // timedelta(milliseconds=1) + 1
        end_ms = (closed - epoch) // timedelta(milliseconds=1)
        if start_ms > end_ms:
            return ()
        payments: dict[str, FundingPayment] = {}
        for page in range(1, 4):
            rows = self._request("funding_payments", (("symbol", symbol), ("incomeType", "FUNDING_FEE"),
                ("startTime", start_ms), ("endTime", end_ms), ("limit", 1000), ("page", page)), signed=True)
            if len(rows) > 1000:
                raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "funding page exceeds limit")
            try:
                for row in rows:
                    if (not isinstance(row, dict) or row["symbol"] != symbol or row["incomeType"] != "FUNDING_FEE"
                            or row["asset"] != "USDT" or type(row["time"]) is not int
                            or not isinstance(row["income"], str)
                            or type(row["tranId"]) not in (str, int) or isinstance(row["tranId"], bool)):
                        raise ValueError("funding ledger identity/asset/type mismatch")
                    posted = epoch + timedelta(milliseconds=row["time"])
                    if not opened < posted <= closed:
                        raise ValueError("funding payment outside actual holding window")
                    payment = FundingPayment(str(row["tranId"]), symbol, posted, _decimal(row["income"], "funding income"))
                    if payment.transaction_id in payments:
                        raise ValueError("duplicate funding transaction")
                    payments[payment.transaction_id] = payment
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "invalid funding ledger payload") from error
            if len(rows) < 1000:
                return tuple(sorted(payments.values(), key=lambda payment: (payment.posted_at, payment.transaction_id)))
        raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "funding ledger pagination limit reached")

    def ensure_one_x(self, symbol: str) -> AccountState:
        """Prepare a flat Demo account at exactly 1x; never raise leverage.

        A single signed POST is followed by a fresh account read. A timeout or
        ambiguous reply is not retried, and cannot authorize subsequent orders.
        """
        if self.market is not ExecutionMarket.USDTM or symbol not in _ALLOWED_SYMBOLS:
            raise ValueError("1x preparation requires a supported USDT-M Demo symbol")
        before = self.get_account_state(symbol)
        if (not before.can_trade or not before.one_way or not before.single_asset
                or before.open_order_count or before.position_quantity != 0 or before.leverage < 1):
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "1x preparation requires an enabled flat one-way single-asset account without orders",
                                        market=self.market)
        if before.leverage == 1:
            return before
        result = self._request("prepare_one_x", (("symbol", symbol), ("leverage", 1)), signed=True)
        if result.get("symbol") != symbol or type(result.get("leverage")) is not int or result["leverage"] != 1:
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE,
                                        "1x preparation response was not confirmed", market=self.market)
        after = self.get_account_state(symbol)
        if (after.leverage != 1 or not after.can_trade or not after.one_way or not after.single_asset
                or after.open_order_count or after.position_quantity != 0):
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "1x preparation account recheck failed", market=self.market)
        return after

    @staticmethod
    def _safe_error_payload(body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body, parse_float=Decimal)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        code = payload.get("code")
        message = payload.get("msg")
        safe: dict[str, Any] = {}
        if type(code) is int:
            safe["code"] = code
        if isinstance(message, str):
            safe["msg"] = re.sub(r"[\r\n\t]+", " ", message)[:_ERROR_TEXT_LIMIT]
        return safe

    def _sync_time(self) -> None:
        started = time.monotonic_ns()
        result = self._request("time")
        server_time = result.get("serverTime")
        if type(server_time) is not int or server_time <= 0:
            failure = TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "serverTime is missing or invalid",
                                           market=self.market)
            failure.operation, failure.method = "time", "GET"
            failure.endpoint = self._route("time")[1]
            failure.phase = "DECODE"
            failure.elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
            failure.timeout_seconds = self.policy.timeout_seconds
            raise failure
        self._server_offset_ms = server_time - self._clock_ms()
        self._last_time_sync_ms = self._clock_ms()

    def get_symbol_rules(self, symbol: str) -> SymbolTradingRules:
        if symbol not in _ALLOWED_SYMBOLS:
            raise ValueError("only BTCUSDT and ETHUSDT are supported")
        params = (("symbol", symbol),) if self.market is ExecutionMarket.SPOT else ()
        result = self._request("exchange_info", params)
        symbols = result.get("symbols")
        if not isinstance(symbols, list):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RULES, "exchangeInfo symbol is missing",
                                        market=self.market)
        matching = [entry for entry in symbols if isinstance(entry, dict) and entry.get("symbol") == symbol]
        if len(matching) != 1:
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RULES, "exchangeInfo symbol is missing or ambiguous",
                                        market=self.market)
        entry = matching[0]
        if "status" in entry and entry["status"] != "TRADING":
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RULES, "symbol is not currently trading")
        filters = entry.get("filters")
        if not isinstance(filters, list):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RULES, "exchangeInfo symbol mismatch",
                                        market=self.market)
        by_type = {item.get("filterType"): item for item in filters if isinstance(item, dict)}
        try:
            price = by_type["PRICE_FILTER"]
            lot = by_type["LOT_SIZE"]
            notional_filter = by_type.get("NOTIONAL") or by_type["MIN_NOTIONAL"]
            min_notional = notional_filter.get("minNotional", notional_filter.get("notional"))
            rules = SymbolTradingRules(
                symbol=symbol, market=self.market,
                min_price=_decimal(price["minPrice"], "minPrice"),
                max_price=(None if _decimal(price["maxPrice"], "maxPrice") == 0
                           else _decimal(price["maxPrice"], "maxPrice")),
                tick_size=_decimal(price["tickSize"], "tickSize"),
                step_size=_decimal(lot["stepSize"], "stepSize"),
                min_quantity=_decimal(lot["minQty"], "minQty"),
                max_quantity=_decimal(lot["maxQty"], "maxQty"),
                min_notional=_decimal(min_notional, "minNotional"),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, TestnetExecutionError):
                raise
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RULES, "required symbol filters are malformed",
                                        market=self.market) from None
        self._latest_symbol_rules[symbol] = (rules, time.monotonic_ns())
        return rules

    @contextmanager
    def preflight_rules(self, rules: SymbolTradingRules, *, max_age: timedelta) -> Iterator[None]:
        """Reuse this client's fresh rule response only in one guarded exit.

        Price/account snapshots are never cached. Validation still runs for
        every order. An old rule, unrelated object or completed scope cannot
        bypass the usual fresh exchangeInfo GET before submission.
        """
        if max_age <= timedelta(0):
            raise ValueError("preflight rule age must be positive")
        recent = self._latest_symbol_rules.get(rules.symbol)
        if (rules.market is not self.market or recent is None or recent[0] is not rules
                or self._preflight_rules is not None):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RULES,
                                        "rules must come from this client's current preflight", market=self.market)
        self._preflight_rules = (rules, (max_age // timedelta(microseconds=1)) * 1000)
        try:
            yield
        finally:
            self._preflight_rules = None

    def place_ioc_limit_order(self, order: IocLimitOrder) -> OrderReceipt:
        if order.market is not self.market:
            raise ValueError("order market does not match client")
        scoped = self._preflight_rules[0] if self._preflight_rules is not None else None
        recent = self._latest_symbol_rules.get(order.symbol)
        fresh = (scoped is not None and recent is not None and recent[0] is scoped
                 and self._preflight_rules is not None
                 and 0 <= time.monotonic_ns() - recent[1] < self._preflight_rules[1])
        rules = scoped if fresh else self.get_symbol_rules(order.symbol)
        assert rules is not None
        self._validate_order(order, rules)
        params: tuple[tuple[str, str | int], ...] = (
            ("symbol", order.symbol), ("side", order.side.value), ("type", "LIMIT"),
            ("timeInForce", "IOC"), ("quantity", format(order.quantity, "f")),
            ("price", format(order.limit_price, "f")), ("newClientOrderId", order.client_order_id),
            ("newOrderRespType", "FULL" if self.market is ExecutionMarket.SPOT else "RESULT"),
        )
        if order.reduce_only:
            params += (("reduceOnly", "true"),)
        result = self._request("order", params, signed=True, submit_before=order.submit_before)
        try:
            receipt = self._receipt(result, order.symbol, order.client_order_id)
            if not receipt.matches_order(order):
                raise ValueError("order response quantity mismatch")
            return receipt
        except (TestnetExecutionError, ValueError, TypeError, KeyError):
            raise TestnetExecutionError(ExecutionErrorKind.SUBMISSION_UNKNOWN, "order receipt is inconsistent",
                                        market=self.market, submission_may_have_reached_exchange=True) from None

    def get_order(self, symbol: str, client_order_id: str, *, include_base_commission: bool = True) -> OrderReceipt:
        if symbol not in _ALLOWED_SYMBOLS:
            raise ValueError("only BTCUSDT and ETHUSDT are supported")
        if type(client_order_id) is not str or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,36}", client_order_id):
            raise ValueError("client_order_id must be 1-36 safe ASCII characters")
        result = self._request("query_order", (("symbol", symbol), ("origClientOrderId", client_order_id)), signed=True)
        try:
            receipt = self._receipt(result, symbol, client_order_id)
        except (TestnetExecutionError, ValueError, TypeError, KeyError):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "invalid queried order receipt",
                                        market=self.market) from None
        if include_base_commission and self.market is ExecutionMarket.SPOT and receipt.executed_quantity > 0:
            from dataclasses import replace
            trades = self._request("trades", (("symbol", symbol), ("orderId", receipt.order_id),
                                             ("limit", 1000)), signed=True)
            try:
                selected = [trade for trade in trades if str(trade["orderId"]) == receipt.order_id
                            and trade["symbol"] == symbol]
                if sum((_decimal(t["qty"], "qty") for t in selected), Decimal("0")) != receipt.executed_quantity:
                    raise ValueError("order fills are incomplete")
                fee = sum((_decimal(t["commission"], "commission") for t in selected
                           if t["commissionAsset"] == symbol[:-4]), Decimal("0"))
                receipt = replace(receipt, base_commission=fee)
            except (KeyError, TypeError, ValueError):
                raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "order trade fees are incomplete") from None
        return receipt

    def get_order_trades(self, symbol: str, order_id: str) -> tuple[AccountTrade, ...]:
        """Signed GETs on fixed Demo hosts, at most three pages, no partial success.

        Request earliest fills for this order. Completeness is also checked
        against the queried order by the separate reconciliation layer.
        """
        if symbol not in _ALLOWED_SYMBOLS or type(order_id) is not str or not re.fullmatch(r"[0-9]{1,20}", order_id):
            raise ValueError("supported symbol and numeric order id are required")
        trades: list[AccountTrade] = []
        seen: set[str] = set()
        from_id = 0
        for _ in range(3):
            payload = self._request("trades", (("symbol", symbol), ("orderId", order_id),
                ("fromId", from_id), ("limit", 1000)), signed=True)
            try:
                if len(payload) > 1000:
                    raise ValueError("oversized trade page")
                for row in payload:
                    if not isinstance(row, dict) or row["symbol"] != symbol or str(row["orderId"]) != order_id:
                        raise ValueError("trade identity mismatch")
                    for key in ("id", "orderId", "time"):
                        if type(row[key]) is not int or row[key] < 0:
                            raise ValueError("invalid numeric trade identity/time")
                    if row["time"] == 0 or row["id"] < from_id or str(row["id"]) in seen:
                        raise ValueError("duplicate or out-of-page trade")
                    buyer = row["isBuyer" if self.market is ExecutionMarket.SPOT else "buyer"]
                    if type(buyer) is not bool:
                        raise ValueError("trade buyer flag is missing")
                    side = Side.BUY if buyer else Side.SELL
                    if self.market is ExecutionMarket.USDTM and (row["side"] != side.value or row["positionSide"] != "BOTH"):
                        raise ValueError("unsupported trade position/side")
                    numeric_fields = ("price", "qty", "quoteQty", "commission") + (
                        ("realizedPnl",) if self.market is ExecutionMarket.USDTM else ())
                    if any(type(row[name]) is not str for name in numeric_fields):
                        raise ValueError("official financial fields must be decimal strings")
                    trade = AccountTrade(self.market, symbol, str(row["id"]), order_id, side,
                        _decimal(row["price"], "price"), _decimal(row["qty"], "qty"),
                        _decimal(row["quoteQty"], "quoteQty"), _decimal(row["commission"], "commission"),
                        row["commissionAsset"], datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=row["time"]),
                        None if self.market is ExecutionMarket.SPOT else _decimal(row["realizedPnl"], "realizedPnl"),
                        json.dumps(row, default=str, sort_keys=True, separators=(",", ":")))
                    trades.append(trade)
                    seen.add(trade.trade_id)
                if len(payload) < 1000:
                    return tuple(trades)
                from_id = max(int(row["id"]) for row in payload) + 1
            except (KeyError, TypeError, ValueError, OverflowError):
                raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE,
                                            "invalid account trade payload", market=self.market) from None
        raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE,
                                    "account trade pagination limit exceeded", market=self.market)

    def _validate_order(self, order: IocLimitOrder, rules: SymbolTradingRules) -> None:
        if order.quantity < rules.min_quantity or order.quantity > rules.max_quantity:
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "quantity outside current exchange filters", market=self.market)
        if order.quantity % rules.step_size != 0:
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "quantity is not aligned to stepSize", market=self.market)
        if order.limit_price < rules.min_price or (rules.max_price is not None and order.limit_price > rules.max_price):
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "limit_price outside current exchange filters", market=self.market)
        if order.limit_price % rules.tick_size != 0:
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "limit_price is not aligned to tickSize", market=self.market)
        notional = order.quantity * order.limit_price
        if notional > self.policy.max_order_notional:
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "order exceeds configured testnet notional cap", market=self.market)
        if notional < rules.min_notional and not order.reduce_only:
            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,
                                        "order is below exchange minNotional", market=self.market)

    def _receipt(self, result: dict[str, Any], symbol: str, client_order_id: str) -> OrderReceipt:
        if result.get("symbol") != symbol or result.get("clientOrderId") != client_order_id:
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "order identity mismatch",
                                        market=self.market, submission_may_have_reached_exchange=True)
        status_text = result.get("status")
        if not isinstance(status_text, str):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "order status is missing",
                                        market=self.market,
                                        submission_may_have_reached_exchange=True)
        try:
            status = OrderState(status_text)
        except (ValueError, TypeError):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "unknown order status",
                                        market=self.market,
                                        submission_may_have_reached_exchange=True) from None
        order_id = result.get("orderId")
        response_client_id = result.get("clientOrderId", client_order_id)
        if (type(order_id) not in (int, str) or not str(order_id).isascii()
                or not str(order_id).isdigit() or str(response_client_id) != client_order_id):
            raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE, "order identity mismatch",
                                        market=self.market,
                                        submission_may_have_reached_exchange=True)
        original = _decimal(result.get("origQty"), "origQty")
        executed = _decimal(result.get("executedQty"), "executedQty")
        quote_value = result.get("cummulativeQuoteQty", result.get("cumQuote"))
        average_value = result.get("avgPrice")
        quote = None if quote_value is None else _decimal(quote_value, "cumulativeQuoteQty")
        average = None if average_value is None else _decimal(average_value, "avgPrice")
        if (type(result.get('price')) is not str or type(result.get('type')) is not str
                or type(result.get('timeInForce')) is not str):
            raise ValueError('order semantics are missing or malformed')
        if self.market is ExecutionMarket.USDTM and (
                type(result.get('reduceOnly')) is not bool or type(result.get('positionSide')) is not str):
            raise ValueError('perpetual order semantics are missing or malformed')
        base_fee: Decimal | None = Decimal("0") if self.market is ExecutionMarket.USDTM or executed == 0 else None
        fills = result.get("fills")
        if self.market is ExecutionMarket.SPOT and isinstance(fills, list):
            if sum((_decimal(row["qty"], "fill qty") for row in fills), Decimal("0")) != executed:
                raise ValueError("fills do not match executed quantity")
            side = Side(result['side'])
            price_limit = _decimal(result['price'], 'price')
            for row in fills:
                fill_price = _decimal(row['price'], 'fill price')
                if (_decimal(row['qty'], 'fill qty') <= 0 or fill_price <= 0
                        or (fill_price > price_limit if side is Side.BUY else fill_price < price_limit)):
                    raise ValueError('fill exceeds order limit')
            base_fee = sum((_decimal(row["commission"], "commission") for row in fills
                            if row["commissionAsset"] == symbol[:-4]), Decimal("0"))
        return OrderReceipt(self.market, symbol, str(order_id), client_order_id, status,
                            original, executed, quote, average, base_fee, side=Side(result['side']),
                            order_type=result['type'], time_in_force=result['timeInForce'],
                            limit_price=_decimal(result['price'], 'price'),
                            reduce_only=result.get('reduceOnly'), position_side=result.get('positionSide'))
