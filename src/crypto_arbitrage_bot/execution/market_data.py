"""Isolated testnet observations using the existing parsing and quality models.

Semantic endpoint enums identify the payload shape; actual_url explicitly
records the testnet origin, so persisted research never pretends to be mainnet.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic_ns
from typing import Any

from crypto_arbitrage_bot.exchanges.binance_spot import _levels
from crypto_arbitrage_bot.exchanges.binance_usdtm import (
    _parse_depth,
    _parse_open_interest,
    _parse_premium,
)
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import PublicGetRequest, RequestPhase
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionClient,
    TestnetExecutionError,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.market import Market, OrderBook
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly, SnapshotSources
from crypto_arbitrage_bot.models.spot import SpotDepthResult, SpotErrorKind, SpotFetchError
from crypto_arbitrage_bot.models.usdtm import (
    PerpDepth,
    UsdtmErrorKind,
    UsdtmFetchError,
    UsdtmFetchResult,
)
from crypto_arbitrage_bot.utils.clock import utc_now

_URLS = {
    PublicEndpoint.SPOT_DEPTH: "https://testnet.binance.vision/api/v3/depth",
    PublicEndpoint.PERP_DEPTH: "https://demo-fapi.binance.com/fapi/v1/depth",
    PublicEndpoint.PERP_PREMIUM_INDEX: "https://demo-fapi.binance.com/fapi/v1/premiumIndex",
    PublicEndpoint.PERP_OPEN_INTEREST: "https://demo-fapi.binance.com/fapi/v1/openInterest",
}


@dataclass(frozen=True, slots=True)
class TestnetGetRequest(PublicGetRequest):
    actual_url: str = field(init=False)

    def __post_init__(self) -> None:
        PublicGetRequest.__post_init__(self)
        if self.endpoint not in _URLS:
            raise ValueError("unsupported testnet observation")
        object.__setattr__(self, "actual_url", _URLS[self.endpoint])

    @property
    def url(self) -> str:
        return self.actual_url


@dataclass(frozen=True)
class ExitDepthObservation:
    """Fresh testnet depths for reduction only, never a complete research snapshot.

    Missing research fields are not manufactured. One-leg recovery requests only
    its own market; ordinary two-leg exits require both books and their skew.
    Original request/receive/E/T timestamps and raw payloads remain on each source.
    """

    symbol: str
    markets: tuple[Market, ...]
    spot_depth: SpotDepthResult | None
    perp_depth: UsdtmFetchResult[PerpDepth] | None
    collected_at: datetime

    def book(self, market: Market) -> OrderBook:
        if market not in self.markets:
            raise ValueError("exit market was not requested")
        book = (self.spot_depth.order_book if self.spot_depth is not None else None) \
            if market is Market.SPOT else (
                self.perp_depth.data.order_book
                if self.perp_depth is not None and isinstance(self.perp_depth.data, PerpDepth) else None)
        if book is None:
            raise ValueError("exit depth unavailable")
        return book


class TestnetCollector:
    def __init__(self, spot: TestnetExecutionClient, perp: TestnetExecutionClient,
                 *, clock: Callable[[], datetime] = utc_now) -> None:
        if spot.market is not ExecutionMarket.SPOT or perp.market is not ExecutionMarket.USDTM:
            raise ValueError("testnet collector requires matching Spot/USDT-M gateways")
        self.spot, self.perp, self.clock = spot, perp, clock

    def _fetch(self, endpoint: PublicEndpoint, symbol: str) -> Any:
        depth = endpoint in (PublicEndpoint.SPOT_DEPTH, PublicEndpoint.PERP_DEPTH)
        params = (("symbol", symbol), ("limit", 100)) if depth else (("symbol", symbol),)
        request = TestnetGetRequest(endpoint, params)
        started = self.clock()
        started_ns = monotonic_ns()
        raw: list[bytes] = []
        received = None
        data: Any = None
        error = None
        failure_status = None
        update_id = None
        try:
            client = self.spot if endpoint is PublicEndpoint.SPOT_DEPTH else self.perp
            action = "depth" if depth else "premium" if endpoint is PublicEndpoint.PERP_PREMIUM_INDEX else "interest"
            payload = client.public_payload(action, symbol, capture=raw.append)
            received = self.clock()
            if endpoint is PublicEndpoint.SPOT_DEPTH:
                update_id = payload.get("lastUpdateId")
                if type(update_id) is not int or update_id < 0:
                    raise ValueError("invalid Spot update ID")
                if "symbol" in payload and payload["symbol"] != symbol:
                    raise ValueError("Spot symbol mismatch")
                data = OrderBook(symbol, Market.SPOT, _levels(payload.get("bids"), "bids"),
                                 _levels(payload.get("asks"), "asks"), received)
                if len(data.bids) > 100 or len(data.asks) > 100:
                    raise ValueError("too many depth levels")
            else:
                data = (_parse_depth(payload, symbol, 100, received) if depth else
                        _parse_premium(payload, symbol, received) if endpoint is PublicEndpoint.PERP_PREMIUM_INDEX
                        else _parse_open_interest(payload, symbol, received))
        except (TestnetExecutionError, ValueError, TypeError, KeyError) as exc:
            data = None
            kind = exc.kind.value if isinstance(exc, TestnetExecutionError) else "INVALID_RESPONSE"
            safe_kind = kind if kind in ("NETWORK_ERROR", "RATE_LIMITED") else "INVALID_RESPONSE"
            message = str(exc)[:240] if isinstance(exc, TestnetExecutionError) else "invalid testnet market payload"
            api_code = exc.api_code if isinstance(exc, TestnetExecutionError) else None
            failure_status = exc.http_status if isinstance(exc, TestnetExecutionError) else None
            phase = {"OPEN": RequestPhase.REQUEST, "READ": RequestPhase.RESPONSE_BODY,
                     "DECODE": RequestPhase.PARSE}.get(exc.phase or "") if isinstance(exc, TestnetExecutionError) else RequestPhase.PARSE
            error = (SpotFetchError(SpotErrorKind(safe_kind), message, api_code, phase) if endpoint is PublicEndpoint.SPOT_DEPTH
                     else UsdtmFetchError(UsdtmErrorKind(safe_kind), message, api_code, phase))
        common: dict[str, Any] = dict(request=request, started_at=started, finished_at=self.clock(), received_at=received,
                      elapsed_ms=(monotonic_ns() - started_ns) // 1_000_000,
                      status_code=failure_status if failure_status is not None else 200 if raw else None, response_headers=(),
                      raw_response=raw[0] if raw else None, body_truncated=False, error=error)
        if endpoint is PublicEndpoint.SPOT_DEPTH:
            return SpotDepthResult(**common, order_book=data, last_update_id=update_id)
        return UsdtmFetchResult(**common, data=data)

    def collect(self, symbol: str) -> SnapshotAssembly:
        sources = SnapshotSources(*(self._fetch(endpoint, symbol) for endpoint in (
            PublicEndpoint.SPOT_DEPTH, PublicEndpoint.PERP_DEPTH,
            PublicEndpoint.PERP_PREMIUM_INDEX, PublicEndpoint.PERP_OPEN_INTEREST)))
        return assemble_snapshot(symbol, sources, assembled_at=self.clock())

    def collect_exit_depth(
        self, symbol: str, *, markets: tuple[Market, ...] = (Market.SPOT, Market.USDTM),
    ) -> ExitDepthObservation:
        """No premium/OI calls, old snapshots, account requests or order submissions."""
        if (not isinstance(markets, tuple) or not markets or len(set(markets)) != len(markets)
                or any(not isinstance(market, Market) for market in markets)):
            raise ValueError("explicit unique exit markets are required")
        # Validate the symbol before the first request, including single-leg recovery.
        TestnetGetRequest(PublicEndpoint.SPOT_DEPTH, (("symbol", symbol), ("limit", 100)))
        spot = self._fetch(PublicEndpoint.SPOT_DEPTH, symbol) if Market.SPOT in markets else None
        perp = self._fetch(PublicEndpoint.PERP_DEPTH, symbol) if Market.USDTM in markets else None
        return ExitDepthObservation(symbol, markets, spot, perp, self.clock())
