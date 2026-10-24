"""Pure depth-only quality gate for confirmed testnet risk reduction.

This is not an entry/research Data Quality override. Reuse the exact existing
book, Decimal and timestamp checks, but never require OI, Mark or Funding to
sell known inventory or reduce a confirmed short. Callers still prove account,
order, inventory and bounded-recovery state before submitting any exit.
"""

from datetime import datetime

from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.execution.market_data import ExitDepthObservation, TestnetGetRequest
from crypto_arbitrage_bot.market_data.quality import _Checks
from crypto_arbitrage_bot.models.data_quality import (
    DataQualityConfig,
    DataQualityReason,
    DataQualityResult,
)
from crypto_arbitrage_bot.models.market import Market, OrderBook, require_market_symbol
from crypto_arbitrage_bot.models.spot import SpotDepthResult
from crypto_arbitrage_bot.models.usdtm import PerpDepth, UsdtmFetchResult
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

R = DataQualityReason


def validate_exit_depth(
    observation: ExitDepthObservation, config: DataQualityConfig, *, evaluated_at: datetime,
    markets: tuple[Market, ...] = (Market.SPOT, Market.USDTM),
) -> DataQualityResult:
    """Scope is supplied by the exit operation; the data cannot silently narrow it."""
    if not isinstance(observation, ExitDepthObservation) or not isinstance(config, DataQualityConfig):
        raise TypeError("exit observation and original quality configuration required")
    if (not isinstance(markets, tuple) or not markets or len(set(markets)) != len(markets)
            or any(not isinstance(market, Market) for market in markets)):
        raise ValueError("explicit unique exit markets required")
    at = utc_timestamp(evaluated_at)
    checks = _Checks(observation.symbol, at, config, None)
    try:
        require_market_symbol(observation.symbol, None)
    except (TypeError, ValueError):
        checks.add(R.SYMBOL_MISMATCH, "exit", "symbol")
    if observation.markets != markets:
        checks.add(R.SOURCE_MISMATCH, "exit", "markets")
    collected = checks.timestamp(observation.collected_at, "exit", "collected_at")
    received: dict[Market, datetime] = {}
    for market in markets:
        spot = market is Market.SPOT
        name = "spot_depth" if spot else "perp_depth"
        source = observation.spot_depth if spot else observation.perp_depth
        endpoint = PublicEndpoint.SPOT_DEPTH if spot else PublicEndpoint.PERP_DEPTH
        age = R.STALE_SPOT_DATA if spot else R.STALE_PERP_DATA
        expected_type = SpotDepthResult if spot else UsdtmFetchResult
        if not isinstance(source, expected_type) or not getattr(source, "ok", False):
            checks.add(R.SOURCE_ERROR, name, "result")
        request = getattr(source, "request", None)
        if not isinstance(request, TestnetGetRequest) or request.endpoint is not endpoint:
            checks.add(R.SOURCE_MISMATCH, name, "request")
        else:
            checks.identity(request, name + ".request")
            checks.symbol_check(dict(request.params).get("symbol"), name + ".request")
        if (getattr(source, "status_code", None) != 200 or getattr(source, "body_truncated", False)
                or getattr(source, "error", None) is not None):
            checks.add(R.SOURCE_ERROR, name, "response")
        times = {field: checks.timestamp(getattr(source, field, None), name, field, age_reason=age)
                 for field in ("started_at", "received_at", "finished_at")}
        start, receive, finish = (times[field] for field in ("started_at", "received_at", "finished_at"))
        if receive is not None:
            received[market] = receive
        if start is not None and receive is not None and finish is not None:
            if not start <= receive <= finish or collected is not None and finish > collected:
                checks.add(R.INVALID_TIMESTAMP, name, "request_window")
        data = getattr(source, "order_book" if spot else "data", None)
        if not isinstance(data, OrderBook if spot else PerpDepth):
            checks.add(R.SOURCE_ERROR, name, "data")
        book = data if spot else getattr(data, "order_book", None)
        checks.book(book, name, market)
        observed = checks.timestamp(getattr(book, "observed_at", None), name, "observed_at", age_reason=age)
        if observed is not None and receive is not None and observed != receive:
            checks.add(R.SOURCE_MISMATCH, name, "observed_at")
        checks.timestamp(getattr(book, "event_time", None), name, "event_time",
                         age_reason=age, required=not spot)
        if not spot:
            checks.timestamp(getattr(data, "transaction_time", None), name, "transaction_time", age_reason=age)
    skew = None
    if Market.SPOT in received and Market.USDTM in received:
        skew = abs(received[Market.USDTM] - received[Market.SPOT])
        if skew > config.max_sampling_skew:
            checks.add(R.SAMPLING_SKEW_TOO_LARGE, "exit", "depth_received_at")
    return DataQualityResult(at, config, tuple(checks.issues), skew)
