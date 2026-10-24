"""Pure snapshot assembly and one explicit collection through existing adapters.

No scheduling, cache, profit calculation, freshness thresholds or decisions.
"""

from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get, prepare_research_get
from crypto_arbitrage_bot.models.market import (
    Market,
    MarketSnapshot,
    OrderBook,
    Timestamp,
    require_market_symbol,
)
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.snapshot import (
    SnapshotAssembly,
    SnapshotIssue,
    SnapshotIssueKind,
    SnapshotSource,
    SnapshotSources,
)
from crypto_arbitrage_bot.models.spot import SpotDepthResult
from crypto_arbitrage_bot.models.usdtm import (
    OpenInterest,
    PerpDepth,
    PremiumIndex,
    UsdtmFetchResult,
)
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

type _SourceResult = (
    SpotDepthResult
    | UsdtmFetchResult[PerpDepth]
    | UsdtmFetchResult[PremiumIndex]
    | UsdtmFetchResult[OpenInterest]
)


def _check_source(
    result: _SourceResult | None,
    source: SnapshotSource,
    endpoint: PublicEndpoint,
    symbol: str,
    research_instrument: ResearchInstrument | None,
) -> SnapshotIssue | None:
    """Check structural completeness and identity only, without data-quality policy."""
    if result is None:
        return SnapshotIssue(source, SnapshotIssueKind.MISSING)
    if not isinstance(result, (SpotDepthResult, UsdtmFetchResult)):
        return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
    if result.request.endpoint is not endpoint:
        return SnapshotIssue(source, SnapshotIssueKind.ENDPOINT_MISMATCH)
    if dict(result.request.params).get("symbol") != symbol:
        return SnapshotIssue(source, SnapshotIssueKind.SYMBOL_MISMATCH)
    if result.request.research_instrument != research_instrument:
        return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
    if not result.ok:
        return SnapshotIssue(source, SnapshotIssueKind.SOURCE_ERROR)
    if result.received_at is None:
        return SnapshotIssue(source, SnapshotIssueKind.MISSING_RECEIVE_TIME)
    try:
        utc_timestamp(result.received_at)
    except (TypeError, ValueError):
        return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)

    book: OrderBook | None = None
    if source is SnapshotSource.SPOT_DEPTH:
        if not isinstance(result, SpotDepthResult) or not isinstance(result.order_book, OrderBook):
            return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
        book = result.order_book
        if book.market is not Market.SPOT:
            return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
        data_symbol = book.symbol
    else:
        if not isinstance(result, UsdtmFetchResult):
            return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
        if source is SnapshotSource.PERP_DEPTH and isinstance(result.data, PerpDepth):
            book = result.data.order_book
            if book.market is not Market.USDTM:
                return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
            data_symbol = book.symbol
        elif source is SnapshotSource.PREMIUM_INDEX and isinstance(result.data, PremiumIndex):
            data_symbol = result.data.symbol
        elif source is SnapshotSource.OPEN_INTEREST and isinstance(result.data, OpenInterest):
            data_symbol = result.data.symbol
        else:
            return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
    data_identity = (book.research_instrument if book is not None else
                     getattr(getattr(result, "data", None), "research_instrument", None))
    if data_identity != research_instrument:
        return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
    if data_symbol != symbol:
        return SnapshotIssue(source, SnapshotIssueKind.SYMBOL_MISMATCH)
    if book is not None and (not book.bids or not book.asks):
        return SnapshotIssue(source, SnapshotIssueKind.INVALID_DATA)
    return None


def assemble_snapshot(
    symbol: str, sources: SnapshotSources, *, assembled_at: Timestamp,
    research_instrument: ResearchInstrument | None = None,
) -> SnapshotAssembly:
    """Pure composition: never fetch, replace missing data, or convert to float.

    A complete inner MarketSnapshot requires four successful matching sources.
    Incomplete attempts retain all supplied source results for diagnosis. The
    depth receive-time delta can still be recorded if only premium/OI is absent.
    """
    require_market_symbol(symbol, research_instrument)
    if not isinstance(sources, SnapshotSources):
        raise TypeError("sources must be SnapshotSources")
    assembled_at = utc_timestamp(assembled_at)
    entries = (
        (sources.spot_depth, SnapshotSource.SPOT_DEPTH, PublicEndpoint.SPOT_DEPTH),
        (sources.perp_depth, SnapshotSource.PERP_DEPTH, PublicEndpoint.PERP_DEPTH),
        (sources.premium_index, SnapshotSource.PREMIUM_INDEX, PublicEndpoint.PERP_PREMIUM_INDEX),
        (sources.open_interest, SnapshotSource.OPEN_INTEREST, PublicEndpoint.PERP_OPEN_INTEREST),
    )
    issues = tuple(
        issue for result, source, endpoint in entries
        if (issue := _check_source(result, source, endpoint, symbol, research_instrument)) is not None
    )
    failed_sources = {issue.source for issue in issues}
    spot = sources.spot_depth
    perp = sources.perp_depth
    sample_delta = None
    if not failed_sources.intersection({SnapshotSource.SPOT_DEPTH, SnapshotSource.PERP_DEPTH}):
        assert spot is not None and spot.received_at is not None
        assert perp is not None and perp.received_at is not None
        sample_delta = utc_timestamp(perp.received_at) - utc_timestamp(spot.received_at)

    snapshot = None
    if not issues:
        premium = sources.premium_index
        interest = sources.open_interest
        assert spot is not None and spot.order_book is not None
        assert perp is not None and perp.data is not None
        assert premium is not None and premium.data is not None
        assert interest is not None and interest.data is not None
        snapshot = MarketSnapshot(
            symbol=symbol, spot_book=spot.order_book, perp_book=perp.data.order_book,
            mark_price=premium.data.mark_price, index_price=premium.data.index_price,
            funding_rate=premium.data.last_funding_rate,
            next_funding_time=premium.data.next_funding_time,
            open_interest=interest.data.open_interest, observed_at=assembled_at,
            research_instrument=research_instrument,
        )
    return SnapshotAssembly(
        symbol=symbol, assembled_at=assembled_at, sources=sources, snapshot=snapshot,
        issues=issues, spot_perp_sample_delta=sample_delta, research_instrument=research_instrument,
    )


class SnapshotCollector:
    """Collect once for one symbol, using the existing read-only adapters.

    Construction does no IO. collect() requests all four sources once, even if
    one returns a typed failure. Nothing is cached or retried between calls.
    """

    def __init__(self, spot: BinanceSpotClient, perp: BinanceUsdtmClient,
                 *, research_instrument: ResearchInstrument | None = None) -> None:
        if research_instrument is not None:
            if not isinstance(research_instrument, ResearchInstrument):
                raise TypeError("research_instrument must be ResearchInstrument")
            if spot.research_instrument != research_instrument or perp.research_instrument != research_instrument:
                raise ValueError("adapter research identities must match collector")
        self.research_instrument = research_instrument
        self._spot = spot
        self._perp = perp

    def collect(
        self, symbol: str, *, spot_limit: int = 100, perp_limit: int = 100,
    ) -> SnapshotAssembly:
        # Validate both depth limits before even the first public GET can start.
        for endpoint, limit in ((PublicEndpoint.SPOT_DEPTH, spot_limit), (PublicEndpoint.PERP_DEPTH, perp_limit)):
            if self.research_instrument is None:
                prepare_get(endpoint, {"symbol": symbol, "limit": limit})
            else:
                prepare_research_get(endpoint, self.research_instrument, {"symbol": symbol, "limit": limit})
        sources = SnapshotSources(
            spot_depth=self._spot.fetch_order_book(symbol, limit=spot_limit),
            perp_depth=self._perp.fetch_order_book(symbol, limit=perp_limit),
            premium_index=self._perp.fetch_premium_index(symbol),
            open_interest=self._perp.fetch_open_interest(symbol),
        )
        return assemble_snapshot(symbol, sources, assembled_at=utc_now(), research_instrument=self.research_instrument)
