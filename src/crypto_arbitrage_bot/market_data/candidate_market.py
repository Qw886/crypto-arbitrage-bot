"""Catalog-qualified public observations; no prediction, profit or trade authority."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from crypto_arbitrage_bot.config.research import CandidateCatalogSettings
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.funding_history import (
    BinanceFundingHistoryClient,
    FundingWindow,
)
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CandidateCatalog,
    CatalogStatus,
    ResearchCandidate,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.market_data.snapshot import SnapshotCollector
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig, DataQualityResult
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


@dataclass(frozen=True)
class CandidateObservation:
    candidate: ResearchCandidate
    assembly: SnapshotAssembly
    data_quality: DataQualityResult


class CandidateMarketCollector:
    """Reuses adapters with explicit identity; no cache of market observations.

    Catalog membership is re-derived from retained official metadata at
    construction, rather than trusting caller-edited candidates. Every read
    checks local metadata age and contract lifetime again. A metadata candidate
    is still not proof of order-book quality or profitability.
    """

    def __init__(self, catalog: CandidateCatalog, criteria: CandidateCatalogSettings,
                 quality: DataQualityConfig, *, timeout_seconds: int,
                 funding_page_limit: int, funding_max_pages: int) -> None:
        if not isinstance(catalog, CandidateCatalog) or not isinstance(criteria, CandidateCatalogSettings):
            raise TypeError("catalog and its criteria are required")
        if not isinstance(quality, DataQualityConfig):
            raise TypeError("quality must be DataQualityConfig")
        derived = build_candidate_catalog(*catalog.sources, criteria, now=utc_now())
        if (catalog.status is not CatalogStatus.COMPLETE or derived.status is not CatalogStatus.COMPLETE
                or catalog.candidates != derived.candidates):
            raise ValueError("candidate catalog is invalid, stale or inconsistent with its sources")
        # Validate the existing bounded transport settings before any IO.
        BinanceSpotClient(timeout_seconds)
        BinanceFundingHistoryClient(timeout_seconds=timeout_seconds,
                                    page_limit=funding_page_limit, max_pages=funding_max_pages)
        self._catalog = derived
        self._criteria = criteria
        self._quality = quality
        self._timeout = timeout_seconds
        self._page_limit = funding_page_limit
        self._max_pages = funding_max_pages
        self._candidates = {item.symbol: item for item in derived.candidates}

    def _candidate(self, symbol: str) -> tuple[ResearchCandidate, ResearchInstrument]:
        if type(symbol) is not str or symbol not in self._candidates:
            raise ValueError("symbol is not an admitted research candidate")
        at = utc_now()
        for source in self._catalog.sources:
            if (source.finished_at > at or not timedelta(0) <= at - source.requested_at
                    <= timedelta(seconds=self._criteria.max_metadata_age_seconds)):
                raise ValueError("candidate metadata expired; collect a new catalog")
        candidate = self._candidates[symbol]
        if candidate.perp_delivery_at - at < timedelta(seconds=self._criteria.min_remaining_lifetime_seconds):
            raise ValueError("candidate remaining contract lifetime no longer meets criteria")
        return candidate, ResearchInstrument(candidate.symbol, candidate.base_asset)

    def collect(self, symbol: str, *, spot_limit: int = 100,
                perp_limit: int = 100) -> CandidateObservation:
        candidate, identity = self._candidate(symbol)
        spot = BinanceSpotClient(self._timeout, research_instrument=identity)
        perp = BinanceUsdtmClient(self._timeout, research_instrument=identity)
        assembly = SnapshotCollector(spot, perp, research_instrument=identity).collect(
            symbol, spot_limit=spot_limit, perp_limit=perp_limit)
        quality = validate_market_data(assembly, self._quality, evaluated_at=utc_now())
        return CandidateObservation(candidate, assembly, quality)

    def fetch_funding_window(self, symbol: str, opened_at: datetime,
                             closed_at: datetime) -> FundingWindow:
        _, identity = self._candidate(symbol)
        opened, closed = utc_timestamp(opened_at), utc_timestamp(closed_at)
        if closed > utc_now():
            raise ValueError("historical funding window must not include future settlements")
        return BinanceFundingHistoryClient(
            timeout_seconds=self._timeout, page_limit=self._page_limit,
            max_pages=self._max_pages, research_instrument=identity,
        ).fetch_window(symbol, opened, closed)
