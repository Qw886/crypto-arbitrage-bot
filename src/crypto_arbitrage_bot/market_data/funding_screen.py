"""Funding preselection of qualified research assets; no PnL or Decision shortcuts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from crypto_arbitrage_bot.config.research import CandidateCatalogSettings, FundingScreenSettings
from crypto_arbitrage_bot.exchanges.binance_usdtm import _parse_premium
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import PremiumCatalogGetRequest
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CandidateCatalog,
    CatalogStatus,
    ResearchCandidate,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.usdtm import PremiumIndex, UsdtmFetchResult
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class ScreenStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"


class ScreenReason(StrEnum):
    INVALID_METADATA = "INVALID_METADATA"
    PREMIUM_SOURCE_FAILED = "PREMIUM_SOURCE_FAILED"
    INVALID_SOURCE_IDENTITY = "INVALID_SOURCE_IDENTITY"
    INVALID_SOURCE_TIME = "INVALID_SOURCE_TIME"
    STALE_PREMIUM_SOURCE = "STALE_PREMIUM_SOURCE"
    MISSING_PREMIUM = "MISSING_PREMIUM"
    INVALID_PREMIUM = "INVALID_PREMIUM"
    STALE_PREMIUM = "STALE_PREMIUM"
    FUTURE_PREMIUM = "FUTURE_PREMIUM"
    NEXT_SETTLEMENT_OUTSIDE_WINDOW = "NEXT_SETTLEMENT_OUTSIDE_WINDOW"
    FUNDING_BELOW_SCREEN_THRESHOLD = "FUNDING_BELOW_SCREEN_THRESHOLD"
    BENCHMARK_UNAVAILABLE = "BENCHMARK_UNAVAILABLE"
    OUTSIDE_COLLECTION_BUDGET = "OUTSIDE_COLLECTION_BUDGET"


@dataclass(frozen=True)
class ScreenExclusion:
    symbol: str
    reason_codes: tuple[ScreenReason, ...]


@dataclass(frozen=True)
class FundingCandidate:
    candidate: ResearchCandidate
    premium: PremiumIndex
    benchmark: bool


@dataclass(frozen=True)
class FundingScreen:
    evaluated_at: datetime
    status: ScreenStatus
    selected: tuple[FundingCandidate, ...]
    exclusions: tuple[ScreenExclusion, ...]
    reason_codes: tuple[ScreenReason, ...]
    qualified_count: int


def screen_candidates(
    catalog: CandidateCatalog, criteria: CandidateCatalogSettings,
    source: UsdtmFetchResult[tuple[dict[str, object], ...]],
    settings: FundingScreenSettings, quality: DataQualityConfig, *, evaluated_at: datetime,
) -> FundingScreen:
    """Higher observed signed funding ranks reads, never authorizes an entry.

    Current funding is not a settlement. Selected assets still require a fresh
    full snapshot and the original depth/fees/scenario/Decision engine. Both
    benchmark assets stay present even with zero/negative observed funding.
    """
    at = utc_timestamp(evaluated_at)
    reasons: list[ScreenReason] = []
    derived = build_candidate_catalog(*catalog.sources, criteria, now=at)
    if (catalog.status is not CatalogStatus.COMPLETE or derived.status is not CatalogStatus.COMPLETE
            or catalog.candidates != derived.candidates):
        reasons.append(ScreenReason.INVALID_METADATA)
    if not source.ok or not isinstance(source.data, tuple) or not source.data:
        reasons.append(ScreenReason.PREMIUM_SOURCE_FAILED)
    if (not isinstance(source.request, PremiumCatalogGetRequest)
            or source.request.endpoint is not PublicEndpoint.PERP_PREMIUM_INDEX
            or source.request.params or source.request.research_instrument is not None):
        reasons.append(ScreenReason.INVALID_SOURCE_IDENTITY)
    try:
        requested, finished = utc_timestamp(source.started_at), utc_timestamp(source.finished_at)
        received = None if source.received_at is None else utc_timestamp(source.received_at)
        if received is None or not requested <= received <= finished <= at:
            reasons.append(ScreenReason.INVALID_SOURCE_TIME)
        elif at - requested > quality.max_data_age or at - received > quality.max_data_age:
            reasons.append(ScreenReason.STALE_PREMIUM_SOURCE)
    except (ValueError, TypeError):
        received = None
        reasons.append(ScreenReason.INVALID_SOURCE_TIME)
    if reasons:
        return FundingScreen(at, ScreenStatus.INVALID, (), (), tuple(dict.fromkeys(reasons)), len(derived.candidates))
    assert source.data is not None and received is not None
    if any(not isinstance(row, dict) or type(row.get("symbol")) is not str or not row["symbol"]
           for row in source.data):
        return FundingScreen(at, ScreenStatus.INVALID, (), (), (ScreenReason.PREMIUM_SOURCE_FAILED,), len(derived.candidates))
    rows = {row["symbol"]: row for row in source.data}
    if len(rows) != len(source.data):
        return FundingScreen(at, ScreenStatus.INVALID, (), (), (ScreenReason.PREMIUM_SOURCE_FAILED,), len(derived.candidates))
    selected: list[FundingCandidate] = []
    ranked: list[FundingCandidate] = []
    excluded: list[ScreenExclusion] = []
    for candidate in derived.candidates:
        symbol = candidate.symbol
        benchmark = symbol in settings.benchmark_symbols
        row = rows.get(symbol)
        failure: ScreenReason | None = None
        if row is None:
            failure = ScreenReason.MISSING_PREMIUM
        else:
            try:
                premium = _parse_premium(row, symbol, received,
                    research_instrument=ResearchInstrument(symbol, candidate.base_asset))
                if premium.event_time > at:
                    failure = ScreenReason.FUTURE_PREMIUM
                elif at - premium.event_time > quality.max_data_age:
                    failure = ScreenReason.STALE_PREMIUM
                elif not at < premium.next_funding_time <= at + timedelta(seconds=settings.max_next_settlement_wait_seconds):
                    failure = ScreenReason.NEXT_SETTLEMENT_OUTSIDE_WINDOW
                elif not benchmark and premium.last_funding_rate <= settings.min_observed_funding_rate:
                    failure = ScreenReason.FUNDING_BELOW_SCREEN_THRESHOLD
                else:
                    (selected if benchmark else ranked).append(FundingCandidate(candidate, premium, benchmark))
            except (ValueError, TypeError, ArithmeticError):
                failure = ScreenReason.INVALID_PREMIUM
        if failure is not None:
            excluded.append(ScreenExclusion(symbol, (failure,)))
    if {item.candidate.symbol for item in selected} != set(settings.benchmark_symbols):
        return FundingScreen(at, ScreenStatus.INVALID, (), tuple(excluded),
                             (ScreenReason.BENCHMARK_UNAVAILABLE,), len(derived.candidates))
    # Stable ties use symbol, without converting the funding Decimal to float.
    selected.sort(key=lambda item: settings.benchmark_symbols.index(item.candidate.symbol))
    ranked.sort(key=lambda item: (item.premium.last_funding_rate.copy_negate(), item.candidate.symbol))
    room = settings.max_candidates - len(selected)
    excluded.extend(ScreenExclusion(item.candidate.symbol, (ScreenReason.OUTSIDE_COLLECTION_BUDGET,))
                    for item in ranked[room:])
    return FundingScreen(at, ScreenStatus.COMPLETE, tuple(selected + ranked[:room]), tuple(excluded), (),
                         len(derived.candidates))
