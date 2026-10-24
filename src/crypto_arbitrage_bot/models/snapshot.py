"""A complete market snapshot or an explicitly incomplete assembly attempt."""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

from crypto_arbitrage_bot.models.market import MarketSnapshot, Timestamp
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.spot import SpotDepthResult
from crypto_arbitrage_bot.models.usdtm import (
    OpenInterest,
    PerpDepth,
    PremiumIndex,
    UsdtmFetchResult,
)


class SnapshotStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class SnapshotSource(StrEnum):
    SPOT_DEPTH = "SPOT_DEPTH"
    PERP_DEPTH = "PERP_DEPTH"
    PREMIUM_INDEX = "PREMIUM_INDEX"
    OPEN_INTEREST = "OPEN_INTEREST"


class SnapshotIssueKind(StrEnum):
    MISSING = "MISSING"
    SOURCE_ERROR = "SOURCE_ERROR"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    ENDPOINT_MISMATCH = "ENDPOINT_MISMATCH"
    MISSING_RECEIVE_TIME = "MISSING_RECEIVE_TIME"
    INVALID_DATA = "INVALID_DATA"


@dataclass(frozen=True)
class SnapshotIssue:
    source: SnapshotSource
    kind: SnapshotIssueKind


@dataclass(frozen=True)
class SnapshotSources:
    """Keep each original result, including its raw bytes, times and any error.

    None means a source was not supplied. A failed result remains available for
    inspection and is never replaced by cached data or zero-valued fields.
    """

    spot_depth: SpotDepthResult | None = None
    perp_depth: UsdtmFetchResult[PerpDepth] | None = None
    premium_index: UsdtmFetchResult[PremiumIndex] | None = None
    open_interest: UsdtmFetchResult[OpenInterest] | None = None


@dataclass(frozen=True)
class SnapshotAssembly:
    """Provenance-preserving result; completeness does not imply data freshness.

    assembled_at (also snapshot.observed_at when complete) is local assembly
    time, never a synthetic exchange event time. Each source retains its own
    receive time and official times. spot_perp_sample_delta compares successful,
    matching depth receive times: perpetual minus spot, exactly as a timedelta.
    """

    symbol: str
    assembled_at: Timestamp
    sources: SnapshotSources
    snapshot: MarketSnapshot | None
    issues: tuple[SnapshotIssue, ...]
    spot_perp_sample_delta: timedelta | None
    research_instrument: ResearchInstrument | None = field(default=None, kw_only=True)

    @property
    def complete(self) -> bool:
        return self.snapshot is not None and not self.issues

    @property
    def status(self) -> SnapshotStatus:
        return SnapshotStatus.COMPLETE if self.complete else SnapshotStatus.INCOMPLETE

    @property
    def spot_perp_sample_gap(self) -> timedelta | None:
        return None if self.spot_perp_sample_delta is None else abs(self.spot_perp_sample_delta)
