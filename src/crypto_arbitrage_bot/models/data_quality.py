"""Configuration and structured diagnostics for market data quality only."""

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from crypto_arbitrage_bot.models.market import Timestamp


class OpenInterestTimeSemantics(StrEnum):
    EXCHANGE_EVENT = "exchange_event"
    LAST_TRANSACTION = "last_transaction"


@dataclass(frozen=True)
class DataQualityConfig:
    max_data_age: timedelta = timedelta(seconds=10)
    max_sampling_skew: timedelta = timedelta(seconds=2)
    open_interest_time_semantics: OpenInterestTimeSemantics = OpenInterestTimeSemantics.EXCHANGE_EVENT

    def __post_init__(self) -> None:
        if not isinstance(self.open_interest_time_semantics, OpenInterestTimeSemantics):
            raise TypeError("open_interest_time_semantics must be an OpenInterestTimeSemantics")
        for name in ("max_data_age", "max_sampling_skew"):
            value = getattr(self, name)
            if not isinstance(value, timedelta):
                raise TypeError(f"{name} must be timedelta")
            if value < timedelta(0):
                raise ValueError(f"{name} must be nonnegative")


class DataQualityStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"


class DataQualityReason(StrEnum):
    INCOMPLETE_SNAPSHOT = "INCOMPLETE_SNAPSHOT"
    MISSING_SPOT_DEPTH = "MISSING_SPOT_DEPTH"
    MISSING_PERP_DEPTH = "MISSING_PERP_DEPTH"
    MISSING_SPOT_BID = "MISSING_SPOT_BID"
    MISSING_SPOT_ASK = "MISSING_SPOT_ASK"
    MISSING_PERP_BID = "MISSING_PERP_BID"
    MISSING_PERP_ASK = "MISSING_PERP_ASK"
    MISSING_MARK_PRICE = "MISSING_MARK_PRICE"
    MISSING_INDEX_PRICE = "MISSING_INDEX_PRICE"
    MISSING_FUNDING = "MISSING_FUNDING"
    MISSING_NEXT_FUNDING_TIME = "MISSING_NEXT_FUNDING_TIME"
    MISSING_OPEN_INTEREST = "MISSING_OPEN_INTEREST"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    CROSSED_SPOT_BOOK = "CROSSED_SPOT_BOOK"
    CROSSED_PERP_BOOK = "CROSSED_PERP_BOOK"
    UNSORTED_BOOK = "UNSORTED_BOOK"
    INVALID_FUNDING = "INVALID_FUNDING"
    INVALID_NEXT_FUNDING_TIME = "INVALID_NEXT_FUNDING_TIME"
    INVALID_OPEN_INTEREST = "INVALID_OPEN_INTEREST"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    SOURCE_ERROR = "SOURCE_ERROR"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"
    STALE_SPOT_DATA = "STALE_SPOT_DATA"
    STALE_PERP_DATA = "STALE_PERP_DATA"
    STALE_PREMIUM_DATA = "STALE_PREMIUM_DATA"
    STALE_OPEN_INTEREST_DATA = "STALE_OPEN_INTEREST_DATA"
    SAMPLING_SKEW_TOO_LARGE = "SAMPLING_SKEW_TOO_LARGE"


@dataclass(frozen=True)
class DataQualityIssue:
    code: DataQualityReason
    source: str
    field: str


@dataclass(frozen=True)
class DataQualityResult:
    evaluated_at: Timestamp
    config: DataQualityConfig
    issues: tuple[DataQualityIssue, ...]
    sampling_skew: timedelta | None

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def status(self) -> DataQualityStatus:
        return DataQualityStatus.VALID if self.valid else DataQualityStatus.INVALID

    @property
    def reason_codes(self) -> tuple[DataQualityReason, ...]:
        return tuple(dict.fromkeys(issue.code for issue in self.issues))
