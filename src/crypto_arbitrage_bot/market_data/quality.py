"""Pure, accumulating validation of all four source observations and their assembly.

Does not fetch, repair, fill missing values, change timestamps, or compute PnL.
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation

from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import PublicGetRequest
from crypto_arbitrage_bot.models.data_quality import (
    DataQualityConfig,
    DataQualityIssue,
    DataQualityReason,
    DataQualityResult,
    OpenInterestTimeSemantics,
)
from crypto_arbitrage_bot.models.market import (
    Market,
    MarketSnapshot,
    OrderBook,
    require_market_symbol,
)
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.models.spot import SpotDepthResult
from crypto_arbitrage_bot.models.usdtm import (
    OpenInterest,
    PerpDepth,
    PremiumIndex,
    UsdtmFetchResult,
)
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

R = DataQualityReason
_DEFAULT_CONFIG = DataQualityConfig()


class _Checks:
    def __init__(self, symbol: str, at: datetime, config: DataQualityConfig,
                 research_instrument: ResearchInstrument | None) -> None:
        self.research_instrument = research_instrument
        self.symbol = symbol
        self.at = at
        self.config = config
        self.issues: list[DataQualityIssue] = []

    def add(self, code: DataQualityReason, source: str, field: str) -> None:
        issue = DataQualityIssue(code, source, field)
        if issue not in self.issues:
            self.issues.append(issue)

    def identity(self, value: object, source: str) -> None:
        if value is not None and getattr(value, "research_instrument", None) != self.research_instrument:
            self.add(R.SOURCE_MISMATCH, source, "research_instrument")

    def symbol_check(self, value: object, source: str) -> None:
        if value != self.symbol:
            self.add(R.SYMBOL_MISMATCH, source, "symbol")

    def number(
        self, value: object, source: str, field: str,
        missing: DataQualityReason, invalid: DataQualityReason,
        *, positive: bool = False, nonnegative: bool = False,
    ) -> bool:
        if value is None:
            self.add(missing, source, field)
            return False
        if not isinstance(value, Decimal) or not value.is_finite():
            self.add(invalid, source, field)
            return False
        if (positive and value <= 0) or (nonnegative and value < 0):
            self.add(invalid, source, field)
            return False
        return True

    def timestamp(
        self, value: object, source: str, field: str,
        *, age_reason: DataQualityReason | None = None, required: bool = True,
    ) -> datetime | None:
        if value is None:
            if required:
                self.add(R.MISSING_TIMESTAMP, source, field)
            return None
        try:
            if not isinstance(value, datetime):
                raise TypeError("timestamp must be datetime")
            at = utc_timestamp(value)
        except (TypeError, ValueError, OverflowError):
            self.add(R.INVALID_TIMESTAMP, source, field)
            return None
        if at > self.at:
            self.add(R.FUTURE_TIMESTAMP, source, field)
        elif age_reason is not None and self.at - at > self.config.max_data_age:
            self.add(age_reason, source, field)
        return at

    def book(self, book: object, source: str, market: Market) -> None:
        self.identity(book, source)
        spot = market is Market.SPOT
        missing_depth = R.MISSING_SPOT_DEPTH if spot else R.MISSING_PERP_DEPTH
        if not isinstance(book, OrderBook):
            self.add(missing_depth, source, "depth")
        self.symbol_check(getattr(book, "symbol", None), source)
        if book is not None and getattr(book, "market", None) is not market:
            self.add(R.SOURCE_MISMATCH, source, "market")
        best: dict[str, Decimal] = {}
        for side, missing_quote in (
            ("bids", R.MISSING_SPOT_BID if spot else R.MISSING_PERP_BID),
            ("asks", R.MISSING_SPOT_ASK if spot else R.MISSING_PERP_ASK),
        ):
            levels = getattr(book, side, None)
            if not isinstance(levels, tuple) or not levels:
                self.add(missing_depth, source, side)
                self.add(missing_quote, source, side)
                continue
            previous: Decimal | None = None
            for index, level in enumerate(levels):
                price = getattr(level, "price", None)
                good_price = self.number(
                    price, source, f"{side}[{index}].price", R.INVALID_PRICE, R.INVALID_PRICE,
                    positive=True,
                )
                self.number(
                    getattr(level, "quantity", None), source, f"{side}[{index}].quantity",
                    R.INVALID_QUANTITY, R.INVALID_QUANTITY, positive=True,
                )
                if index == 0 and price is None:
                    self.add(missing_quote, source, side)
                if good_price and isinstance(price, Decimal):
                    if index == 0:
                        best[side] = price
                    if previous is not None and (
                        (side == "bids" and price >= previous)
                        or (side == "asks" and price <= previous)
                    ):
                        self.add(R.UNSORTED_BOOK, source, side)
                    previous = price
        if "bids" in best and "asks" in best and best["bids"] > best["asks"]:
            self.add(R.CROSSED_SPOT_BOOK if spot else R.CROSSED_PERP_BOOK, source, "bid_ask")

    def premium(self, data: object, source: str, *, unified: bool = False) -> None:
        self.number(getattr(data, "mark_price", None), source, "mark_price",
                    R.MISSING_MARK_PRICE, R.INVALID_PRICE, positive=True)
        self.number(getattr(data, "index_price", None), source, "index_price",
                    R.MISSING_INDEX_PRICE, R.INVALID_PRICE, positive=True)
        funding_name = "funding_rate" if unified else "last_funding_rate"
        self.number(getattr(data, funding_name, None), source, funding_name,
                    R.MISSING_FUNDING, R.INVALID_FUNDING)
        next_time = getattr(data, "next_funding_time", None)
        if next_time is None:
            self.add(R.MISSING_NEXT_FUNDING_TIME, source, "next_funding_time")
        else:
            try:
                if not isinstance(next_time, datetime):
                    raise TypeError("next funding time must be datetime")
                next_time = utc_timestamp(next_time)
                event_time = getattr(data, "event_time", None)
                if next_time <= self.at or (
                    isinstance(event_time, datetime) and next_time <= utc_timestamp(event_time)
                ):
                    raise ValueError("next funding time must be in the future")
            except (TypeError, ValueError, OverflowError):
                self.add(R.INVALID_NEXT_FUNDING_TIME, source, "next_funding_time")

    def interest(self, data: object, source: str) -> None:
        self.number(getattr(data, "open_interest", None), source, "open_interest",
                    R.MISSING_OPEN_INTEREST, R.INVALID_OPEN_INTEREST, nonnegative=True)


def _same(left: object, right: object) -> bool:
    if left is right:
        return True
    try:
        return left == right
    except (TypeError, ValueError, InvalidOperation):
        return False


def validate_market_data(
    assembly: SnapshotAssembly, config: DataQualityConfig = _DEFAULT_CONFIG,
    *, evaluated_at: datetime,
) -> DataQualityResult:
    """Validate against an explicit UTC evaluation time; thresholds are inclusive.

    Age uses request, receive and observation/exchange timestamps. Explicit OI
    last-transaction semantics retain/check the official time without treating
    an unchanged transaction timestamp as the publication time of a fresh GET.
    Spot's absent exchange time remains None. USDT-M exchange times are required.
    Sampling skew is recomputed from the two depth receive times, not cached
    assembly metadata. All fields and sources are examined even after an error.
    """
    if not isinstance(assembly, SnapshotAssembly) or not isinstance(config, DataQualityConfig):
        raise TypeError("expected SnapshotAssembly and DataQualityConfig")
    at = utc_timestamp(evaluated_at, "evaluated_at")
    checks = _Checks(assembly.symbol, at, config, assembly.research_instrument)
    try:
        require_market_symbol(assembly.symbol, assembly.research_instrument)
    except (TypeError, ValueError):
        checks.add(R.SYMBOL_MISMATCH, "snapshot", "symbol")
    if not assembly.complete:
        checks.add(R.INCOMPLETE_SNAPSHOT, "snapshot", "status")
    checks.timestamp(assembly.assembled_at, "snapshot", "assembled_at")
    records = (
        ("spot_depth", PublicEndpoint.SPOT_DEPTH, R.STALE_SPOT_DATA),
        ("perp_depth", PublicEndpoint.PERP_DEPTH, R.STALE_PERP_DATA),
        ("premium_index", PublicEndpoint.PERP_PREMIUM_INDEX, R.STALE_PREMIUM_DATA),
        ("open_interest", PublicEndpoint.PERP_OPEN_INTEREST, R.STALE_OPEN_INTEREST_DATA),
    )
    received: dict[str, datetime] = {}
    data_items: dict[str, object] = {}
    for name, endpoint, age_reason in records:
        record = getattr(assembly.sources, name, None)
        if not isinstance(record, (SpotDepthResult, UsdtmFetchResult)):
            checks.add(R.SOURCE_ERROR, name, "result")
        request = getattr(record, "request", None)
        if not isinstance(request, PublicGetRequest):
            checks.add(R.SOURCE_ERROR, name, "request")
        else:
            checks.identity(request, name + ".request")
            checks.symbol_check(dict(request.params).get("symbol"), name + ".request")
            if request.endpoint is not endpoint:
                checks.add(R.SOURCE_MISMATCH, name, "endpoint")
        if record is not None and (
            getattr(record, "error", None) is not None or not getattr(record, "ok", False)
        ):
            checks.add(R.SOURCE_ERROR, name, "result")
        times = {
            field: checks.timestamp(getattr(record, field, None), name, field, age_reason=age_reason)
            for field in ("started_at", "received_at", "finished_at")
        }
        start, receive, finish = (times[field] for field in ("started_at", "received_at", "finished_at"))
        if receive is not None:
            received[name] = receive
        if start is not None and receive is not None and finish is not None:
            if not start <= receive <= finish:
                checks.add(R.INVALID_TIMESTAMP, name, "request_window")

        data = getattr(record, "order_book" if name == "spot_depth" else "data", None)
        expected_type = {
            "spot_depth": OrderBook, "perp_depth": PerpDepth,
            "premium_index": PremiumIndex, "open_interest": OpenInterest,
        }[name]
        if not isinstance(data, expected_type):
            checks.add(R.SOURCE_ERROR, name, "data")
        data_items[name] = data
        if name in ("spot_depth", "perp_depth"):
            book = data if name == "spot_depth" else getattr(data, "order_book", None)
            checks.book(book, name, Market.SPOT if name == "spot_depth" else Market.USDTM)
            checks.timestamp(getattr(book, "observed_at", None), name, "observed_at", age_reason=age_reason)
            checks.timestamp(getattr(book, "event_time", None), name, "event_time",
                             age_reason=age_reason, required=name == "perp_depth")
            if name == "perp_depth":
                checks.timestamp(getattr(data, "transaction_time", None), name, "transaction_time",
                                 age_reason=age_reason)
        else:
            checks.identity(data, name)
            checks.symbol_check(getattr(data, "symbol", None), name)
            event_age_reason = (None if name == "open_interest" and
                                config.open_interest_time_semantics is OpenInterestTimeSemantics.LAST_TRANSACTION
                                else age_reason)
            checks.timestamp(getattr(data, "event_time", None), name, "event_time", age_reason=event_age_reason)
            checks.timestamp(getattr(data, "observed_at", None), name, "observed_at", age_reason=age_reason)
            if name == "premium_index":
                checks.premium(data, name)
            else:
                checks.interest(data, name)

    snapshot = assembly.snapshot
    if not isinstance(snapshot, MarketSnapshot):
        checks.add(R.INCOMPLETE_SNAPSHOT, "snapshot", "data")
    else:
        checks.identity(snapshot, "snapshot")
        checks.symbol_check(snapshot.symbol, "snapshot")
        checks.book(snapshot.spot_book, "snapshot.spot", Market.SPOT)
        checks.book(snapshot.perp_book, "snapshot.perp", Market.USDTM)
        checks.premium(snapshot, "snapshot", unified=True)
        checks.interest(snapshot, "snapshot")
        checks.timestamp(snapshot.observed_at, "snapshot", "observed_at")
        premium = data_items.get("premium_index")
        comparisons = (
            ("spot_book", data_items.get("spot_depth")),
            ("perp_book", getattr(data_items.get("perp_depth"), "order_book", None)),
            ("mark_price", getattr(premium, "mark_price", None)),
            ("index_price", getattr(premium, "index_price", None)),
            ("funding_rate", getattr(premium, "last_funding_rate", None)),
            ("next_funding_time", getattr(premium, "next_funding_time", None)),
            ("open_interest", getattr(data_items.get("open_interest"), "open_interest", None)),
        )
        for field, original in comparisons:
            if not _same(getattr(snapshot, field, None), original):
                checks.add(R.SOURCE_MISMATCH, "snapshot", field)

    skew = None
    if "spot_depth" in received and "perp_depth" in received:
        skew = abs(received["perp_depth"] - received["spot_depth"])
        if skew > config.max_sampling_skew:
            checks.add(R.SAMPLING_SKEW_TOO_LARGE, "snapshot", "depth_received_at")
    return DataQualityResult(at, config, tuple(checks.issues), skew)
