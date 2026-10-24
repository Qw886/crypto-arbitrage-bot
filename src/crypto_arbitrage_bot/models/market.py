"""Immutable domain types. These enforce structure, not live data quality."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TypeAlias

from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp

Price: TypeAlias = Decimal
Quantity: TypeAlias = Decimal
Notional: TypeAlias = Decimal
FeeRate: TypeAlias = Decimal
FundingRate: TypeAlias = Decimal
Timestamp: TypeAlias = datetime


class Asset(StrEnum):
    BTC = "BTC"
    ETH = "ETH"
    USDT = "USDT"


class Market(StrEnum):
    SPOT = "SPOT"
    USDTM = "USDTM"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


SUPPORTED_SYMBOLS = {"BTCUSDT": Asset.BTC, "ETHUSDT": Asset.ETH}


def require_market_symbol(symbol: str, research_instrument: ResearchInstrument | None) -> None:
    if research_instrument is not None:
        if not isinstance(research_instrument, ResearchInstrument):
            raise TypeError("research identity must be ResearchInstrument")
        research_instrument.require_symbol(symbol)
    elif type(symbol) is not str or symbol not in SUPPORTED_SYMBOLS:
        raise ValueError("only BTCUSDT and ETHUSDT are supported without a research identity")


@dataclass(frozen=True)
class BidAsk:
    bid: Price
    ask: Price

    def __post_init__(self) -> None:
        require_decimal(self.bid, "bid", positive=True)
        require_decimal(self.ask, "ask", positive=True)
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")


@dataclass(frozen=True)
class OrderBookLevel:
    price: Price
    quantity: Quantity

    def __post_init__(self) -> None:
        require_decimal(self.price, "price", positive=True)
        require_decimal(self.quantity, "quantity", positive=True)


@dataclass(frozen=True)
class OrderBook:
    symbol: str
    market: Market
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    observed_at: Timestamp
    event_time: Timestamp | None = None
    research_instrument: ResearchInstrument | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        require_market_symbol(self.symbol, self.research_instrument)
        if not isinstance(self.market, Market):
            raise TypeError("market must be Market")
        for name in ("bids", "asks"):
            levels = getattr(self, name)
            if not isinstance(levels, tuple) or any(
                not isinstance(level, OrderBookLevel) for level in levels
            ):
                raise TypeError(f"{name} must be a tuple of OrderBookLevel")
        object.__setattr__(self, "observed_at", utc_timestamp(self.observed_at))
        if self.event_time is not None:
            object.__setattr__(self, "event_time", utc_timestamp(self.event_time))

    @property
    def base_asset(self) -> Asset | str:
        if self.symbol in SUPPORTED_SYMBOLS:
            return SUPPORTED_SYMBOLS[self.symbol]
        assert self.research_instrument is not None
        return self.research_instrument.base_asset

    @property
    def quote_asset(self) -> Asset:
        return Asset.USDT

    @property
    def best_bid(self) -> Price | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Price | None:
        return self.asks[0].price if self.asks else None


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    spot_book: OrderBook
    perp_book: OrderBook
    mark_price: Price
    index_price: Price
    funding_rate: FundingRate
    next_funding_time: Timestamp
    open_interest: Quantity
    observed_at: Timestamp
    research_instrument: ResearchInstrument | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        require_market_symbol(self.symbol, self.research_instrument)
        if (self.spot_book.research_instrument != self.research_instrument
                or self.perp_book.research_instrument != self.research_instrument):
            raise ValueError("research identity must match both books")
        if self.spot_book.symbol != self.symbol or self.perp_book.symbol != self.symbol:
            raise ValueError("snapshot books must refer to the same symbol")
        if self.spot_book.market is not Market.SPOT or self.perp_book.market is not Market.USDTM:
            raise ValueError("snapshot requires a spot book and a USDT-M book")
        require_decimal(self.mark_price, "mark_price", positive=True)
        require_decimal(self.index_price, "index_price", positive=True)
        require_decimal(self.funding_rate, "funding_rate")
        require_decimal(self.open_interest, "open_interest", nonnegative=True)
        object.__setattr__(self, "observed_at", utc_timestamp(self.observed_at))
        object.__setattr__(self, "next_funding_time", utc_timestamp(self.next_funding_time))
