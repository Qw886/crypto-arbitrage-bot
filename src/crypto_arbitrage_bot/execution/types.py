"""Typed, Decimal-only requests and responses for the testnet order gateway."""

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from decimal import Decimal, localcontext
from enum import StrEnum

from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS, Side
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class ExecutionMarket(StrEnum):
    SPOT = "SPOT"
    USDTM = "USDTM"


@dataclass(frozen=True)
class FundingPayment:
    """Actual account FUNDING_FEE amount, never a scenario rate estimate."""

    transaction_id: str
    symbol: str
    posted_at: datetime
    amount: Decimal
    asset: str = "USDT"

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id.strip():
            raise ValueError("invalid funding transaction identity")
        if self.symbol not in SUPPORTED_SYMBOLS or self.asset != "USDT":
            raise ValueError("unsupported funding symbol/asset")
        require_decimal(self.amount, "funding ledger amount")
        object.__setattr__(self, "posted_at", utc_timestamp(self.posted_at))


class OrderState(StrEnum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXPIRED_IN_MATCH = "EXPIRED_IN_MATCH"


@dataclass(frozen=True, slots=True)
class AccountTrade:
    """One official test-account fill; commission asset is never guessed."""

    market: ExecutionMarket
    symbol: str
    trade_id: str
    order_id: str
    side: Side
    price: Decimal
    quantity: Decimal
    quote_quantity: Decimal
    commission: Decimal
    commission_asset: str
    traded_at: datetime
    official_perp_pnl: Decimal | None = None
    raw_json: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.market, ExecutionMarket) or not isinstance(self.side, Side):
            raise TypeError("trade market/side must be typed enums")
        if self.symbol not in SUPPORTED_SYMBOLS:
            raise ValueError("unsupported trade symbol")
        for value in (self.trade_id, self.order_id):
            if type(value) is not str or not value.isascii() or not value.isdigit():
                raise ValueError("trade/order identity must be a numeric string")
        for name in ("price", "quantity", "quote_quantity"):
            require_decimal(getattr(self, name), name, positive=True)
        require_decimal(self.commission, "commission")
        if not isinstance(self.commission_asset, str) or not re.fullmatch(r"[A-Z0-9]{1,20}", self.commission_asset):
            raise ValueError("commission asset is missing or invalid")
        if self.market is ExecutionMarket.USDTM:
            if self.official_perp_pnl is None:
                raise ValueError("perpetual trade requires official PnL evidence")
            require_decimal(self.official_perp_pnl, "official_perp_pnl")
        elif self.official_perp_pnl is not None:
            raise ValueError("spot trades cannot contain perpetual PnL")
        object.__setattr__(self, "traded_at", utc_timestamp(self.traded_at))


_CLIENT_ORDER_ID = re.compile(r"^[A-Za-z0-9._:/-]{1,36}$", re.ASCII)


@dataclass(frozen=True, slots=True)
class IocLimitOrder:
    """A bounded-price order. IOC can fill only now and at the supplied limit."""

    market: ExecutionMarket
    symbol: str
    side: Side
    quantity: Decimal
    limit_price: Decimal
    client_order_id: str
    reduce_only: bool = False
    submit_before: datetime | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.market, ExecutionMarket):
            raise TypeError("market must be an ExecutionMarket")
        if self.symbol not in SUPPORTED_SYMBOLS:
            raise ValueError("only BTCUSDT and ETHUSDT are supported")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side")
        require_decimal(self.quantity, "quantity", positive=True)
        require_decimal(self.limit_price, "limit_price", positive=True)
        if type(self.client_order_id) is not str or not _CLIENT_ORDER_ID.fullmatch(self.client_order_id):
            raise ValueError("client_order_id must be 1-36 safe ASCII characters")
        if type(self.reduce_only) is not bool or (self.reduce_only and self.market is not ExecutionMarket.USDTM):
            raise ValueError("reduce_only is only supported on USDT-M")
        if self.submit_before is not None:
            if self.reduce_only:
                raise ValueError("an entry deadline cannot prevent reduce-only exits")
            object.__setattr__(self, "submit_before", utc_timestamp(self.submit_before))


@dataclass(frozen=True, slots=True)
class SymbolTradingRules:
    symbol: str
    market: ExecutionMarket
    min_price: Decimal
    max_price: Decimal | None
    tick_size: Decimal
    step_size: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    min_notional: Decimal

    def __post_init__(self) -> None:
        if self.symbol not in SUPPORTED_SYMBOLS:
            raise ValueError("only BTCUSDT and ETHUSDT are supported")
        if not isinstance(self.market, ExecutionMarket):
            raise TypeError("market must be an ExecutionMarket")
        require_decimal(self.min_price, "min_price", nonnegative=True)
        if self.max_price is not None:
            require_decimal(self.max_price, "max_price", positive=True)
            if self.min_price > self.max_price:
                raise ValueError("minimum price exceeds maximum price")
        for name in ("tick_size", "step_size", "min_quantity", "max_quantity"):
            require_decimal(getattr(self, name), name, positive=True)
        require_decimal(self.min_notional, "min_notional", nonnegative=True)
        if self.min_quantity > self.max_quantity:
            raise ValueError("minimum quantity exceeds maximum quantity")


@dataclass(frozen=True, slots=True)
class OrderReceipt:
    market: ExecutionMarket
    symbol: str
    order_id: str
    client_order_id: str
    status: OrderState
    original_quantity: Decimal
    executed_quantity: Decimal
    cumulative_quote_quantity: Decimal | None
    average_price: Decimal | None
    base_commission: Decimal | None = Decimal("0")
    side: Side | None = dataclass_field(default=None, kw_only=True)
    order_type: str | None = dataclass_field(default=None, kw_only=True)
    time_in_force: str | None = dataclass_field(default=None, kw_only=True)
    limit_price: Decimal | None = dataclass_field(default=None, kw_only=True)
    reduce_only: bool | None = dataclass_field(default=None, kw_only=True)
    position_side: str | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.market, ExecutionMarket) or not isinstance(self.status, OrderState):
            raise TypeError("receipt market/status must be typed enums")
        if self.symbol not in SUPPORTED_SYMBOLS:
            raise ValueError("unsupported receipt symbol")
        for field in ("original_quantity", "executed_quantity"):
            require_decimal(getattr(self, field), field, nonnegative=True)
        for field in ("cumulative_quote_quantity", "average_price"):
            value = getattr(self, field)
            if value is not None:
                require_decimal(value, field, nonnegative=True)
        if self.executed_quantity > self.original_quantity:
            raise ValueError("executed quantity exceeds original quantity")
        if self.base_commission is not None:
            require_decimal(self.base_commission, "base_commission", nonnegative=True)
            if self.base_commission > self.executed_quantity:
                raise ValueError("base commission exceeds executed quantity")
        if self.side is not None and not isinstance(self.side, Side):
            raise TypeError("receipt side must be a typed enum or unknown")
        for name in ('order_type', 'time_in_force', 'position_side'):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise TypeError("receipt semantics must be strings or unknown")
        if self.limit_price is not None:
            require_decimal(self.limit_price, 'receipt limit_price', positive=True)
        if self.reduce_only is not None and type(self.reduce_only) is not bool:
            raise TypeError("receipt reduce_only must be boolean or unknown")

    def matches_order(self, order: IocLimitOrder) -> bool:
        """Unknown semantics cannot establish a match to durable order intent."""
        identity_matches = (type(self.order_id) is str and self.order_id.isascii() and self.order_id.isdigit()
                and self.market is order.market and self.symbol == order.symbol
                and self.client_order_id == order.client_order_id
                and self.original_quantity == order.quantity and self.side is order.side
                and self.order_type == 'LIMIT' and self.time_in_force == 'IOC'
                and self.limit_price == order.limit_price
                and ((self.reduce_only is order.reduce_only and self.position_side == 'BOTH')
                     if self.market is ExecutionMarket.USDTM else
                     (self.reduce_only is None and self.position_side is None)))
        if not identity_matches:
            return False
        if ((self.status is OrderState.FILLED and self.executed_quantity != self.original_quantity)
                or (self.status is OrderState.REJECTED and self.executed_quantity != 0)):
            return False
        if self.executed_quantity == 0:
            return self.average_price in (None, Decimal(0)) and self.cumulative_quote_quantity in (None, Decimal(0))
        if self.average_price is None and self.cumulative_quote_quantity is None:
            return False
        # Compare quote directly; dividing to obtain a VWAP can round at a limit.
        with localcontext() as context:
            context.prec = max(80, len(self.executed_quantity.as_tuple().digits)
                               + len(order.limit_price.as_tuple().digits))
            quote_limit = self.executed_quantity * order.limit_price
            for value, bound in ((self.average_price, order.limit_price),
                                 (self.cumulative_quote_quantity, quote_limit)):
                if value is not None and (value <= 0 or
                        (value > bound if order.side is Side.BUY else value < bound)):
                    return False
        return True


@dataclass(frozen=True, slots=True)
class AccountState:
    market: ExecutionMarket
    symbol: str
    quote_available: Decimal
    base_available: Decimal
    position_quantity: Decimal
    can_trade: bool
    one_way: bool
    single_asset: bool
    leverage: Decimal
    open_order_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.market, ExecutionMarket) or any(type(getattr(self, name)) is not bool for name in
                                                             ("can_trade", "one_way", "single_asset")):
            raise TypeError("invalid account market/mode types")
        for name in ("quote_available", "base_available", "leverage"):
            require_decimal(getattr(self, name), name, nonnegative=True)
        require_decimal(self.position_quantity, "position_quantity")
        if self.symbol not in SUPPORTED_SYMBOLS or type(self.open_order_count) is not int or self.open_order_count < 0:
            raise ValueError("invalid account identity/order count")
