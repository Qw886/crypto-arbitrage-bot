"""Offline fill contracts; no exchange execution is implemented."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.models.market import Market, Side
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


class TradeLeg(StrEnum):
    SPOT_ENTRY = "SPOT_ENTRY"
    SPOT_EXIT = "SPOT_EXIT"
    PERP_ENTRY = "PERP_ENTRY"
    PERP_EXIT = "PERP_EXIT"

    @property
    def side(self) -> Side:
        return Side.BUY if self in (self.SPOT_ENTRY, self.PERP_EXIT) else Side.SELL

    @property
    def market(self) -> Market:
        return Market.SPOT if self in (self.SPOT_ENTRY, self.SPOT_EXIT) else Market.USDTM


class FillStatus(StrEnum):
    FILLED = "FILLED"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"


@dataclass(frozen=True)
class FillSlice:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        require_decimal(self.price, "price", positive=True)
        require_decimal(self.quantity, "quantity", positive=True)


@dataclass(frozen=True)
class SimulatedFill:
    symbol: str
    leg: TradeLeg
    requested_quantity: Decimal
    filled_quantity: Decimal
    quote_notional: Decimal
    vwap: Decimal | None
    status: FillStatus
    fills: tuple[FillSlice, ...]
    best_price: Decimal | None

    def __post_init__(self) -> None:
        require_decimal(self.requested_quantity, "requested_quantity", positive=True)
        require_decimal(self.filled_quantity, "filled_quantity", nonnegative=True)
        require_decimal(self.quote_notional, "quote_notional", nonnegative=True)
        if self.filled_quantity > self.requested_quantity:
            raise ValueError("filled quantity exceeds requested quantity")
        if self.status is FillStatus.FILLED:
            if self.filled_quantity != self.requested_quantity or self.vwap is None:
                raise ValueError("complete fill requires full requested quantity and VWAP")
            require_decimal(self.vwap, "vwap", positive=True)
        elif self.status is FillStatus.INSUFFICIENT_DEPTH:
            if self.filled_quantity >= self.requested_quantity or self.vwap is not None:
                raise ValueError("incomplete fill cannot expose a full-execution VWAP")
        else:
            raise ValueError("invalid fill status")
        if self.best_price is not None:
            require_decimal(self.best_price, "best_price", positive=True)

    @property
    def complete(self) -> bool:
        return self.status is FillStatus.FILLED


@dataclass(frozen=True)
class CalculationResult:
    """Four-leg simulated cash flow, with every amount expressed in USDT.

    Entry/exit basis use execution VWAP and already include spread/slippage.
    This model assumes fees are separate quote-equivalent costs and all four
    fills have the same net base-asset quantity. It is not a realized trade.
    """

    symbol: str
    quantity: Decimal
    entry_basis: Decimal
    exit_basis: Decimal
    price_pnl: Decimal
    funding_total: Decimal
    spot_fees: Decimal
    perp_fees: Decimal
    total_fees: Decimal
    other_costs: Decimal
    net_profit: Decimal

    def __post_init__(self) -> None:
        require_decimal(self.quantity, "quantity", positive=True)
        for name in ("entry_basis", "exit_basis", "price_pnl", "funding_total", "net_profit"):
            require_decimal(getattr(self, name), name)
        for name in ("spot_fees", "perp_fees", "total_fees", "other_costs"):
            require_decimal(getattr(self, name), name, nonnegative=True)


@dataclass(frozen=True)
class PricePnlDecomposition:
    """Attribution of execution PnL, not additional costs to subtract again.

    price_pnl = mid_basis_pnl - spread_costs - slippage_costs + rounding_residual.
    The residual exposes only the difference from finite Decimal arithmetic;
    it is zero when the input calculation fits within the 50-digit context.
    """

    mid_basis_pnl: Decimal
    spread_costs: Decimal
    slippage_costs: Decimal
    price_pnl: Decimal
    rounding_residual: Decimal

    def __post_init__(self) -> None:
        for name in ("mid_basis_pnl", "price_pnl", "rounding_residual"):
            require_decimal(getattr(self, name), name)
        for name in ("spread_costs", "slippage_costs"):
            require_decimal(getattr(self, name), name, nonnegative=True)
