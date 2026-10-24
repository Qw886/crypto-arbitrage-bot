"""Pure, offline order-book consumption; never places or submits orders."""

from decimal import Decimal, Inexact, localcontext

from crypto_arbitrage_bot.models.calculation import (
    FillSlice,
    FillStatus,
    SimulatedFill,
    TradeLeg,
)
from crypto_arbitrage_bot.models.market import OrderBook, Side
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, require_decimal


def _check_book_order(book: OrderBook) -> None:
    """Check algorithm preconditions, not freshness or live data quality."""
    if any(left.price <= right.price for left, right in zip(book.bids, book.bids[1:])):
        raise ValueError("bids must have unique prices in descending order")
    if any(left.price >= right.price for left, right in zip(book.asks, book.asks[1:])):
        raise ValueError("asks must have unique prices in ascending order")
    if book.bids and book.asks and book.bids[0].price > book.asks[0].price:
        raise ValueError("crossed order book: best bid exceeds best ask")


@decimal_calculation
def simulate_fill(book: OrderBook, leg: TradeLeg, quantity: Decimal) -> SimulatedFill:
    """Consume available depth at its actual prices, including a partial last level.

    BUY legs consume asks and SELL legs consume bids. An insufficient-depth
    result retains only the observable partial fills; its ``vwap`` is None so
    downstream callers cannot treat that partial execution as a complete hedge.
    Empty depth also produces INSUFFICIENT_DEPTH. There is no extrapolation.
    Quantity and notional accounting must be exact in the 50-digit calculation
    context; inputs exceeding that capability raise ValueError. Only the final
    average price may be rounded (for example, a recurring decimal VWAP).
    """
    if not isinstance(book, OrderBook):
        raise TypeError("book must be OrderBook")
    if not isinstance(leg, TradeLeg):
        raise TypeError("leg must be TradeLeg")
    require_decimal(quantity, "quantity", positive=True)
    if book.market is not leg.market:
        raise ValueError("trade leg does not match the order book market")
    _check_book_order(book)

    levels = book.asks if leg.side is Side.BUY else book.bids
    try:
        with localcontext() as context:
            context.traps[Inexact] = True
            remaining = quantity
            notional = Decimal("0")
            slices: list[FillSlice] = []
            for level in levels:
                consumed = min(remaining, level.quantity)
                slices.append(FillSlice(price=level.price, quantity=consumed))
                notional += consumed * level.price
                remaining -= consumed
                if remaining == 0:
                    break
            filled = quantity - remaining
    except Inexact as exc:
        raise ValueError("fill quantity/notional exceeds supported Decimal precision") from exc
    complete = remaining == 0
    return SimulatedFill(
        symbol=book.symbol,
        leg=leg,
        requested_quantity=quantity,
        filled_quantity=filled,
        quote_notional=notional,
        vwap=notional / quantity if complete else None,
        status=FillStatus.FILLED if complete else FillStatus.INSUFFICIENT_DEPTH,
        fills=tuple(slices),
        best_price=levels[0].price if levels else None,
    )
