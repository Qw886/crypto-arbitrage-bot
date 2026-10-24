"""Pure, offline cash-flow and execution-cost calculations.

All values are Decimal, quantities are base-asset units, and profits/capital
are USDT. Funding is an explicit signed input, never predicted here. Entry
and exit prices are supplied observations or clearly labelled assumptions;
no convergence, future funding payment, or executable opportunity is inferred.
"""

from decimal import Decimal

from crypto_arbitrage_bot.arbitrage.basis import mid_price
from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.models.calculation import (
    CalculationResult,
    FillSlice,
    PricePnlDecomposition,
    SimulatedFill,
    TradeLeg,
)
from crypto_arbitrage_bot.models.market import BidAsk, Side, require_market_symbol
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, require_decimal

ZERO = Decimal("0")


@decimal_calculation
def price_pnl(
    quantity: Decimal,
    spot_entry: Decimal,
    spot_exit: Decimal,
    perp_entry: Decimal,
    perp_exit: Decimal,
) -> Decimal:
    """Long-spot plus short-perpetual execution PnL, before fees/funding.

    Supply actual average buy/sell execution prices. Spread and slippage are
    already included and must not be subtracted a second time.
    """
    require_decimal(quantity, "quantity", positive=True)
    for name, value in (
        ("spot_entry", spot_entry), ("spot_exit", spot_exit),
        ("perp_entry", perp_entry), ("perp_exit", perp_exit),
    ):
        require_decimal(value, name, positive=True)
    return quantity * ((spot_exit - spot_entry) + (perp_entry - perp_exit))


@decimal_calculation
def basis_pnl(quantity: Decimal, entry_basis: Decimal, exit_basis: Decimal) -> Decimal:
    """PnL from the change in basis; both bases must use the same convention."""
    require_decimal(quantity, "quantity", positive=True)
    require_decimal(entry_basis, "entry_basis")
    require_decimal(exit_basis, "exit_basis")
    return quantity * (entry_basis - exit_basis)


@decimal_calculation
def net_profit(
    price_pnl_value: Decimal,
    funding_total: Decimal,
    fees: FourLegFees,
    other_costs: Decimal = ZERO,
) -> Decimal:
    """Combine execution PnL with signed short-side funding and actual costs.

    There are intentionally no spread/slippage arguments: execution PnL has
    already charged them. Positive funding is income; negative is a cost.
    """
    require_decimal(price_pnl_value, "price_pnl_value")
    require_decimal(funding_total, "funding_total")
    require_decimal(other_costs, "other_costs", nonnegative=True)
    if not isinstance(fees, FourLegFees):
        raise TypeError("fees must be FourLegFees")
    return price_pnl_value + funding_total - fees.total - other_costs


@decimal_calculation
def decompose_price_pnl(
    quantity: Decimal,
    *,
    spot_entry_quote: BidAsk,
    spot_exit_quote: BidAsk,
    perp_entry_quote: BidAsk,
    perp_exit_quote: BidAsk,
    spot_entry: Decimal,
    spot_exit: Decimal,
    perp_entry: Decimal,
    perp_exit: Decimal,
) -> PricePnlDecomposition:
    """Attribute four distinct execution VWAPs to midpoint/spread/slippage.

    Each quote must correspond to its own execution time. Buy VWAP cannot
    improve on the best ask; sell VWAP cannot improve on the best bid in this
    market-order book model. Midpoints are used only for attribution.
    """
    direct_pnl = price_pnl(quantity, spot_entry, spot_exit, perp_entry, perp_exit)
    execution_prices = (spot_entry, spot_exit, perp_entry, perp_exit)
    quotes = (spot_entry_quote, spot_exit_quote, perp_entry_quote, perp_exit_quote)
    legs = (TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT, TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT)
    mids = []
    spread_per_unit = ZERO
    slippage_per_unit = ZERO
    for quote, execution_price, leg in zip(quotes, execution_prices, legs, strict=True):
        if not isinstance(quote, BidAsk):
            raise TypeError(f"{leg.value} quote must be BidAsk")
        midpoint = mid_price(quote.bid, quote.ask)
        mids.append(midpoint)
        if leg.side is Side.BUY:
            if execution_price < quote.ask:
                raise ValueError(f"{leg.value} buy VWAP is below best ask")
            spread_per_unit += quote.ask - midpoint
            slippage_per_unit += execution_price - quote.ask
        else:
            if execution_price > quote.bid:
                raise ValueError(f"{leg.value} sell VWAP is above best bid")
            spread_per_unit += midpoint - quote.bid
            slippage_per_unit += quote.bid - execution_price
    spot_entry_mid, spot_exit_mid, perp_entry_mid, perp_exit_mid = mids
    midpoint_pnl = basis_pnl(
        quantity, perp_entry_mid - spot_entry_mid, perp_exit_mid - spot_exit_mid,
    )
    spread_costs = quantity * spread_per_unit
    slippage_costs = quantity * slippage_per_unit
    residual = direct_pnl - (midpoint_pnl - spread_costs - slippage_costs)
    return PricePnlDecomposition(
        mid_basis_pnl=midpoint_pnl,
        spread_costs=spread_costs,
        slippage_costs=slippage_costs,
        price_pnl=direct_pnl,
        rounding_residual=residual,
    )


def _validate_fill(fill: SimulatedFill, expected_leg: TradeLeg,
                   research_instrument: ResearchInstrument | None = None) -> None:
    """Called only inside round_trip_from_fills' fixed Decimal context."""
    if not isinstance(fill, SimulatedFill):
        raise TypeError(f"{expected_leg.value} must be SimulatedFill")
    if fill.leg is not expected_leg:
        raise ValueError(f"expected {expected_leg.value} fill")
    require_market_symbol(fill.symbol, research_instrument)
    if not fill.complete or fill.filled_quantity != fill.requested_quantity:
        raise ValueError("all four fills must be complete; insufficient depth cannot produce profit")
    if fill.vwap is None or fill.best_price is None:
        raise ValueError("complete fill requires VWAP and best price")
    if not isinstance(fill.fills, tuple) or not fill.fills:
        raise ValueError("complete fill requires nonempty execution slices")
    if any(not isinstance(part, FillSlice) for part in fill.fills):
        raise TypeError("execution slices must be FillSlice")
    executed_quantity = sum((part.quantity for part in fill.fills), ZERO)
    executed_notional = sum((part.price * part.quantity for part in fill.fills), ZERO)
    if executed_quantity != fill.filled_quantity or executed_notional != fill.quote_notional:
        raise ValueError("fill quantities/notional disagree with execution slices")
    if fill.vwap != executed_notional / executed_quantity:
        raise ValueError("fill VWAP disagrees with execution slices")
    if fill.fills[0].price != fill.best_price:
        raise ValueError("first execution slice must use the best price")
    previous = fill.best_price
    for part in fill.fills:
        if expected_leg.side is Side.BUY and part.price < previous:
            raise ValueError("buy execution slices must walk asks in ascending order")
        if expected_leg.side is Side.SELL and part.price > previous:
            raise ValueError("sell execution slices must walk bids in descending order")
        previous = part.price


@decimal_calculation
def round_trip_from_fills(
    spot_entry: SimulatedFill,
    spot_exit: SimulatedFill,
    perp_entry: SimulatedFill,
    perp_exit: SimulatedFill,
    fees: FourLegFees,
    funding_total: Decimal,
    other_costs: Decimal = ZERO,
    *, research_instrument: ResearchInstrument | None = None,
) -> CalculationResult:
    """Calculate a complete equal-base-quantity, quote-fee simulated round trip.

    Unequal gross fills (including base-asset fee inventory changes) are
    rejected. Fees here are separate quote costs. No quantity mismatch is
    silently treated as a delta-neutral position.

    Cash flow uses exact summed execution notionals, avoiding multiplication
    back from a potentially recurring/rounded average execution price.
    """
    fills = (spot_entry, spot_exit, perp_entry, perp_exit)
    legs = (TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT, TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT)
    for fill, leg in zip(fills, legs, strict=True):
        _validate_fill(fill, leg, research_instrument)
    if len({fill.symbol for fill in fills}) != 1:
        raise ValueError("all four fills must have the same symbol")
    if len({fill.filled_quantity for fill in fills}) != 1:
        raise ValueError("all four fills must have identical base quantities; net inventory must match")
    gross_pnl = (
        (spot_exit.quote_notional - spot_entry.quote_notional)
        + (perp_entry.quote_notional - perp_exit.quote_notional)
    )
    profit = net_profit(gross_pnl, funding_total, fees, other_costs)
    # _validate_fill guarantees these are Decimal rather than None.
    assert spot_entry.vwap is not None and spot_exit.vwap is not None
    assert perp_entry.vwap is not None and perp_exit.vwap is not None
    return CalculationResult(
        symbol=spot_entry.symbol,
        quantity=spot_entry.filled_quantity,
        entry_basis=perp_entry.vwap - spot_entry.vwap,
        exit_basis=perp_exit.vwap - spot_exit.vwap,
        price_pnl=gross_pnl,
        funding_total=funding_total,
        spot_fees=fees.spot_total,
        perp_fees=fees.perp_total,
        total_fees=fees.total,
        other_costs=other_costs,
        net_profit=profit,
    )


@decimal_calculation
def occupied_capital(
    spot_cash: Decimal,
    perp_notional_reserve: Decimal,
    fee_reserve: Decimal,
    stress_reserve: Decimal,
) -> Decimal:
    """Sum spot cash, full perpetual notional, and separate fee/stress reserves.

    The caller supplies the full simulated perpetual notional reserve. There
    is no leverage parameter or short-sale cash credit. Full collateral is a
    research assumption, not a claim that liquidation risk is eliminated.
    Fee/stress reserves are distinct budgets and must not overlap.
    """
    require_decimal(spot_cash, "spot_cash", positive=True)
    require_decimal(perp_notional_reserve, "perp_notional_reserve", positive=True)
    require_decimal(fee_reserve, "fee_reserve", nonnegative=True)
    require_decimal(stress_reserve, "stress_reserve", nonnegative=True)
    return spot_cash + perp_notional_reserve + fee_reserve + stress_reserve


@decimal_calculation
def decompose_fills(
    spot_entry: SimulatedFill,
    spot_exit: SimulatedFill,
    perp_entry: SimulatedFill,
    perp_exit: SimulatedFill,
    *,
    spot_entry_quote: BidAsk,
    spot_exit_quote: BidAsk,
    perp_entry_quote: BidAsk,
    perp_exit_quote: BidAsk,
    research_instrument: ResearchInstrument | None = None,
) -> PricePnlDecomposition:
    """Attribute exact fill notionals without multiplying recurring VWAPs back.

    The supplied top-of-book quotes must belong to the corresponding fill.
    This is the preferred reconciliation entry point for order-book fills.
    """
    zero_fees = FourLegFees(ZERO, ZERO, ZERO, ZERO)
    cashflow = round_trip_from_fills(
        spot_entry, spot_exit, perp_entry, perp_exit, zero_fees, ZERO,
        research_instrument=research_instrument,
    )
    fills = (spot_entry, spot_exit, perp_entry, perp_exit)
    quotes = (spot_entry_quote, spot_exit_quote, perp_entry_quote, perp_exit_quote)
    mids: list[Decimal] = []
    spread = ZERO
    slippage = ZERO
    q = cashflow.quantity
    for fill, quote in zip(fills, quotes, strict=True):
        if not isinstance(quote, BidAsk):
            raise TypeError("quotes must be BidAsk")
        midpoint = mid_price(quote.bid, quote.ask)
        mids.append(midpoint)
        if fill.leg.side is Side.BUY:
            if fill.best_price != quote.ask:
                raise ValueError("buy fill best price must equal its quote ask")
            spread += q * (quote.ask - midpoint)
            slippage += fill.quote_notional - q * quote.ask
        else:
            if fill.best_price != quote.bid:
                raise ValueError("sell fill best price must equal its quote bid")
            spread += q * (midpoint - quote.bid)
            slippage += q * quote.bid - fill.quote_notional
    se, sx, fe, fx = mids
    mid_pnl = q * ((fe - se) - (fx - sx))
    attributed = mid_pnl - spread - slippage
    return PricePnlDecomposition(
        mid_basis_pnl=mid_pnl,
        spread_costs=spread,
        slippage_costs=slippage,
        price_pnl=cashflow.price_pnl,
        rounding_residual=cashflow.price_pnl - attributed,
    )


@decimal_calculation
def decomposed_net_profit(
    decomposition: PricePnlDecomposition,
    funding_total: Decimal,
    fees: FourLegFees,
    other_costs: Decimal = ZERO,
) -> Decimal:
    """Independently sum the published cost attribution; do not reuse price_pnl."""
    if not isinstance(decomposition, PricePnlDecomposition):
        raise TypeError("decomposition must be PricePnlDecomposition")
    attributed = (
        decomposition.mid_basis_pnl
        - decomposition.spread_costs
        - decomposition.slippage_costs
        + decomposition.rounding_residual
    )
    return net_profit(attributed, funding_total, fees, other_costs)


@decimal_calculation
def net_return(net_profit_value: Decimal, capital: Decimal) -> Decimal:
    """Return the holding-period return as a fraction, without annualization."""
    require_decimal(net_profit_value, "net_profit_value")
    require_decimal(capital, "capital", positive=True)
    return net_profit_value / capital
