"""Pure price and basis calculations; all prices are quote units per base unit."""

from decimal import Decimal

from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, require_decimal


@decimal_calculation
def mid_price(bid: Decimal, ask: Decimal) -> Decimal:
    """Return an observational midpoint, never an assumed executable price."""
    require_decimal(bid, "bid", positive=True)
    require_decimal(ask, "ask", positive=True)
    if bid > ask:
        raise ValueError("bid must not exceed ask")
    return (bid + ask) / Decimal(2)


@decimal_calculation
def mid_basis(spot_mid: Decimal, perp_mid: Decimal) -> Decimal:
    """Return perp midpoint minus spot midpoint, in quote/base units."""
    require_decimal(spot_mid, "spot_mid", positive=True)
    require_decimal(perp_mid, "perp_mid", positive=True)
    return perp_mid - spot_mid


@decimal_calculation
def basis_pct(basis: Decimal, spot_mid: Decimal) -> Decimal:
    """Return the basis fraction, e.g. Decimal('0.01') represents 1%."""
    require_decimal(basis, "basis")
    require_decimal(spot_mid, "spot_mid", positive=True)
    return basis / spot_mid


@decimal_calculation
def executable_entry_basis(spot_ask: Decimal, perp_bid: Decimal) -> Decimal:
    """Return the top-of-book short-perp/long-spot entry basis.

    This quote is a reference for an infinitesimal position. Order-book VWAP
    remains necessary to evaluate a specified simulated position size.
    """
    require_decimal(spot_ask, "spot_ask", positive=True)
    require_decimal(perp_bid, "perp_bid", positive=True)
    return perp_bid - spot_ask
