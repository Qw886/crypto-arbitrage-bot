"""Funding cash flow for a constant-quantity hypothetical perpetual short.

Only supplied settlement events are accrued. This module does not predict rates
or manufacture settlement events from a holding duration.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")


@dataclass(frozen=True)
class FundingEvent:
    """A provided actual or simulated settlement, not an indicative live rate.

    ``settlement_mark_price`` is quote/base; ``funding_rate`` is a signed
    dimensionless fraction. ``settled_at`` is normalized to aware UTC.
    """

    event_id: str
    settled_at: datetime
    settlement_mark_price: Decimal
    funding_rate: Decimal
    symbol: str

    def __post_init__(self) -> None:
        _require_identifier(self.event_id, "event_id")
        _require_identifier(self.symbol, "symbol")
        require_decimal(self.settlement_mark_price, "settlement_mark_price", positive=True)
        require_decimal(self.funding_rate, "funding_rate")
        object.__setattr__(self, "settled_at", utc_timestamp(self.settled_at, "settled_at"))


@decimal_calculation
def short_funding_cashflow(
    quantity: Decimal, settlement_mark_price: Decimal, funding_rate: Decimal
) -> Decimal:
    """Return quote cash flow: positive funding credits the short, negative debits it."""
    require_decimal(quantity, "quantity", nonnegative=True)
    require_decimal(settlement_mark_price, "settlement_mark_price", positive=True)
    require_decimal(funding_rate, "funding_rate")
    return quantity * settlement_mark_price * funding_rate


@decimal_calculation
def accrue_short_funding(
    quantity: Decimal,
    opened_at: datetime,
    closed_at: datetime,
    events: Iterable[FundingEvent],
    symbol: str,
) -> Decimal:
    """Accrue matching events satisfying ``opened_at < settled_at <= closed_at``.

    Quantity is the constant short position in base units during this window.
    This interval convention is an explicit simulation rule, not a guarantee
    about exchange processing at the settlement boundary. Events for other
    symbols are ignored. Duplicate IDs or duplicate symbol/settlement times
    are rejected, including duplicates outside the holding window, so a history
    input cannot silently double-count the same settlement.
    """
    require_decimal(quantity, "quantity", nonnegative=True)
    _require_identifier(symbol, "symbol")
    opened = utc_timestamp(opened_at, "opened_at")
    closed = utc_timestamp(closed_at, "closed_at")
    if closed < opened:
        raise ValueError("closed_at must not precede opened_at")

    seen_ids: set[tuple[str, str]] = set()
    seen_times: set[tuple[str, datetime]] = set()
    total = Decimal(0)
    for event in events:
        if not isinstance(event, FundingEvent):
            raise TypeError("events must contain FundingEvent instances")
        identity = (event.symbol, event.event_id)
        settlement = (event.symbol, event.settled_at)
        if identity in seen_ids or settlement in seen_times:
            raise ValueError("duplicate funding event ID or symbol/settlement time")
        seen_ids.add(identity)
        seen_times.add(settlement)
        if event.symbol == symbol and opened < event.settled_at <= closed:
            total += short_funding_cashflow(
                quantity, event.settlement_mark_price, event.funding_rate
            )
    return total
