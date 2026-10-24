"""Observed funding adjustments; not an exchange settlement or asset eligibility."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


@dataclass(frozen=True)
class FundingAdjustment:
    symbol: str
    adjusted_funding_rate_cap: Decimal
    adjusted_funding_rate_floor: Decimal
    funding_interval_hours: int
    observed_at: datetime
    disclaimer: bool | None = None
    # The official endpoint has no update/event timestamp. Never synthesize one.
    official_time: None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Bulk metadata can contain Unicode/non-USDT symbols. This is not a
        # ResearchInstrument, qualified catalog pair, or execution permission.
        if type(self.symbol) is not str or re.fullmatch(r"\w{1,64}", self.symbol) is None:
            raise ValueError("official funding adjustment symbol required")
        require_decimal(self.adjusted_funding_rate_cap, "funding cap")
        require_decimal(self.adjusted_funding_rate_floor, "funding floor")
        if self.adjusted_funding_rate_cap < self.adjusted_funding_rate_floor:
            raise ValueError("funding floor must not exceed cap")
        if type(self.funding_interval_hours) is not int or self.funding_interval_hours <= 0:
            raise ValueError("funding interval must be a positive integer")
        if self.disclaimer is not None and type(self.disclaimer) is not bool:
            raise ValueError("disclaimer must be a boolean when supplied")
        object.__setattr__(self, "observed_at", utc_timestamp(self.observed_at))
