"""Explicit research thresholds and decisions; never trading authorizations."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.models.estimate import OfflineEstimate
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


@dataclass(frozen=True)
class DecisionConfig:
    """Caller-injected thresholds, with no market-specific default values.

    Profit, buffer and notional are USDT; return is a decimal fraction;
    residual delta is an absolute base-asset quantity. All five mandatory
    thresholds are hard limits. Separate higher profit/return targets define
    an OBSERVE band above the hard limits and below PASS. Bounds are inclusive,
    except that profit and return must always be strictly positive.

    The buffer compares directly to final net profit. It is not an extra cost
    and is neither deducted from profit nor added to min_net_profit.
    """

    min_net_profit: Decimal
    min_net_return: Decimal
    min_matched_notional: Decimal
    max_residual_delta: Decimal
    min_profit_buffer: Decimal
    pass_net_profit: Decimal
    pass_net_return: Decimal

    def __post_init__(self) -> None:
        for name in (
            "min_net_profit", "min_net_return", "min_matched_notional",
            "max_residual_delta", "min_profit_buffer", "pass_net_profit", "pass_net_return",
        ):
            require_decimal(getattr(self, name), name, nonnegative=True)
        if self.pass_net_profit <= max(
            self.min_net_profit, self.min_profit_buffer,
        ):
            raise ValueError("pass_net_profit must exceed both hard profit limits")
        if self.pass_net_return <= self.min_net_return:
            raise ValueError("pass_net_return must exceed min_net_return")


class Decision(StrEnum):
    PASS = "PASS"
    OBSERVE = "OBSERVE"
    REJECT = "REJECT"


class DecisionReason(StrEnum):
    INVALID_MARKET_DATA = "INVALID_MARKET_DATA"
    INCOMPLETE_ESTIMATE = "INCOMPLETE_ESTIMATE"
    INVALID_ESTIMATE_VALUES = "INVALID_ESTIMATE_VALUES"
    INCONSISTENT_EXECUTION = "INCONSISTENT_EXECUTION"
    NON_POSITIVE_NET_PROFIT = "NON_POSITIVE_NET_PROFIT"
    NON_POSITIVE_NET_RETURN = "NON_POSITIVE_NET_RETURN"
    NET_PROFIT_BELOW_THRESHOLD = "NET_PROFIT_BELOW_THRESHOLD"
    NET_RETURN_BELOW_THRESHOLD = "NET_RETURN_BELOW_THRESHOLD"
    MATCHED_NOTIONAL_TOO_SMALL = "MATCHED_NOTIONAL_TOO_SMALL"
    RESIDUAL_DELTA_TOO_LARGE = "RESIDUAL_DELTA_TOO_LARGE"
    PROFIT_BUFFER_INSUFFICIENT = "PROFIT_BUFFER_INSUFFICIENT"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
    PASS_NET_PROFIT_BELOW_TARGET = "PASS_NET_PROFIT_BELOW_TARGET"
    PASS_NET_RETURN_BELOW_TARGET = "PASS_NET_RETURN_BELOW_TARGET"
    PASS_ALL_RULES = "PASS_ALL_RULES"


@dataclass(frozen=True)
class DecisionResult:
    """A research classification with original evidence and no tradable flag.

    PASS only means worth further simulation/observation, never an order or
    a guarantee. matched_notional is the smaller actual entry fill notional,
    not capital occupied or a rounded VWAP multiplied back by quantity.
    """

    status: Decision
    reason_codes: tuple[DecisionReason, ...]
    estimate: OfflineEstimate
    config: DecisionConfig
    matched_notional: Decimal | None
