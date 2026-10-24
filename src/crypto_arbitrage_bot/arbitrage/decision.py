"""Pure classification of a prior estimate, without repeating its mathematics."""

from decimal import Decimal

from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.decision import (
    Decision,
    DecisionConfig,
    DecisionReason,
    DecisionResult,
)
from crypto_arbitrage_bot.models.estimate import EstimateStatus, OfflineEstimate


def _finite(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite()


def decide(estimate: OfflineEstimate, config: DecisionConfig) -> DecisionResult:
    """Apply hard limits, higher observation targets, then PASS.

    Uses the quality result embedded by estimate_once, so an unrelated VALID
    result cannot override a failed estimate. No clock, IO, resimulation,
    funding accrual, fee adjustment, or profit/return calculation occurs here.
    Callers needing a different quality evaluation time must obtain a new
    estimate through the existing quality gate first.
    """
    if not isinstance(estimate, OfflineEstimate):
        raise TypeError("estimate must be OfflineEstimate")
    if not isinstance(config, DecisionConfig):
        raise TypeError("config must be DecisionConfig")

    reasons: list[DecisionReason] = []
    quality = estimate.data_quality
    if quality is None or not quality.valid or not estimate.assembly.complete:
        reasons.append(DecisionReason.INVALID_MARKET_DATA)

    if (
        not estimate.complete or not estimate.assembly.complete
        or estimate.issues or estimate.sizing is None
        or not estimate.sizing.acceptable or estimate.sizing.reasons
        or estimate.fees is None or estimate.decomposition is None
        or len(estimate.fills) != 4
    ):
        reasons.append(DecisionReason.INCOMPLETE_ESTIMATE)

    if estimate.status is EstimateStatus.INSUFFICIENT_DEPTH or any(
        not fill.complete for fill in estimate.fills
    ):
        reasons.append(DecisionReason.INSUFFICIENT_DEPTH)
    if any(fill.complete and not fill.fills for fill in estimate.fills):
        reasons.append(DecisionReason.INCOMPLETE_ESTIMATE)
    if any(
        fill.complete and (not _finite(fill.quote_notional) or fill.quote_notional <= 0)
        for fill in estimate.fills
    ):
        reasons.append(DecisionReason.INVALID_ESTIMATE_VALUES)

    # Presence/type validation is not recalculation of the upstream cash flow.
    profit, net_return = estimate.net_profit, estimate.net_return
    quantity, residual = estimate.matched_quantity, estimate.residual_delta
    capital = estimate.capital_occupied
    required_values = (
        profit, net_return, quantity, residual, capital,
        estimate.executable_entry_basis, estimate.estimated_funding,
    )
    if any(value is None for value in required_values):
        reasons.append(DecisionReason.INCOMPLETE_ESTIMATE)
    if any(value is not None and not _finite(value) for value in required_values):
        reasons.append(DecisionReason.INVALID_ESTIMATE_VALUES)
    if (_finite(quantity) and quantity is not None and quantity <= 0) or (
        _finite(capital) and capital is not None and capital <= 0
    ):
        reasons.append(DecisionReason.INVALID_ESTIMATE_VALUES)

    matched_notional = None
    expected_legs = (
        TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT, TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT,
    )
    if len(estimate.fills) == 4:
        if not _finite(quantity) or any(
            fill.leg is not leg or fill.symbol != estimate.symbol
            or fill.requested_quantity != quantity or fill.filled_quantity != quantity
            for fill, leg in zip(estimate.fills, expected_legs, strict=True)
        ) or (
            estimate.calculation is not None and (
                estimate.calculation.symbol != estimate.symbol
                or estimate.calculation.quantity != quantity
            )
        ):
            reasons.append(DecisionReason.INCONSISTENT_EXECUTION)
        if all(fill.complete for fill in estimate.fills):
            notionals = (estimate.fills[0].quote_notional, estimate.fills[2].quote_notional)
            if all(_finite(value) and value > 0 for value in notionals):
                matched_notional = min(notionals)
            else:
                reasons.append(DecisionReason.INVALID_ESTIMATE_VALUES)

    if _finite(profit) and profit is not None:
        if profit <= 0:
            reasons.append(DecisionReason.NON_POSITIVE_NET_PROFIT)
        if profit < config.min_net_profit:
            reasons.append(DecisionReason.NET_PROFIT_BELOW_THRESHOLD)
        if profit < config.min_profit_buffer:
            reasons.append(DecisionReason.PROFIT_BUFFER_INSUFFICIENT)
    if _finite(net_return) and net_return is not None:
        if net_return <= 0:
            reasons.append(DecisionReason.NON_POSITIVE_NET_RETURN)
        if net_return < config.min_net_return:
            reasons.append(DecisionReason.NET_RETURN_BELOW_THRESHOLD)
    if matched_notional is not None and matched_notional < config.min_matched_notional:
        reasons.append(DecisionReason.MATCHED_NOTIONAL_TOO_SMALL)
    if _finite(residual) and residual is not None and (
        residual.copy_abs() > config.max_residual_delta
    ):
        reasons.append(DecisionReason.RESIDUAL_DELTA_TOO_LARGE)

    if reasons:
        return DecisionResult(
            Decision.REJECT, tuple(dict.fromkeys(reasons)), estimate, config, matched_notional,
        )

    assert profit is not None and net_return is not None
    if profit < config.pass_net_profit:
        reasons.append(DecisionReason.PASS_NET_PROFIT_BELOW_TARGET)
    if net_return < config.pass_net_return:
        reasons.append(DecisionReason.PASS_NET_RETURN_BELOW_TARGET)
    return DecisionResult(
        Decision.OBSERVE if reasons else Decision.PASS,
        tuple(reasons) if reasons else (DecisionReason.PASS_ALL_RULES,),
        estimate, config, matched_notional,
    )
