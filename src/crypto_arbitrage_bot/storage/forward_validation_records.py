"""Map an existing forward result without invoking the validator or maths."""

from datetime import datetime
from uuid import uuid4

from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.forward_validation import ForwardValidationResult, ValidationStatus
from crypto_arbitrage_bot.models.validation_record import ForwardValidationRecord
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


def build_forward_validation_record(
    result: ForwardValidationResult, *, validation_id: str | None = None,
    created_at: datetime | None = None,
) -> ForwardValidationRecord:
    if not isinstance(result, ForwardValidationResult):
        raise TypeError("result must be ForwardValidationResult")
    identifier = str(uuid4()) if validation_id is None else validation_id
    for value in (identifier, result.opportunity_id, result.symbol):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("validation id, opportunity id and symbol must be nonempty")
    entry = None if result.entry_time is None else utc_timestamp(result.entry_time)
    at = None if result.validation_time is None else utc_timestamp(result.validation_time)
    if entry is not None and at is not None and at <= entry:
        raise ValueError("validation time must be after entry time")
    if result.holding_duration is not None:
        if entry is None or at is None or result.holding_duration != at - entry:
            raise ValueError("holding duration must agree with entry and validation times")
    if result.status is ValidationStatus.VALIDATED:
        if any(value is None for value in (
            entry, at, result.holding_duration, result.validated_simulated_net_profit,
            result.validated_simulated_net_return,
        )):
            raise ValueError("VALIDATED result must include times and profit/return")
    elif result.calculation is not None or result.validated_simulated_net_return is not None:
        raise ValueError("failed validation cannot expose usable profit/return")
    exit_prices = {}
    for fill in result.exit_fills:
        if fill.leg not in (TradeLeg.SPOT_EXIT, TradeLeg.PERP_EXIT) or fill.leg in exit_prices:
            raise ValueError("exit fills must have unique exit legs")
        if fill.symbol != result.symbol:
            raise ValueError("exit fill symbol differs from validation")
        exit_prices[fill.leg] = fill.vwap
    fees = result.fees
    return ForwardValidationRecord(
        id=identifier, opportunity_id=result.opportunity_id, symbol=result.symbol,
        entry_time=entry, validation_time=at, holding_duration=result.holding_duration,
        status=result.status.value, reason_codes=tuple(code.value for code in result.reason_codes),
        original_estimated_net_profit=result.original_estimated_net_profit,
        original_estimated_net_return=result.original_estimated_net_return,
        validated_simulated_net_profit=result.validated_simulated_net_profit,
        validated_simulated_net_return=result.validated_simulated_net_return,
        profit_error=result.profit_error, return_error=result.return_error,
        original_entry_basis=result.original_entry_basis, validated_exit_basis=result.validated_exit_basis,
        basis_change=result.basis_change, funding_total=result.funding_total,
        spot_exit_vwap=exit_prices.get(TradeLeg.SPOT_EXIT),
        perp_exit_vwap=exit_prices.get(TradeLeg.PERP_EXIT),
        spot_pnl=result.spot_pnl, perp_pnl=result.perp_pnl, price_pnl=result.price_pnl,
        spot_entry_fee=None if fees is None else fees.spot_entry,
        perp_entry_fee=None if fees is None else fees.perp_entry,
        spot_exit_fee=None if fees is None else fees.spot_exit,
        perp_exit_fee=None if fees is None else fees.perp_exit,
        other_costs=result.other_costs, matched_quantity=result.matched_quantity,
        capital_occupied=result.capital_occupied,
        created_at=utc_timestamp(utc_now() if created_at is None else created_at),
        details_json=exact_json(result),
    )
