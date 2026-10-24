"""One offline forward exit validation using supplied subsequent market evidence."""

from dataclasses import replace
from datetime import datetime
from decimal import Decimal, DecimalException

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.arbitrage.fees import FourLegFees, calculate_fee
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent, accrue_short_funding
from crypto_arbitrage_bot.arbitrage.profit import decompose_fills, net_return, round_trip_from_fills
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.data_quality import DataQualityReason, OpenInterestTimeSemantics
from crypto_arbitrage_bot.models.forward_validation import ForwardValidationResult
from crypto_arbitrage_bot.models.forward_validation import ValidationReason as R
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.models.market import BidAsk
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, utc_timestamp


@decimal_calculation
def validate_forward_outcome(
    record: OpportunityRecord, exit_snapshot: SnapshotAssembly, *,
    validation_time: datetime, funding_events: tuple[FundingEvent, ...],
) -> ForwardValidationResult:
    """Retain original entries; simulate only exits; never fetch or write data.

    Entry time is the saved estimate's opened_at, not logger created_at. Both
    validation time and subsequent sources must also postdate created_at and
    the original assembled snapshot. Return uses original occupied capital.
    Caller supplies observed subsequent data; this function cannot authenticate
    its historical origin. Out-of-window events invalidate the attempt.
    """
    if not isinstance(record, OpportunityRecord) or not isinstance(exit_snapshot, SnapshotAssembly):
        raise TypeError("OpportunityRecord and SnapshotAssembly are required")
    result = ForwardValidationResult(record.id, record.symbol, S.INVALID, (), None,
                                     record.net_profit, record.net_return)
    try:
        at = utc_timestamp(validation_time)
    except (ValueError, TypeError):
        return replace(result, reason_codes=(R.INVALID_VALIDATION_TIME,))
    result = replace(result, validation_time=at)
    try:
        entry = decode_entry(record)
    except (KeyError, IndexError):
        return replace(result, status=S.INCOMPLETE, reason_codes=(R.MISSING_ENTRY_EVIDENCE,))
    except (ValueError, TypeError, AttributeError, DecimalException, OverflowError):
        return replace(result, reason_codes=(R.INVALID_ENTRY_EVIDENCE,))
    quantity = record.matched_quantity
    assert quantity is not None and record.capital_occupied is not None
    result = replace(result, entry_time=entry.opened_at, matched_quantity=quantity,
                     capital_occupied=record.capital_occupied, original_entry_basis=entry.entry_basis,
                     entry_fills=(entry.spot, entry.perp))
    quality = validate_market_data(exit_snapshot, entry.quality_config, evaluated_at=at)
    result = replace(result, exit_data_quality=quality)
    reasons = []
    identity = entry.research_instrument
    snapshot_identity = exit_snapshot.snapshot
    if (exit_snapshot.research_instrument != identity
            or (snapshot_identity is not None and (
                snapshot_identity.research_instrument != identity
                or snapshot_identity.spot_book.research_instrument != identity
                or snapshot_identity.perp_book.research_instrument != identity))):
        reasons.append(R.RESEARCH_IDENTITY_MISMATCH)
    if (exit_snapshot.symbol != record.symbol or entry.spot.symbol != record.symbol
            or entry.perp.symbol != record.symbol or DataQualityReason.SYMBOL_MISMATCH in quality.reason_codes):
        reasons.append(R.SYMBOL_MISMATCH)
    if any(fill.requested_quantity != quantity or fill.filled_quantity != quantity
           for fill in (entry.spot, entry.perp)):
        reasons.append(R.QUANTITY_MISMATCH)
    try:
        anchor = max(entry.opened_at, utc_timestamp(record.created_at),
                     utc_timestamp(record.snapshot_assembled_at))
        source_times = [exit_snapshot.assembled_at]
        for source_name in ("spot_depth", "perp_depth", "premium_index", "open_interest"):
            source = getattr(exit_snapshot.sources, source_name)
            received_at = getattr(source, "received_at", None)
            if received_at is not None:
                source_times.append(received_at)
            data = getattr(source, "data", None)
            observation = (getattr(source, "order_book", None) if source_name == "spot_depth"
                           else getattr(data, "order_book", data))
            # Under the explicitly saved policy, OI time is a transaction time,
            # not the timestamp of this new HTTP observation. Receive/observed
            # times still must postdate entry; the quality layer checks freshness.
            oi_transaction_time = (source_name == "open_interest" and
                entry.quality_config.open_interest_time_semantics is OpenInterestTimeSemantics.LAST_TRANSACTION)
            for observed in (getattr(observation, "observed_at", None),
                             None if oi_transaction_time else getattr(observation, "event_time", None),
                             getattr(data, "transaction_time", None)):
                if observed is not None:
                    source_times.append(observed)
        if at <= anchor or any(utc_timestamp(time) <= anchor for time in source_times):
            reasons.append(R.VALIDATION_TIME_NOT_AFTER_ENTRY)
    except (ValueError, TypeError):
        reasons.append(R.INVALID_VALIDATION_TIME)
    if not quality.valid:
        reasons.append(R.INVALID_EXIT_MARKET_DATA)
    if not isinstance(funding_events, tuple) or any(not isinstance(event, FundingEvent) for event in funding_events):
        reasons.append(R.INVALID_FUNDING_EVENTS)
    else:
        result = replace(result, funding_events=funding_events)
        if any(event.symbol != record.symbol for event in funding_events):
            reasons.append(R.SYMBOL_MISMATCH)
        if any(not entry.opened_at < event.settled_at <= at for event in funding_events):
            reasons.append(R.FUNDING_EVENT_OUTSIDE_HOLDING_WINDOW)
    if reasons:
        return replace(result, reason_codes=tuple(dict.fromkeys(reasons)))
    result = replace(result, holding_duration=at - entry.opened_at)
    try:
        funding = accrue_short_funding(quantity, entry.opened_at, at, funding_events, record.symbol)
    except (ValueError, TypeError, DecimalException):
        return replace(result, reason_codes=(R.INVALID_FUNDING_EVENTS,))
    snapshot = exit_snapshot.snapshot
    assert snapshot is not None
    spot_exit = simulate_fill(snapshot.spot_book, TradeLeg.SPOT_EXIT, quantity)
    perp_exit = simulate_fill(snapshot.perp_book, TradeLeg.PERP_EXIT, quantity)
    result = replace(result, exit_fills=(spot_exit, perp_exit))
    if not spot_exit.complete or not perp_exit.complete:
        return replace(result, status=S.INCOMPLETE, reason_codes=(R.INSUFFICIENT_EXIT_DEPTH,))
    try:
        fees = FourLegFees(
            entry.spot_fee,
            sum((calculate_fee(part.price, part.quantity, entry.rates.spot_exit)
                 for part in spot_exit.fills), Decimal(0)),
            entry.perp_fee,
            sum((calculate_fee(part.price, part.quantity, entry.rates.perp_exit)
                 for part in perp_exit.fills), Decimal(0)),
        )
        calculation = round_trip_from_fills(entry.spot, spot_exit, entry.perp, perp_exit,
                                          fees, funding, entry.other_costs,
                                          research_instrument=identity)
        if calculation.entry_basis != entry.entry_basis:
            return replace(result, reason_codes=(R.INVALID_ENTRY_EVIDENCE,))
        assert snapshot.spot_book.best_bid is not None and snapshot.spot_book.best_ask is not None
        assert snapshot.perp_book.best_bid is not None and snapshot.perp_book.best_ask is not None
        decomposition = decompose_fills(
            entry.spot, spot_exit, entry.perp, perp_exit,
            spot_entry_quote=entry.spot_quote, perp_entry_quote=entry.perp_quote,
            spot_exit_quote=BidAsk(snapshot.spot_book.best_bid, snapshot.spot_book.best_ask),
            perp_exit_quote=BidAsk(snapshot.perp_book.best_bid, snapshot.perp_book.best_ask),
            research_instrument=identity,
        )
        rate = net_return(calculation.net_profit, record.capital_occupied)
    except (ValueError, TypeError, DecimalException):
        return replace(result, reason_codes=(R.INVALID_ENTRY_EVIDENCE,))
    return replace(
        result, status=S.VALIDATED, calculation=calculation, fees=fees, decomposition=decomposition,
        spot_pnl=spot_exit.quote_notional - entry.spot.quote_notional,
        perp_pnl=entry.perp.quote_notional - perp_exit.quote_notional,
        validated_simulated_net_return=rate,
        profit_error=None if record.net_profit is None else calculation.net_profit - record.net_profit,
        return_error=None if record.net_return is None else rate - record.net_return,
        basis_change=calculation.exit_basis - calculation.entry_basis,
    )
