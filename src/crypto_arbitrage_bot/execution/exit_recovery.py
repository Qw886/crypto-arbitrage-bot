"""Bounded Demo-only repair of a confirmed, terminal Perp exit.

Only applies after Spot has fully exited and the original Perp exit is a
terminal order with zero or partial fills. Only the confirmed residual is
closed. Unknown orders are queried, never blindly repeated.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    PairExecutionResult,
    PairExecutionState,
    PairOrderPlan,
    _order_data,
    _receipt_from_json,
    _receipt_json,
)
from crypto_arbitrage_bot.execution.spot_exit_recovery import confirmed_spot_exit_inventory
from crypto_arbitrage_bot.execution.types import (
    ExecutionMarket,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot

_TERMINAL = frozenset((OrderState.FILLED, OrderState.EXPIRED, OrderState.CANCELED,
                       OrderState.EXPIRED_IN_MATCH, OrderState.REJECTED))


def _terminal_quantity_confirmed(receipt: OrderReceipt) -> bool:
    return (receipt.status in _TERMINAL
            and (receipt.status is not OrderState.FILLED or receipt.executed_quantity == receipt.original_quantity)
            and (receipt.status is not OrderState.REJECTED or receipt.executed_quantity == 0))


def recover_perp_exit(bot: "TestnetBot", plan: PairOrderPlan) -> PairExecutionResult:
    """Caller owns the execution process lock; one new IOC at most per call."""
    current = bot.journal.get(plan.attempt_id)
    if current is None:
        raise ExecutionJournalError("recovery parent missing")
    if current.state is not PairExecutionState.RECOVERY_REQUIRED or bot.config.max_exit_recovery_attempts == 0:
        return current
    if current.failure_code not in ('PERP_EXIT_INCOMPLETE', 'PERP_FLATTEN_INCOMPLETE', 'RECONCILIATION_UNRESOLVED'):
        return current  # Explicit manual/unknown recovery states do not authorize trades.
    se, pe, sx, px = current.spot_entry, current.perp_entry, current.spot_exit, current.perp_exit
    if (se is None or pe is None or sx is None or px is None or se.base_commission is None
            or se.status is not OrderState.FILLED or pe.status is not OrderState.FILLED
            or px.client_order_id != plan.perp_exit.client_order_id
            or se.executed_quantity != plan.spot_entry.quantity
            or pe.executed_quantity != plan.perp_entry.quantity
            or px.status not in _TERMINAL or px.executed_quantity >= pe.executed_quantity
            or px.status is OrderState.FILLED
            or px.status is OrderState.REJECTED and px.executed_quantity != 0):
        return current
    if sx.status is not OrderState.FILLED or sx.executed_quantity != se.executed_quantity - se.base_commission:
        if not bot.config.balance_evidence_enabled:
            return current
        if not confirmed_spot_exit_inventory(bot, plan):
            return current
    # Query the original order before doing anything that can mutate exposure.
    original = bot.perp.get_order(plan.symbol, px.client_order_id)
    if (not bot.coordinator._matches(original, plan.perp_exit)
            or original.order_id != px.order_id or original.status is not px.status
            or original.executed_quantity != px.executed_quantity):
        # Newly known fills or altered status require general reconciliation.
        return current
    with bot.journal._connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS testnet_exit_recoveries (
            client_order_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL, order_json TEXT NOT NULL,
            original_exit_json TEXT NOT NULL, receipt_json TEXT,
            created_at TEXT NOT NULL, UNIQUE(attempt_id, ordinal))""")
        rows = db.execute("SELECT * FROM testnet_exit_recoveries WHERE attempt_id=? ORDER BY ordinal",
                          (plan.attempt_id,)).fetchall()
    # The original terminal order may already have closed some of the short.
    # Persisted recovery fills add to it once; never resubmit the original ID.
    completed = original.executed_quantity
    last_receipt = None
    for ordinal, row in enumerate(rows, start=1):
        try:
            data = json.loads(row["order_json"])
            saved_original = _receipt_from_json(row["original_exit_json"])
            if (row['ordinal'] != ordinal or ordinal > bot.config.max_exit_recovery_attempts
                    or row['client_order_id'] != f'{plan.attempt_id[:24]}-rx{ordinal}'
                    or data['client_order_id'] != row['client_order_id']
                    or data['market'] != ExecutionMarket.USDTM.value or data['symbol'] != plan.symbol
                    or data['side'] != Side.BUY.value or data['reduce_only'] is not True
                    or type(data['quantity']) is not str or type(data['limit_price']) is not str
                    or data.get('submit_before') is not None
                    or saved_original is None
                    or (saved_original.market, saved_original.symbol, saved_original.order_id,
                        saved_original.client_order_id, saved_original.status, saved_original.original_quantity,
                        saved_original.executed_quantity) != (px.market, px.symbol, px.order_id,
                            px.client_order_id, px.status, px.original_quantity, px.executed_quantity)):
                raise ValueError('recovery evidence identity conflict')
            actual = IocLimitOrder(ExecutionMarket.USDTM, plan.symbol, Side.BUY,
                Decimal(data['quantity']), Decimal(data['limit_price']), row['client_order_id'], True)
            if actual.quantity != pe.executed_quantity - completed:
                raise ValueError('recovery evidence quantity conflict')
            receipt = _receipt_from_json(row['receipt_json'])
        except (ValueError, TypeError, KeyError, ArithmeticError):
            bot.logger.error(exact_json({'stage': 'EXIT_RECOVERY', 'attempt_id': plan.attempt_id,
                'reason': 'RECOVERY_EVIDENCE_INVALID', 'ordinal': ordinal}))
            return current
        if receipt is None:
            receipt = bot.perp.get_order(plan.symbol, actual.client_order_id)
            if not bot.coordinator._matches(receipt, actual) or not _terminal_quantity_confirmed(receipt):
                return current
            with bot.journal._connect() as db:
                db.execute("UPDATE testnet_exit_recoveries SET receipt_json=? WHERE client_order_id=?",
                           (_receipt_json(receipt), actual.client_order_id))
        if not bot.coordinator._matches(receipt, actual) or not _terminal_quantity_confirmed(receipt):
            return current
        completed += receipt.executed_quantity
        last_receipt = receipt
    remaining = pe.executed_quantity - completed
    state = bot.perp.get_account_state(plan.symbol)
    if (remaining < 0 or state.position_quantity != -remaining or state.open_order_count
            or not state.can_trade or not state.one_way or not state.single_asset or state.leverage != 1):
        return current
    if remaining == 0 and last_receipt is not None:
        bot.journal.update(plan.attempt_id, state=PairExecutionState.CLOSED, perp_exit=last_receipt,
                           failure_code=None, failure_reason=None)
        result = bot.journal.get(plan.attempt_id)
        assert result is not None
        return result
    if len(rows) >= bot.config.max_exit_recovery_attempts:
        return current
    rules = bot.perp.get_symbol_rules(plan.symbol)
    markets = (TradeLeg.PERP_EXIT.market,)
    try:
        depths = bot._collect_exit_depth(plan.symbol, markets=markets)
    except Exception as error:
        bot._log_execution_failure(plan, "RECOVERY_BLOCKED", error, stage="RECOVERY_QUOTE")
        return current
    # Reuse the book-based execution simulator for the remaining size. The
    # additional limit buffer is explicitly bounded, never a PnL assumption.
    price = bot._exit_limit(depths, TradeLeg.PERP_EXIT, remaining, rules)
    with localcontext() as context:
        context.prec = 50
        price = ((price * (1 + bot.config.max_recovery_price_deviation)) / rules.tick_size).to_integral_value(
            rounding=ROUND_CEILING) * rules.tick_size
    order = replace(plan.perp_exit, quantity=remaining, limit_price=price,
                    client_order_id=f"{plan.attempt_id[:24]}-rx{len(rows) + 1}", reduce_only=True)
    bot.perp._validate_order(order, rules)
    try:
        bot._require_exit_quality(depths, "EXIT_MARKET_EXPIRED_DURING_PREFLIGHT", markets=markets)
    except Exception as error:
        bot._log_execution_failure(plan, "RECOVERY_BLOCKED", error, stage="RECOVERY_QUOTE_RECHECK")
        return current
    # Unique IDs and a durable claim prevent repeat POSTs across processes or
    # crashes. Recovery orders/receipts are retained without overwriting intent.
    with bot.journal._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        count = db.execute("SELECT count(*) FROM testnet_exit_recoveries WHERE attempt_id=?",
                           (plan.attempt_id,)).fetchone()[0]
        if count != len(rows):
            return current
        parent = db.execute("SELECT state FROM testnet_pair_attempts WHERE attempt_id=?",
                            (plan.attempt_id,)).fetchone()
        if parent is None or parent[0] != PairExecutionState.RECOVERY_REQUIRED.value:
            return current
        db.execute("INSERT INTO testnet_exit_recoveries VALUES(?,?,?,?,?,NULL,?)",
                   (order.client_order_id, plan.attempt_id, len(rows) + 1, json.dumps(_order_data(order)),
                    _receipt_json(px), datetime.now(UTC).isoformat()))
    try:
        receipt = bot.perp.place_ioc_limit_order(order)
    except Exception as error:
        # Intent remains durable; next call queries this exact ID before any
        # further submission, even if the POST may not have reached Binance.
        bot._log_execution_failure(plan, "RECOVERY_SUBMISSION_UNCONFIRMED", error, stage="RECOVERY_SUBMISSION")
        return current
    if not bot.coordinator._matches(receipt, order) or not _terminal_quantity_confirmed(receipt):
        return current
    with bot.journal._connect() as db:
        db.execute("UPDATE testnet_exit_recoveries SET receipt_json=? WHERE client_order_id=?",
                   (_receipt_json(receipt), order.client_order_id))
    after = bot.perp.get_account_state(plan.symbol)
    if (receipt.executed_quantity == remaining and after.position_quantity == 0 and after.open_order_count == 0
            and after.one_way and after.single_asset and after.leverage == 1):
        bot.journal.update(plan.attempt_id, state=PairExecutionState.CLOSED, perp_exit=receipt,
                           failure_code=None, failure_reason=None)
    result = bot.journal.get(plan.attempt_id)
    assert result is not None
    return result
