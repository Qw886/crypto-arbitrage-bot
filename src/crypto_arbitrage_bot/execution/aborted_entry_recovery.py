"""Native-evidence TESTNET abort of confirmed unequal/partial entry legs.

Only closes attributable inventory and negative Perp exposure. Unknown entry
or exit IDs are queried, never replaced. This is not a strategy decision or
financial accounting module. Caller owns the execution journal process lock.
"""

import json
from dataclasses import replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from enum import StrEnum
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.exit_recovery import _terminal_quantity_confirmed
from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    PairExecutionResult,
    PairExecutionState,
    PairOrderPlan,
    _order_data,
    _receipt_from_json,
    _receipt_json,
)
from crypto_arbitrage_bot.execution.spot_exit_recovery import (
    _baseline,
    _inventory_consumed,
    _signature,
)
from crypto_arbitrage_bot.execution.types import (
    ExecutionMarket,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.execution.unhedged_entry_recovery import _compatible, _exits
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot

_RECOVERABLE = frozenset(("PERP_SUBMISSION_UNKNOWN", "SPOT_SUBMISSION_UNKNOWN", "SPOT_COMMISSION_UNKNOWN",
                         "SPOT_FLATTEN_INCOMPLETE", "PERP_FLATTEN_INCOMPLETE", "RECONCILIATION_UNRESOLVED"))


class EntryAbortBlock(StrEnum):
    EVIDENCE_INVALID = "ENTRY_ABORT_EVIDENCE_INVALID"
    ACCOUNT_CONFLICT = "ENTRY_ABORT_ACCOUNT_CONFLICT"
    INVENTORY_CONFLICT = "ENTRY_ABORT_INVENTORY_CONFLICT"
    RETRY_LIMIT = "ENTRY_ABORT_RETRY_LIMIT"


def _blocked(bot: "TestnetBot", plan: PairOrderPlan, reason: EntryAbortBlock) -> None:
    bot.logger.error(exact_json({"stage": "ABORTED_ENTRY_RECOVERY", "attempt_id": plan.attempt_id,
                                "symbol": plan.symbol, "reason": reason.value}))


def _perp_exits(bot: "TestnetBot", plan: PairOrderPlan, quantity: Decimal,
                intent: IocLimitOrder | None) -> tuple[Decimal, int, OrderReceipt | None]:
    with bot.journal._connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS testnet_exit_recoveries (
            client_order_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL, order_json TEXT NOT NULL,
            original_exit_json TEXT NOT NULL, receipt_json TEXT,
            created_at TEXT NOT NULL, UNIQUE(attempt_id, ordinal))""")
        rows = db.execute("SELECT * FROM testnet_exit_recoveries WHERE attempt_id=? ORDER BY ordinal",
                          (plan.attempt_id,)).fetchall()
    if intent is None:
        if rows:
            raise ValueError("Perp repair without original exit intent")
        return quantity, 0, None
    if (intent.market is not ExecutionMarket.USDTM or intent.symbol != plan.symbol
            or intent.side is not Side.BUY or not intent.reduce_only
            or intent.submit_before is not None or intent.quantity != quantity):
        raise ValueError("Perp flatten intent mismatch")
    original = bot.perp.get_order(plan.symbol, intent.client_order_id)
    saved = bot.journal.get(plan.attempt_id)
    if saved is None:
        raise ExecutionJournalError("entry abort parent missing")
    if (not bot.coordinator._matches(original, intent) or not _terminal_quantity_confirmed(original)
            or not _compatible(saved.perp_exit, original)):
        raise ValueError("Perp original flatten is unresolved or changed")
    completed = original.executed_quantity
    for ordinal, row in enumerate(rows, start=1):
        data = json.loads(row["order_json"])
        saved_original = _receipt_from_json(row["original_exit_json"])
        if (row["ordinal"] != ordinal or ordinal > bot.config.max_exit_recovery_attempts
                or row["client_order_id"] != f"{plan.attempt_id[:24]}-rx{ordinal}"
                or data["client_order_id"] != row["client_order_id"]
                or data["market"] != "USDTM" or data["symbol"] != plan.symbol
                or data["side"] != "BUY" or data["reduce_only"] is not True
                or data.get("submit_before") is not None
                or type(data["quantity"]) is not str or type(data["limit_price"]) is not str
                or saved_original is None or _signature(saved_original) != _signature(original)):
            raise ValueError("Perp repair evidence mismatch")
        order = IocLimitOrder(ExecutionMarket.USDTM, plan.symbol, Side.BUY,
                             Decimal(data["quantity"]), Decimal(data["limit_price"]), row["client_order_id"], True)
        if order.quantity != quantity - completed:
            raise ValueError("Perp repair quantity mismatch")
        receipt = bot.perp.get_order(plan.symbol, order.client_order_id)
        cached = _receipt_from_json(row["receipt_json"])
        if (not bot.coordinator._matches(receipt, order) or not _terminal_quantity_confirmed(receipt)
                or not _compatible(cached, receipt)):
            raise ValueError("Perp repair unresolved or changed")
        completed += receipt.executed_quantity
        with bot.journal._connect() as db:
            db.execute("UPDATE testnet_exit_recoveries SET receipt_json=? WHERE client_order_id=?",
                       (_receipt_json(receipt), order.client_order_id))
    remaining = quantity - completed
    if remaining < 0:
        raise ValueError("Perp exit exceeds confirmed entry")
    bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED, perp_exit=original,
                       failure_code=saved.failure_code, failure_reason=saved.failure_reason)
    return remaining, len(rows), original


def _submit_repair(bot: "TestnetBot", plan: PairOrderPlan, order: IocLimitOrder,
                   original: OrderReceipt, count: int) -> None:
    # Only fixed table names/columns. Reuse the existing exit journals.
    spot = order.market is ExecutionMarket.SPOT
    table = "testnet_spot_exit_recoveries" if spot else "testnet_exit_recoveries"
    with bot.journal._connect() as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute(f"SELECT count(*) FROM {table} WHERE attempt_id=?", (plan.attempt_id,)).fetchone()[0]
        parent = db.execute("SELECT state FROM testnet_pair_attempts WHERE attempt_id=?",
                            (plan.attempt_id,)).fetchone()
        if rows != count or parent is None or parent[0] != "RECOVERY_REQUIRED":
            return
        if spot:
            db.execute("INSERT INTO testnet_spot_exit_recoveries VALUES(?,?,1,?,?,?,NULL,?)",
                (order.client_order_id, plan.attempt_id, count + 1, exact_json(_order_data(order)),
                 _receipt_json(original), bot.clock().isoformat()))
        else:
            db.execute("INSERT INTO testnet_exit_recoveries VALUES(?,?,?,?,?,NULL,?)",
                (order.client_order_id, plan.attempt_id, count + 1, exact_json(_order_data(order)),
                 _receipt_json(original), bot.clock().isoformat()))
    client = bot.spot if spot else bot.perp
    try:
        receipt = client.place_ioc_limit_order(order)
    except Exception as error:
        bot._log_execution_failure(plan, "ENTRY_ABORT_SUBMISSION_UNCONFIRMED", error, stage="ABORTED_ENTRY_RECOVERY")
        return
    if not bot.coordinator._matches(receipt, order):
        return
    if spot:
        try:
            _inventory_consumed(receipt)
        except ValueError:
            return
    elif not _terminal_quantity_confirmed(receipt):
        return
    with bot.journal._connect() as db:
        db.execute(f"UPDATE {table} SET receipt_json=? WHERE client_order_id=?",
                   (_receipt_json(receipt), order.client_order_id))


@decimal_calculation
def recover_aborted_entry(bot: "TestnetBot", plan: PairOrderPlan) -> PairExecutionResult | None:
    """None means the existing normal hedge/exit workflow owns this attempt.

Both entry IDs must be confirmed terminal. Known partials are flattened;
no hedge is completed by sending another SELL. At most one new exit per call.
"""
    current = bot.journal.get(plan.attempt_id)
    if current is None:
        raise ExecutionJournalError("entry abort parent missing")
    if (current.state is not PairExecutionState.RECOVERY_REQUIRED or current.failure_code not in _RECOVERABLE
            or not bot.config.balance_evidence_enabled or bot.config.max_exit_recovery_attempts == 0):
        return None
    # Keep already-proven full neutral entries on their original holding/exit path.
    if (current.spot_entry is not None and current.perp_entry is not None
            and current.spot_entry.status is current.perp_entry.status is OrderState.FILLED
            and current.spot_entry.executed_quantity == current.perp_entry.executed_quantity == plan.spot_entry.quantity
            and current.spot_entry.base_commission == 0):
        return None
    intents = bot.journal.intents(plan.attempt_id)
    pe = [i for i in intents if i.client_order_id == plan.perp_entry.client_order_id]
    if not pe:
        return None  # The separate Spot-only recovery owns a never-submitted hedge.
    se = [i for i in intents if i.client_order_id == plan.spot_entry.client_order_id]
    sx = [i for i in intents if i.client_order_id == plan.spot_exit.client_order_id]
    px = [i for i in intents if i.client_order_id == plan.perp_exit.client_order_id]
    if (len(pe) != 1 or pe[0] != plan.perp_entry or pe[0].reduce_only or len(se) != 1
            or se[0] != plan.spot_entry or se[0].reduce_only or len(sx) > 1 or len(px) > 1
            or len(intents) != len(se) + len(pe) + len(sx) + len(px)
            or current.spot_exit is not None and not sx or current.perp_exit is not None and not px):
        _blocked(bot, plan, EntryAbortBlock.EVIDENCE_INVALID)
        return current
    baseline = _baseline(bot, plan)
    if baseline is None:
        return current
    spot_entry = bot.spot.get_order(plan.symbol, se[0].client_order_id)
    perp_entry = bot.perp.get_order(plan.symbol, pe[0].client_order_id)
    try:
        if (not bot.coordinator._matches(spot_entry, se[0]) or not _compatible(current.spot_entry, spot_entry)
                or not bot.coordinator._matches(perp_entry, pe[0]) or not _compatible(current.perp_entry, perp_entry)
                or not _terminal_quantity_confirmed(perp_entry)):
            raise ValueError("entry evidence unresolved or changed")
        _inventory_consumed(spot_entry)
        if (spot_entry.status is perp_entry.status is OrderState.FILLED
                and spot_entry.executed_quantity == perp_entry.executed_quantity == plan.spot_entry.quantity
                and spot_entry.base_commission == 0):
            return None  # Lost ACK of a full hedge is reconciled normally, not aborted.
        spot_remaining, spot_count, original_spot = _exits(bot, plan, spot_entry, sx[0] if sx else None)
        perp_remaining, perp_count, original_perp = _perp_exits(
            bot, plan, perp_entry.executed_quantity, px[0] if px else None)
    except (ValueError, TypeError, KeyError, ArithmeticError):
        _blocked(bot, plan, EntryAbortBlock.EVIDENCE_INVALID)
        return current
    spot, perp = bot.spot.get_account_state(plan.symbol), bot.perp.get_account_state(plan.symbol)
    if (spot.market is not ExecutionMarket.SPOT or perp.market is not ExecutionMarket.USDTM
            or spot.symbol != plan.symbol or perp.symbol != plan.symbol
            or not spot.can_trade or not perp.can_trade or spot.open_order_count or perp.open_order_count
            or not perp.one_way or not perp.single_asset or perp.leverage != 1
            or perp.position_quantity != -perp_remaining):
        _blocked(bot, plan, EntryAbortBlock.ACCOUNT_CONFLICT)
        return current
    if spot.base_available != baseline + spot_remaining:
        _blocked(bot, plan, EntryAbortBlock.INVENTORY_CONFLICT)
        return current
    bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED,
        spot_entry=spot_entry, perp_entry=perp_entry, failure_code=current.failure_code,
        failure_reason=current.failure_reason)
    if spot_remaining == perp_remaining == 0:
        bot.journal.update(plan.attempt_id, state=PairExecutionState.FLATTENED if
                           spot_entry.executed_quantity or perp_entry.executed_quantity else PairExecutionState.FAILED,
            failure_code="ABORTED_ENTRY_RESTORED", failure_reason="Both confirmed entry exposures restored; balance review required")
        return bot.journal.get(plan.attempt_id) or current
    is_spot = spot_remaining > 0
    original = original_spot if is_spot else original_perp
    count = spot_count if is_spot else perp_count
    if original is not None and count >= bot.config.max_exit_recovery_attempts:
        _blocked(bot, plan, EntryAbortBlock.RETRY_LIMIT)
        return current
    client = bot.spot if is_spot else bot.perp
    quantity = spot_remaining if is_spot else perp_remaining
    leg = TradeLeg.SPOT_EXIT if is_spot else TradeLeg.PERP_EXIT
    rules = client.get_symbol_rules(plan.symbol)
    markets = (leg.market,)
    depths = bot._collect_exit_depth(plan.symbol, markets=markets)
    price = bot._exit_limit(depths, leg, quantity, rules)
    with localcontext() as context:
        context.prec = 50
        direction = Decimal("-1") if is_spot else Decimal("1")
        price = ((price * (1 + direction * bot.config.max_recovery_price_deviation)) / rules.tick_size).to_integral_value(
            rounding=ROUND_FLOOR if is_spot else ROUND_CEILING) * rules.tick_size
    prototype = plan.spot_exit if is_spot else plan.perp_exit
    suffix = "sxr" if is_spot else "rx"
    order = replace(prototype, quantity=quantity, limit_price=price, reduce_only=not is_spot, submit_before=None,
                    client_order_id=prototype.client_order_id if original is None else
                    f"{plan.attempt_id[:24]}-{suffix}{count + 1}")
    client._validate_order(order, rules)
    bot._require_exit_quality(depths, "EXIT_MARKET_EXPIRED_DURING_PREFLIGHT", markets=markets)
    if original is not None:
        _submit_repair(bot, plan, order, original, count)
    else:
        receipt, _ = bot.coordinator._submit(client, order)
        if receipt is not None:
            bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED,
                spot_exit=receipt if is_spot else None, perp_exit=None if is_spot else receipt,
                failure_code="SPOT_FLATTEN_INCOMPLETE" if is_spot else "PERP_FLATTEN_INCOMPLETE",
                failure_reason="Entry abort exit submitted; confirm fills and account restoration")
    return bot.journal.get(plan.attempt_id) or current
