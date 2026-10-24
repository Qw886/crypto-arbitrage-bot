"""TESTNET-only abort of a Spot entry whose Perp leg was never submitted.

Caller owns the journal process lock. Unknown or active orders are never
replaced. Native pre-entry inventory and current account state must agree.
No research decision, Funding, fee or profit calculation is performed here.
"""

import json
from dataclasses import replace
from decimal import ROUND_FLOOR, Decimal, localcontext
from enum import StrEnum
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
from crypto_arbitrage_bot.execution.spot_exit_recovery import (
    _baseline,
    _initialize,
    _inventory_consumed,
    _signature,
)
from crypto_arbitrage_bot.execution.types import (
    ExecutionMarket,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot

UNHEDGED_RECOVERY_CODES = frozenset(("SPOT_SUBMISSION_UNKNOWN", "SPOT_FLATTEN_INCOMPLETE",
                                   "SPOT_COMMISSION_UNKNOWN", "RECONCILIATION_UNRESOLVED"))


class UnhedgedEntryBlock(StrEnum):
    EVIDENCE_INVALID = "UNHEDGED_ENTRY_EVIDENCE_INVALID"
    ACCOUNT_CONFLICT = "UNHEDGED_ENTRY_ACCOUNT_CONFLICT"
    INVENTORY_CONFLICT = "UNHEDGED_ENTRY_INVENTORY_CONFLICT"
    RETRY_LIMIT = "UNHEDGED_ENTRY_RETRY_LIMIT"


def _blocked(bot: "TestnetBot", plan: PairOrderPlan, reason: UnhedgedEntryBlock) -> None:
    bot.logger.error(exact_json({"stage": "UNHEDGED_ENTRY_RECOVERY", "attempt_id": plan.attempt_id,
                                "symbol": plan.symbol, "reason": reason.value}))


def _compatible(cached: OrderReceipt | None, fresh: OrderReceipt) -> bool:
    if cached is None:
        return True
    if (cached.market, cached.symbol, cached.order_id, cached.client_order_id, cached.original_quantity) != (
            fresh.market, fresh.symbol, fresh.order_id, fresh.client_order_id, fresh.original_quantity):
        return False
    if cached.status in (OrderState.NEW, OrderState.PARTIALLY_FILLED):
        # An active order may finish later. Its identity cannot change and
        # confirmed quantities/commissions cannot move backwards.
        return (fresh.executed_quantity >= cached.executed_quantity
                and (cached.base_commission is None or fresh.base_commission is not None
                     and fresh.base_commission >= cached.base_commission))
    return (_signature(cached) == _signature(fresh)
            and (cached.base_commission is None or cached.base_commission == fresh.base_commission))


def _exits(bot: "TestnetBot", plan: PairOrderPlan, entry: OrderReceipt,
           original_order: IocLimitOrder | None) -> tuple[Decimal, int, OrderReceipt | None]:
    """Query exact durable IDs and count each confirmed terminal execution once."""
    target = entry.executed_quantity - (entry.base_commission or Decimal("0"))
    _initialize(bot)
    with bot.journal._connect() as db:
        rows = db.execute("SELECT * FROM testnet_spot_exit_recoveries WHERE attempt_id=? ORDER BY ordinal",
                          (plan.attempt_id,)).fetchall()
    if original_order is None:
        if rows:
            raise ValueError("repair without original flatten intent")
        return target, 0, None
    if (original_order.market is not ExecutionMarket.SPOT or original_order.symbol != plan.symbol
            or original_order.side is not Side.SELL or original_order.reduce_only
            or original_order.submit_before is not None or original_order.quantity != target):
        raise ValueError("invalid original flatten intent")
    original = bot.spot.get_order(plan.symbol, original_order.client_order_id)
    if not bot.coordinator._matches(original, original_order):
        raise ValueError("original flatten identity mismatch")
    saved = bot.journal.get(plan.attempt_id)
    if saved is None:
        raise ExecutionJournalError("unhedged parent missing")
    if not _compatible(saved.spot_exit, original):
        raise ValueError("original flatten changed")
    consumed = _inventory_consumed(original)
    for ordinal, row in enumerate(rows, start=1):
        data = json.loads(row["order_json"])
        original_copy = _receipt_from_json(row["original_exit_json"])
        if (row["schema_version"] != 1 or row["ordinal"] != ordinal
                or ordinal > bot.config.max_exit_recovery_attempts
                or row["client_order_id"] != f"{plan.attempt_id[:24]}-sxr{ordinal}"
                or data["client_order_id"] != row["client_order_id"]
                or data["market"] != "SPOT" or data["symbol"] != plan.symbol
                or data["side"] != "SELL" or data["reduce_only"] is not False
                or data.get("submit_before") is not None
                or type(data["quantity"]) is not str or type(data["limit_price"]) is not str
                or original_copy is None or _signature(original_copy) != _signature(original)
                or original_copy.base_commission != original.base_commission):
            raise ValueError("invalid persisted flatten repair")
        order = IocLimitOrder(ExecutionMarket.SPOT, plan.symbol, Side.SELL,
                             Decimal(data["quantity"]), Decimal(data["limit_price"]), row["client_order_id"])
        if order.quantity != target - consumed:
            raise ValueError("repair quantity differs from proven remaining inventory")
        receipt = bot.spot.get_order(plan.symbol, order.client_order_id)
        cached = _receipt_from_json(row["receipt_json"])
        if (not bot.coordinator._matches(receipt, order)
                or cached is not None and (_signature(cached) != _signature(receipt)
                    or cached.base_commission != receipt.base_commission)):
            raise ValueError("repair receipt identity changed")
        consumed += _inventory_consumed(receipt)
        with bot.journal._connect() as db:
            db.execute("UPDATE testnet_spot_exit_recoveries SET receipt_json=? WHERE client_order_id=?",
                       (_receipt_json(receipt), order.client_order_id))
    remaining = target - consumed
    if remaining < 0:
        raise ValueError("flatten consumed historical inventory")
    # Cache only confirmed terminal evidence; do not invent FILLED receipts.
    bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED,
        spot_entry=entry, spot_exit=original, failure_code=saved.failure_code, failure_reason=saved.failure_reason)
    return remaining, len(rows), original


@decimal_calculation
def recover_unhedged_entry(bot: "TestnetBot", plan: PairOrderPlan) -> PairExecutionResult:
    """Abort rather than open a late hedge; at most one bounded Spot SELL per call.

The initial flatten uses the original durable Spot exit ID. Confirmed terminal
partials share the existing repair journal/budget. Lost ACKs or checkpoints
only cause GETs of the same IDs. A restored inventory is FLATTENED, not CLOSED
or a successful strategy trade. Existing balance review still blocks re-entry.
"""
    current = bot.journal.get(plan.attempt_id)
    if current is None:
        raise ExecutionJournalError("unhedged recovery parent missing")
    if (current.state is not PairExecutionState.RECOVERY_REQUIRED
            or current.failure_code not in UNHEDGED_RECOVERY_CODES or not bot.config.balance_evidence_enabled
            or bot.config.max_exit_recovery_attempts == 0
            or current.perp_entry is not None or current.perp_exit is not None):
        return current
    intents = bot.journal.intents(plan.attempt_id)
    # Any Perp intent may have reached the exchange, even with no saved receipt.
    if any(i.market is ExecutionMarket.USDTM for i in intents):
        return current
    entries = [i for i in intents if i.client_order_id == plan.spot_entry.client_order_id]
    exits = [i for i in intents if i.client_order_id == plan.spot_exit.client_order_id]
    if (len(entries) != 1 or entries[0] != plan.spot_entry or len(exits) > 1
            or len(intents) != len(entries) + len(exits) or current.spot_exit is not None and not exits):
        _blocked(bot, plan, UnhedgedEntryBlock.EVIDENCE_INVALID)
        return current
    baseline = _baseline(bot, plan)
    if baseline is None:
        return current  # Never capture a replacement pre-entry frame.
    entry = bot.spot.get_order(plan.symbol, plan.spot_entry.client_order_id)
    try:
        if not bot.coordinator._matches(entry, plan.spot_entry) or not _compatible(current.spot_entry, entry):
            raise ValueError("entry receipt changed")
        _inventory_consumed(entry)  # Terminal state, quantity and commission proof.
        remaining, count, original = _exits(bot, plan, entry, exits[0] if exits else None)
    except (ValueError, TypeError, KeyError, ArithmeticError):
        _blocked(bot, plan, UnhedgedEntryBlock.EVIDENCE_INVALID)
        return current
    spot, perp = bot.spot.get_account_state(plan.symbol), bot.perp.get_account_state(plan.symbol)
    if (spot.market is not ExecutionMarket.SPOT or perp.market is not ExecutionMarket.USDTM
            or spot.symbol != plan.symbol or perp.symbol != plan.symbol
            or not spot.can_trade or not perp.can_trade or spot.open_order_count or perp.open_order_count
            or not perp.one_way or not perp.single_asset or perp.leverage != 1 or perp.position_quantity != 0):
        _blocked(bot, plan, UnhedgedEntryBlock.ACCOUNT_CONFLICT)
        return current
    if spot.base_available != baseline + remaining:
        _blocked(bot, plan, UnhedgedEntryBlock.INVENTORY_CONFLICT)
        return current
    if remaining == 0:
        bot.journal.update(plan.attempt_id, state=PairExecutionState.FLATTENED if entry.executed_quantity else
                           PairExecutionState.FAILED, spot_entry=entry, spot_exit=original,
            failure_code="UNHEDGED_ENTRY_ABORTED" if entry.executed_quantity else "UNHEDGED_ENTRY_NO_FILL",
            failure_reason="Perp never submitted; confirmed Spot inventory restored; balance review required")
        return bot.journal.get(plan.attempt_id) or current
    if original is not None and count >= bot.config.max_exit_recovery_attempts:
        _blocked(bot, plan, UnhedgedEntryBlock.RETRY_LIMIT)
        return current
    rules = bot.spot.get_symbol_rules(plan.symbol)
    markets = (TradeLeg.SPOT_EXIT.market,)
    depths = bot._collect_exit_depth(plan.symbol, markets=markets)
    price = bot._exit_limit(depths, TradeLeg.SPOT_EXIT, remaining, rules)
    with localcontext() as context:
        context.prec = 50
        price = ((price * (1 - bot.config.max_recovery_price_deviation)) / rules.tick_size).to_integral_value(
            rounding=ROUND_FLOOR) * rules.tick_size
    order = replace(plan.spot_exit, quantity=remaining, limit_price=price,
                    client_order_id=plan.spot_exit.client_order_id if original is None else
                    f"{plan.attempt_id[:24]}-sxr{count + 1}", reduce_only=False, submit_before=None)
    bot.spot._validate_order(order, rules)
    bot._require_exit_quality(depths, "EXIT_MARKET_EXPIRED_DURING_PREFLIGHT", markets=markets)
    if original is None:
        receipt, _ = bot.coordinator._submit(bot.spot, order)
        if receipt is not None:
            bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED,
                spot_entry=entry, spot_exit=receipt, failure_code="SPOT_FLATTEN_INCOMPLETE",
                failure_reason="Unhedged Spot flatten submitted; confirm terminal fill and inventory")
        return bot.journal.get(plan.attempt_id) or current
    with bot.journal._connect() as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("BEGIN IMMEDIATE")
        saved_count = db.execute("SELECT count(*) FROM testnet_spot_exit_recoveries WHERE attempt_id=?",
                                 (plan.attempt_id,)).fetchone()[0]
        parent = db.execute("SELECT state FROM testnet_pair_attempts WHERE attempt_id=?",
                            (plan.attempt_id,)).fetchone()
        if saved_count != count or parent is None or parent[0] != "RECOVERY_REQUIRED":
            return current
        db.execute("INSERT INTO testnet_spot_exit_recoveries VALUES(?,?,1,?,?,?,NULL,?)",
            (order.client_order_id, plan.attempt_id, count + 1, exact_json(_order_data(order)),
             _receipt_json(original), bot.clock().isoformat()))
    try:
        receipt = bot.spot.place_ioc_limit_order(order)
    except Exception as error:
        bot._log_execution_failure(plan, "UNHEDGED_FLATTEN_UNCONFIRMED", error, stage="UNHEDGED_ENTRY_RECOVERY")
        return current
    if not bot.coordinator._matches(receipt, order):
        return current
    try:
        _inventory_consumed(receipt)
    except ValueError:
        return current
    with bot.journal._connect() as db:
        db.execute("UPDATE testnet_spot_exit_recoveries SET receipt_json=? WHERE client_order_id=?",
                   (_receipt_json(receipt), order.client_order_id))
    return current  # Next invocation confirms current balances before FLATTENED.
