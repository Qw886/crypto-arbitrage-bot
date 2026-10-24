"""Bounded TESTNET recovery of a confirmed terminal Spot exit.

Original inventory is protected by the immutable native pre-entry frame.
Unknown orders are queried at their persisted IDs, never blindly resent.
PnL, fees and research decisions are not computed in this module.
"""

import json
from dataclasses import dataclass, replace
from decimal import ROUND_FLOOR, Decimal, localcontext
from enum import StrEnum
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.balance_frame_store import BalanceFrameStore
from crypto_arbitrage_bot.execution.balance_ledger import frame_digest
from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    PairExecutionResult,
    PairExecutionState,
    PairOrderPlan,
    _order_data,
    _receipt_from_json,
    _receipt_json,
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

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot

_TERMINAL = frozenset((OrderState.FILLED, OrderState.EXPIRED, OrderState.EXPIRED_IN_MATCH,
                       OrderState.CANCELED, OrderState.REJECTED))
_RECOVERABLE = frozenset(('SPOT_EXIT_INCOMPLETE', 'PERP_EXIT_INCOMPLETE', 'RECONCILIATION_UNRESOLVED'))


class SpotRecoveryBlock(StrEnum):
    INVALID_EVIDENCE = 'SPOT_RECOVERY_EVIDENCE_INVALID'
    MISSING_PREENTRY_FRAME = 'SPOT_RECOVERY_PREENTRY_EVIDENCE_REQUIRED'
    INVENTORY_CONFLICT = 'SPOT_RECOVERY_INVENTORY_CONFLICT'
    ACCOUNT_CONFLICT = 'SPOT_RECOVERY_ACCOUNT_CONFLICT'


@dataclass(frozen=True)
class SpotExitProof:
    original: OrderReceipt
    target_quantity: Decimal
    remaining_quantity: Decimal
    recovery_count: int


def _block(bot: "TestnetBot", plan: PairOrderPlan, reason: SpotRecoveryBlock) -> None:
    bot.logger.error(exact_json({'stage': 'SPOT_EXIT_RECOVERY', 'attempt_id': plan.attempt_id,
                                'reason': reason.value}))


def _initialize(bot: "TestnetBot") -> None:
    with bot.journal._connect() as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('''CREATE TABLE IF NOT EXISTS testnet_spot_exit_recoveries (
            client_order_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES testnet_pair_attempts(attempt_id),
            schema_version INTEGER NOT NULL CHECK(schema_version=1), ordinal INTEGER NOT NULL,
            order_json TEXT NOT NULL, original_exit_json TEXT NOT NULL, receipt_json TEXT,
            created_at TEXT NOT NULL, UNIQUE(attempt_id, ordinal))''')


def _signature(receipt: OrderReceipt) -> tuple[object, ...]:
    return (receipt.market, receipt.symbol, receipt.order_id, receipt.client_order_id,
            receipt.status, receipt.original_quantity, receipt.executed_quantity)


def _inventory_consumed(receipt: OrderReceipt) -> Decimal:
    if (receipt.status not in _TERMINAL
            or receipt.status is OrderState.FILLED and receipt.executed_quantity != receipt.original_quantity
            or receipt.status is OrderState.REJECTED and receipt.executed_quantity != 0):
        raise ValueError('terminal execution evidence required')
    if receipt.executed_quantity == 0:
        if receipt.base_commission not in (None, Decimal('0')):
            raise ValueError('commission without execution')
        return Decimal('0')  # No execution, hence no execution commission.
    if receipt.base_commission is None:
        raise ValueError('base commission not confirmed')
    return receipt.executed_quantity + receipt.base_commission


def read_spot_exit_proof(bot: "TestnetBot", plan: PairOrderPlan) -> SpotExitProof | None:
    """Fixed order GETs; checkpoint receipts are cached only after validation."""
    current = bot.journal.get(plan.attempt_id)
    if current is None or current.spot_entry is None or current.spot_exit is None:
        return None
    entry = current.spot_entry
    if (entry.status is not OrderState.FILLED or entry.executed_quantity != plan.spot_entry.quantity
            or entry.base_commission is None):
        return None
    target = entry.executed_quantity - entry.base_commission
    intents = [i for i in bot.journal.intents(plan.attempt_id) if i.client_order_id == plan.spot_exit.client_order_id]
    if (not bot.coordinator._matches(entry, plan.spot_entry) or len(intents) != 1
            or intents[0].market is not ExecutionMarket.SPOT or intents[0].symbol != plan.symbol
            or intents[0].side is not Side.SELL or intents[0].reduce_only
            or intents[0].quantity != target):
        _block(bot, plan, SpotRecoveryBlock.INVALID_EVIDENCE)
        return None
    original = bot.spot.get_order(plan.symbol, plan.spot_exit.client_order_id)
    if (not bot.coordinator._matches(original, intents[0])
            or _signature(original) != _signature(current.spot_exit)):
        _block(bot, plan, SpotRecoveryBlock.INVALID_EVIDENCE)
        return None
    _initialize(bot)
    with bot.journal._connect() as db:
        rows = db.execute('SELECT * FROM testnet_spot_exit_recoveries WHERE attempt_id=? ORDER BY ordinal',
                          (plan.attempt_id,)).fetchall()
    try:
        consumed = _inventory_consumed(original)
        for ordinal, row in enumerate(rows, start=1):
            data = json.loads(row['order_json'])
            saved_original = _receipt_from_json(row['original_exit_json'])
            if (row['schema_version'] != 1 or row['ordinal'] != ordinal
                    or ordinal > bot.config.max_exit_recovery_attempts
                    or row['client_order_id'] != f'{plan.attempt_id[:24]}-sxr{ordinal}'
                    or data['client_order_id'] != row['client_order_id']
                    or data['market'] != 'SPOT' or data['symbol'] != plan.symbol
                    or data['side'] != 'SELL' or data['reduce_only'] is not False
                    or type(data['quantity']) is not str or type(data['limit_price']) is not str
                    or data.get('submit_before') is not None or saved_original is None
                    or _signature(saved_original) != _signature(original)
                    or saved_original.base_commission != original.base_commission):
                raise ValueError('invalid persisted Spot recovery identity')
            order = IocLimitOrder(ExecutionMarket.SPOT, plan.symbol, Side.SELL, Decimal(data['quantity']),
                                  Decimal(data['limit_price']), row['client_order_id'])
            if order.quantity != target - consumed:
                raise ValueError('invalid persisted Spot recovery quantity')
            receipt = _receipt_from_json(row['receipt_json'])
            if receipt is None:
                receipt = bot.spot.get_order(plan.symbol, order.client_order_id)
                if not bot.coordinator._matches(receipt, order):
                    raise ValueError('queried Spot recovery identity mismatch')
                _inventory_consumed(receipt)
                with bot.journal._connect() as db:
                    db.execute('UPDATE testnet_spot_exit_recoveries SET receipt_json=? WHERE client_order_id=?',
                               (_receipt_json(receipt), order.client_order_id))
            if not bot.coordinator._matches(receipt, order):
                raise ValueError('cached Spot recovery identity mismatch')
            consumed += _inventory_consumed(receipt)
        remaining = target - consumed
        if remaining < 0:
            raise ValueError('inventory overconsumed')
        return SpotExitProof(original, target, remaining, len(rows))
    except (ValueError, TypeError, KeyError, ArithmeticError):
        _block(bot, plan, SpotRecoveryBlock.INVALID_EVIDENCE)
        return None


def _baseline(bot: "TestnetBot", plan: PairOrderPlan) -> Decimal | None:
    before = BalanceFrameStore(bot.balance_directory).get(plan.attempt_id, plan.symbol, plan.spot_entry.quantity)
    markers = [i for i in BalanceCheckRepository(bot.journal).before_intents() if i.attempt_id == plan.attempt_id]
    if (not before.success or before.frame is None or len(markers) != 1
            or markers[0].symbol != plan.symbol or markers[0].quantity != plan.spot_entry.quantity
            or frame_digest(markers[0], before.frame.snapshots) != markers[0].frame_sha256):
        _block(bot, plan, SpotRecoveryBlock.MISSING_PREENTRY_FRAME)
        return None
    asset = plan.symbol[:-4]
    amount = next((b.amount for s in before.frame.snapshots if s.market is ExecutionMarket.SPOT
                   for b in s.balances if b.asset == asset), None)
    if amount is None:
        _block(bot, plan, SpotRecoveryBlock.MISSING_PREENTRY_FRAME)
    return amount


def confirmed_spot_exit_inventory(bot: "TestnetBot", plan: PairOrderPlan) -> bool:
    """A partial Spot receipt alone cannot authorize Perp repair/closure."""
    baseline = _baseline(bot, plan)
    if baseline is None:
        return False
    proof = read_spot_exit_proof(bot, plan)
    if proof is None or proof.remaining_quantity != 0:
        return False
    account = bot.spot.get_account_state(plan.symbol)
    if not account.can_trade or account.open_order_count:
        _block(bot, plan, SpotRecoveryBlock.ACCOUNT_CONFLICT)
        return False
    if account.base_available != baseline:
        _block(bot, plan, SpotRecoveryBlock.INVENTORY_CONFLICT)
        return False
    return True


def recover_spot_exit(bot: "TestnetBot", plan: PairOrderPlan) -> PairExecutionResult:
    """Caller owns process lock. At most one new Spot order per invocation.

Once Spot exits are confirmed, only the original Perp exit is submitted; an
existing Perp intent is queried instead. Its partial/zero terminal residual
is then handled by the separate original bounded Perp recovery workflow.
"""
    current = bot.journal.get(plan.attempt_id)
    if current is None:
        raise ExecutionJournalError('Spot recovery parent missing')
    if (current.state is not PairExecutionState.RECOVERY_REQUIRED or not bot.config.balance_evidence_enabled
            or bot.config.max_exit_recovery_attempts == 0 or current.failure_code not in _RECOVERABLE
            or current.spot_entry is None or current.perp_entry is None or current.spot_exit is None
            or current.perp_entry.status is not OrderState.FILLED
            or current.perp_entry.executed_quantity != plan.perp_entry.quantity
            or not bot.coordinator._matches(current.perp_entry, plan.perp_entry)):
        return current
    # The legacy fully filled Spot path belongs to the existing Perp repair.
    with bot.journal._connect() as db:
        has_rows = db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_spot_exit_recoveries'").fetchone()
        has_rows = has_rows and db.execute('SELECT 1 FROM testnet_spot_exit_recoveries WHERE attempt_id=?',
                                           (plan.attempt_id,)).fetchone()
    if current.spot_exit.status is OrderState.FILLED and not has_rows:
        return current
    baseline = _baseline(bot, plan)
    if baseline is None:
        return current
    proof = read_spot_exit_proof(bot, plan)
    if proof is None:
        return current
    spot, perp = bot.spot.get_account_state(plan.symbol), bot.perp.get_account_state(plan.symbol)
    if (not spot.can_trade or spot.open_order_count or not perp.can_trade or perp.open_order_count
            or not perp.one_way or not perp.single_asset or perp.leverage != 1):
        _block(bot, plan, SpotRecoveryBlock.ACCOUNT_CONFLICT)
        return current
    if spot.base_available != baseline + proof.remaining_quantity:
        _block(bot, plan, SpotRecoveryBlock.INVENTORY_CONFLICT)
        return current
    perp_intents = [i for i in bot.journal.intents(plan.attempt_id) if i.client_order_id == plan.perp_exit.client_order_id]
    if perp_intents:
        if proof.remaining_quantity != 0 or len(perp_intents) != 1:
            return current
        intent = perp_intents[0]
        if (intent.market is not ExecutionMarket.USDTM or intent.symbol != plan.symbol
                or intent.side is not Side.BUY or not intent.reduce_only
                or intent.quantity != current.perp_entry.executed_quantity):
            _block(bot, plan, SpotRecoveryBlock.INVALID_EVIDENCE)
            return current
        receipt = bot.perp.get_order(plan.symbol, plan.perp_exit.client_order_id)
        if not bot.coordinator._matches(receipt, perp_intents[0]) or receipt.status not in _TERMINAL:
            return current
        if receipt.status is OrderState.FILLED and receipt.executed_quantity == current.perp_entry.executed_quantity:
            if perp.position_quantity == 0:
                bot.journal.update(plan.attempt_id, state=PairExecutionState.CLOSED, perp_exit=receipt,
                                   failure_code=None, failure_reason=None)
        else:
            bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED, perp_exit=receipt,
                               failure_code='PERP_EXIT_INCOMPLETE', failure_reason='Confirmed original Perp exit needs bounded repair')
        result = bot.journal.get(plan.attempt_id)
        assert result is not None
        return result
    if current.perp_exit is not None or perp.position_quantity != -current.perp_entry.executed_quantity:
        _block(bot, plan, SpotRecoveryBlock.ACCOUNT_CONFLICT)
        return current
    market = ExecutionMarket.SPOT if proof.remaining_quantity else ExecutionMarket.USDTM
    client = bot.spot if market is ExecutionMarket.SPOT else bot.perp
    if market is ExecutionMarket.SPOT and proof.recovery_count >= bot.config.max_exit_recovery_attempts:
        return current
    quantity = proof.remaining_quantity if market is ExecutionMarket.SPOT else current.perp_entry.executed_quantity
    rules = client.get_symbol_rules(plan.symbol)
    leg = TradeLeg.SPOT_EXIT if market is ExecutionMarket.SPOT else TradeLeg.PERP_EXIT
    markets = (leg.market,)
    depths = bot._collect_exit_depth(plan.symbol, markets=markets)
    price = bot._exit_limit(depths, leg, quantity, rules)
    if market is ExecutionMarket.SPOT:
        with localcontext() as context:
            context.prec = 50
            price = ((price * (1 - bot.config.max_recovery_price_deviation)) / rules.tick_size).to_integral_value(
                rounding=ROUND_FLOOR) * rules.tick_size
        order = replace(plan.spot_exit, quantity=quantity, limit_price=price,
                        client_order_id=f'{plan.attempt_id[:24]}-sxr{proof.recovery_count + 1}')
    else:
        order = replace(plan.perp_exit, quantity=quantity, limit_price=price, reduce_only=True)
    client._validate_order(order, rules)
    bot._require_exit_quality(depths, 'EXIT_MARKET_EXPIRED_DURING_PREFLIGHT', markets=markets)
    if market is ExecutionMarket.USDTM:
        # _submit persists this original ID before POST. On restart, its intent
        # above prevents a second POST even if the receipt/checkpoint was lost.
        exit_receipt, _ = bot.coordinator._submit(bot.perp, order)
        if exit_receipt is not None:
            bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED, perp_exit=exit_receipt,
                failure_code='PERP_EXIT_INCOMPLETE', failure_reason='Confirming Perp exit after Spot recovery')
        return bot.journal.get(plan.attempt_id) or current
    with bot.journal._connect() as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('BEGIN IMMEDIATE')
        count = db.execute('SELECT count(*) FROM testnet_spot_exit_recoveries WHERE attempt_id=?',
                           (plan.attempt_id,)).fetchone()[0]
        parent = db.execute('SELECT state FROM testnet_pair_attempts WHERE attempt_id=?', (plan.attempt_id,)).fetchone()
        if count != proof.recovery_count or parent is None or parent[0] != 'RECOVERY_REQUIRED':
            return current
        db.execute('INSERT INTO testnet_spot_exit_recoveries VALUES(?,?,1,?,?,?,NULL,?)',
            (order.client_order_id, plan.attempt_id, count + 1, exact_json(_order_data(order)),
             _receipt_json(proof.original), bot.clock().isoformat()))
    try:
        receipt = client.place_ioc_limit_order(order)
    except Exception as error:
        bot._log_execution_failure(plan, 'SPOT_RECOVERY_SUBMISSION_UNCONFIRMED', error, stage='SPOT_RECOVERY_SUBMISSION')
        return current
    if not bot.coordinator._matches(receipt, order):
        return current
    try:
        _inventory_consumed(receipt)
    except ValueError:
        return current  # Query the same ID later for terminal fills/commission.
    with bot.journal._connect() as db:
        db.execute('UPDATE testnet_spot_exit_recoveries SET receipt_json=? WHERE client_order_id=?',
                   (_receipt_json(receipt), order.client_order_id))
    return current  # Next round verifies balances before closing any Perp.
