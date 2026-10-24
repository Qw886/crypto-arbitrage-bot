"""Offline terminal-entry abort: protect old inventory and never repeat unknown POSTs."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_trade_accounting import AT

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import balance_frame_store
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.balance_ledger import balance_entry_blocker
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.spot_exit_recovery import _receipt_json
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationStatus,
    reconcile_closed_attempt,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket, OrderState
from crypto_arbitrage_bot.execution.unhedged_entry_recovery import recover_unhedged_entry
from crypto_arbitrage_bot.models.market import Market

D = Decimal


@pytest.fixture
def unhedged(tmp_path, monkeypatch, request):
    symbol = getattr(request, "param", "ETHUSDT")
    quantity = D(".002") if symbol == "BTCUSDT" else D(".01")
    settings = load_settings(Path("docs/testnet-bot.toml"))
    settings = settings.model_copy(update={"testnet": settings.testnet.model_copy(update={
        "balance_evidence_enabled": True, "balance_evidence_directory": tmp_path / "frames"})})
    bot, transport = demo_bot(settings, tmp_path)
    transport.base[symbol] = D("1")
    before = (BalanceSnapshot(ExecutionMarket.SPOT, AT, AT,
        (AssetBalance("USDT", D("10000")), AssetBalance(symbol[:-4], D("1")))),
        BalanceSnapshot(ExecutionMarket.USDTM, AT, AT, (AssetBalance("USDT", D("10000")),)))
    monkeypatch.setattr(balance_frame_store, "capture_balance_snapshot",
                        lambda client, **kw: before[0 if client is bot.spot else 1])
    plan = bot._plan(symbol, quantity, "unhedged-" + symbol)
    deadline = bot.clock() + timedelta(seconds=60)
    plan = replace(plan, spot_entry=replace(plan.spot_entry, submit_before=deadline),
                   perp_entry=replace(plan.perp_entry, submit_before=deadline))
    submit = bot.spot.place_ioc_limit_order
    def lost_ack(order):
        submit(order)
        raise TimeoutError("mock entry ACK lost")
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", lost_ack)
    assert bot.coordinator.open_pair(plan).state is State.RECOVERY_REQUIRED
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", submit)
    assert len(transport.orders) == 1 and transport.positions[symbol] == 0
    return bot, transport, plan


def finish(bot, plan):
    for _ in range(5):
        bot._managed_exits()
        if bot.journal.get(plan.attempt_id).state is State.FLATTENED:
            break


def repairs(bot):
    with bot.journal._connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_spot_exit_recoveries'").fetchone():
            return []
        return db.execute("SELECT * FROM testnet_spot_exit_recoveries ORDER BY ordinal").fetchall()


def partial_submission(bot, transport, monkeypatch, amount):
    submit = bot.spot.place_ioc_limit_order
    def partial(order):
        receipt = submit(order)
        transport.base[order.symbol] += order.quantity - amount
        payload = transport.orders[order.client_order_id]
        payload.update(status="EXPIRED", executedQty=str(amount),
                       cummulativeQuoteQty=str(amount * order.limit_price))
        payload["fills"][0]["qty"] = str(amount)
        return replace(receipt, status=OrderState.EXPIRED, executed_quantity=amount,
                       cumulative_quote_quantity=amount * order.limit_price)
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", partial)
    return submit


@pytest.mark.parametrize("unhedged", ["BTCUSDT", "ETHUSDT"], indirect=True)
@pytest.mark.parametrize("restart", [False, True])
def test_original_unknown_entry_is_flattened_without_opening_late_perp(unhedged, restart):
    bot, transport, plan = unhedged
    if restart:
        bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    pair = bot.journal.get(plan.attempt_id)
    assert pair.state is State.FLATTENED and pair.failure_code == "UNHEDGED_ENTRY_ABORTED"
    assert pair.spot_entry.executed_quantity == pair.spot_exit.executed_quantity == plan.spot_entry.quantity
    assert pair.perp_entry is pair.perp_exit is None
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0
    assert len(transport.orders) == 2 and not repairs(bot)
    assert len(bot.journal.intents(plan.attempt_id)) == 2
    assert not any(m == "POST" and "demo-fapi" in url for m, url in transport.requests)
    requests = len(transport.requests)
    finish(bot, plan)
    assert len(transport.requests) == requests  # Terminal attempt not executed again.
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"
    accounting = reconcile_closed_attempt(bot, plan.attempt_id)
    assert accounting.status is ReconciliationStatus.INCOMPLETE and accounting.calculation is None


def test_entry_deadline_cannot_block_abort_or_create_a_late_hedge(unhedged):
    bot, transport, plan = unhedged
    now = bot.clock()
    bot.clock = lambda: now + timedelta(days=1)
    transport.clock = bot.collector.clock = bot.clock
    assert bot.clock() > plan.perp_entry.submit_before
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert len(transport.orders) == 2


def test_flatten_ack_loss_queries_original_id_after_restart(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    submit, seen = bot.spot.place_ioc_limit_order, []
    def lost(order):
        seen.append(order.client_order_id)
        submit(order)
        raise TimeoutError("mock exit ACK lost")
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", lost)
    recover_unhedged_entry(bot, plan)
    assert bot.journal.get(plan.attempt_id).spot_exit is None and len(transport.orders) == 2
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert seen == [plan.spot_exit.client_order_id]
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 2


def test_partial_flatten_uses_only_remaining_inventory(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    submit = partial_submission(bot, transport, monkeypatch, D(".004"))
    recover_unhedged_entry(bot, plan)
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", submit)
    finish(bot, plan)
    pair = bot.journal.get(plan.attempt_id)
    assert pair.state is State.FLATTENED and pair.spot_exit.status is OrderState.EXPIRED
    assert pair.spot_exit.executed_quantity == D(".004")
    assert len(transport.orders) == 3 and transport.base[plan.symbol] == 1
    assert D(json.loads(repairs(bot)[0]["order_json"])["quantity"]) == D(".006")


def test_repair_ack_loss_is_not_resubmitted(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    submit = partial_submission(bot, transport, monkeypatch, D(".004"))
    recover_unhedged_entry(bot, plan)
    seen = []
    def lost(order):
        seen.append(order.client_order_id)
        submit(order)
        raise TimeoutError("mock repair ACK lost")
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", lost)
    recover_unhedged_entry(bot, plan)
    assert len(repairs(bot)) == 1 and repairs(bot)[0]["receipt_json"] is None
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert len(seen) == 1 and len(transport.orders) == 3


def test_multiple_terminal_partials_keep_bounded_retry_budget_on_restart(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    partial_submission(bot, transport, monkeypatch, D(".002"))
    for _ in range(5):
        recover_unhedged_entry(bot, plan)
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    assert len(repairs(bot)) == bot.config.max_exit_recovery_attempts == 2
    assert len(transport.orders) == 4 and transport.base[plan.symbol] == D("1.004")


@pytest.mark.parametrize("status", [OrderState.NEW, OrderState.PARTIALLY_FILLED])
def test_live_entry_never_authorizes_flatten(unhedged, status):
    bot, transport, plan = unhedged
    transport.orders[plan.spot_entry.client_order_id]["status"] = status.value
    finish(bot, plan)
    assert len(transport.orders) == 1 and not repairs(bot)
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


def test_unknown_order_query_never_guesses_inventory_or_resubmits(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    monkeypatch.setattr(bot.spot, "get_order", Mock(side_effect=TimeoutError("offline query timeout")))
    finish(bot, plan)
    assert len(transport.orders) == 1 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("change", ["missing", "invalid", "balance"])
def test_missing_or_tampered_native_preentry_frame_never_trades(unhedged, change):
    bot, transport, plan = unhedged
    path = bot.balance_directory / (plan.attempt_id + ".before.json")
    if change == "missing":
        path.unlink()
    elif change == "invalid":
        path.write_text("{", encoding="utf-8")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["snapshots"][0]["balances"][1]["amount"] = "0"
        path.write_text(json.dumps(data), encoding="utf-8")
    count = len(transport.requests)
    recover_unhedged_entry(bot, plan)
    assert len(transport.requests) == count and len(transport.orders) == 1


@pytest.mark.parametrize("delta", [".001", "-.001"])
def test_inventory_conflict_protects_existing_holdings(unhedged, delta):
    bot, transport, plan = unhedged
    transport.base[plan.symbol] += D(delta)
    finish(bot, plan)
    assert len(transport.orders) == 1 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("field,value", [("position_quantity", D("-.01")), ("position_quantity", D(".01")),
    ("open_order_count", 1), ("can_trade", False), ("one_way", False), ("single_asset", False),
    ("leverage", D("2")), ("symbol", "BTCUSDT"), ("market", ExecutionMarket.SPOT)])
def test_perp_account_conflict_blocks_spot_sale(unhedged, monkeypatch, field, value):
    bot, transport, plan = unhedged
    account = bot.perp.get_account_state(plan.symbol)
    monkeypatch.setattr(bot.perp, "get_account_state", lambda *a: replace(account, **{field: value}))
    finish(bot, plan)
    assert len(transport.orders) == 1


@pytest.mark.parametrize("field,value", [("open_order_count", 1), ("can_trade", False),
                                       ("symbol", "BTCUSDT"), ("market", ExecutionMarket.USDTM)])
def test_spot_account_conflict_blocks_sale(unhedged, monkeypatch, field, value):
    bot, transport, plan = unhedged
    account = bot.spot.get_account_state(plan.symbol)
    monkeypatch.setattr(bot.spot, "get_account_state", lambda *a: replace(account, **{field: value}))
    finish(bot, plan)
    assert len(transport.orders) == 1


@pytest.mark.parametrize("code", [None, "MANUAL_REVIEW_REQUIRED", "UNRECOGNIZED_FAILURE"])
def test_manual_state_never_grants_trade_permission(unhedged, code):
    bot, transport, plan = unhedged
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, failure_code=code)
    count = len(transport.requests)
    finish(bot, plan)
    assert len(transport.requests) == count and len(transport.orders) == 1


@pytest.mark.parametrize("setting,value", [("balance_evidence_enabled", False), ("max_exit_recovery_attempts", 0)])
def test_disabled_native_recovery_preserves_old_behavior(unhedged, setting, value):
    bot, transport, plan = unhedged
    bot.config = bot.config.model_copy(update={setting: value})
    count = len(transport.requests)
    recover_unhedged_entry(bot, plan)
    assert len(transport.requests) == count and len(transport.orders) == 1


def test_any_durable_perp_entry_intent_blocks_unhedged_spot_only_recovery(unhedged):
    bot, transport, plan = unhedged
    bot.journal.record_order(plan.perp_entry)
    count = len(transport.requests)
    recover_unhedged_entry(bot, plan)
    assert len(transport.requests) == count and len(transport.orders) == 1


def test_journal_write_failure_prevents_order_submission(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    monkeypatch.setattr(bot.journal, "record_order", Mock(side_effect=ExecutionJournalError("offline disk error")))
    with pytest.raises(ExecutionJournalError):
        recover_unhedged_entry(bot, plan)
    assert len(transport.orders) == 1


def test_checkpoint_failure_after_execution_does_not_repeat_sell(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    update = bot.journal.update
    monkeypatch.setattr(bot.journal, "update", Mock(side_effect=ExecutionJournalError("offline checkpoint error")))
    with pytest.raises(ExecutionJournalError):
        recover_unhedged_entry(bot, plan)
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).spot_exit is None
    monkeypatch.setattr(bot.journal, "update", update)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 2


def test_stale_market_never_submits_flatten(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    assembly = bot.collector.collect_exit_depth(plan.symbol, markets=(Market.SPOT,))
    now = bot.clock()
    monkeypatch.setattr(bot.collector, "collect_exit_depth", lambda *a, **kw: assembly)
    bot.clock = lambda: now + timedelta(minutes=1)
    with pytest.raises(ExecutionGuardError):
        recover_unhedged_entry(bot, plan)
    assert len(transport.orders) == 1


def test_insufficient_exit_depth_never_extrapolates(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    monkeypatch.setattr(bot, "_exit_limit", Mock(side_effect=ExecutionGuardError("INSUFFICIENT_TESTNET_DEPTH")))
    with pytest.raises(ExecutionGuardError):
        recover_unhedged_entry(bot, plan)
    assert len(transport.orders) == 1


@pytest.mark.parametrize("status", [OrderState.EXPIRED, OrderState.CANCELED, OrderState.REJECTED])
def test_terminal_zero_entry_is_confirmed_failed_without_sell(unhedged, status):
    bot, transport, plan = unhedged
    payload = transport.orders[plan.spot_entry.client_order_id]
    payload.update(status=status.value, executedQty="0", cummulativeQuoteQty="0", avgPrice='0', fills=[])
    transport.base[plan.symbol] = D("1")
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FAILED
    assert len(transport.orders) == 1 and transport.base[plan.symbol] == 1


def test_terminal_partial_entry_is_flattened_at_actual_quantity(unhedged):
    bot, transport, plan = unhedged
    payload = transport.orders[plan.spot_entry.client_order_id]
    payload.update(status="EXPIRED", executedQty=".004", cummulativeQuoteQty="8")
    payload["fills"][0]["qty"] = ".004"
    transport.base[plan.symbol] = D("1.004")
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert bot.journal.intents(plan.attempt_id)[1].quantity == D(".004")
    assert len(transport.orders) == 2 and transport.base[plan.symbol] == 1


def test_known_base_entry_commission_is_subtracted_once(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    transport.orders[plan.spot_entry.client_order_id]["fills"][0].update(
        commission=".001", commissionAsset="ETH")
    transport.base[plan.symbol] = D("1.009")
    get_order = bot.spot.get_order
    def query(symbol, client_order_id):
        receipt = get_order(symbol, client_order_id)
        return replace(receipt, base_commission=D(".001")) if client_order_id == plan.spot_entry.client_order_id else receipt
    monkeypatch.setattr(bot.spot, "get_order", query)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert bot.journal.intents(plan.attempt_id)[1].quantity == D(".009")
    assert transport.base[plan.symbol] == 1


def test_untradable_dust_is_not_rounded_up_using_old_inventory(unhedged):
    bot, transport, plan = unhedged
    payload = transport.orders[plan.spot_entry.client_order_id]
    payload.update(status="EXPIRED", executedQty=".0001", cummulativeQuoteQty=".2")
    payload["fills"][0]["qty"] = ".0001"
    transport.base[plan.symbol] = D("1.0001")
    finish(bot, plan)
    assert len(transport.orders) == 1 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("change", ["order_id", "commission", "quantity"])
def test_runner_checks_existing_receipt_before_generic_reconcile_can_overwrite_it(unhedged, monkeypatch, change):
    bot, transport, plan = unhedged
    entry = bot.spot.get_order(plan.symbol, plan.spot_entry.client_order_id)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, spot_entry=entry,
                       failure_code="SPOT_SUBMISSION_UNKNOWN")
    get_order = bot.spot.get_order
    def changed(symbol, cid):
        receipt = get_order(symbol, cid)
        if cid != plan.spot_entry.client_order_id:
            return receipt
        changes = {"order_id": {"order_id": "999"}, "commission": {"base_commission": D(".001")},
                   "quantity": {"status": OrderState.EXPIRED, "executed_quantity": D(".004")}}
        return replace(receipt, **changes[change])
    monkeypatch.setattr(bot.spot, "get_order", changed)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).spot_entry == entry
    assert len(transport.orders) == 1


@pytest.mark.parametrize("status,quantity", [(OrderState.FILLED, ".004"), (OrderState.REJECTED, ".004")])
def test_contradictory_terminal_entry_never_authorizes_flatten(unhedged, monkeypatch, status, quantity):
    bot, transport, plan = unhedged
    get_order = bot.spot.get_order
    monkeypatch.setattr(bot.spot, "get_order", lambda symbol, cid: replace(get_order(symbol, cid),
        status=status, executed_quantity=D(quantity)))
    finish(bot, plan)
    assert len(transport.orders) == 1 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


def test_missing_base_commission_proof_keeps_recovery_blocked(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    get_order = bot.spot.get_order
    monkeypatch.setattr(bot.spot, "get_order", lambda symbol, cid: replace(get_order(symbol, cid), base_commission=None))
    finish(bot, plan)
    assert len(transport.orders) == 1


def test_current_zero_inventory_is_not_substituted_for_unknown_exit_receipt(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    recover_unhedged_entry(bot, plan)
    get_order = bot.spot.get_order
    def unknown_exit(symbol, cid):
        if cid == plan.spot_exit.client_order_id:
            raise TimeoutError("offline original flatten query failure")
        return get_order(symbol, cid)
    monkeypatch.setattr(bot.spot, "get_order", unknown_exit)
    finish(bot, plan)
    assert transport.base[plan.symbol] == 1 and len(transport.orders) == 2
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


def test_crash_after_durable_intent_before_post_never_blindly_retries(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    class Crash(BaseException):
        pass
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", Mock(side_effect=Crash()))
    with pytest.raises(Crash):
        recover_unhedged_entry(bot, plan)
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert len(bot.journal.intents(plan.attempt_id)) == 2 and len(transport.orders) == 1
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("status", [OrderState.NEW, OrderState.PARTIALLY_FILLED])
def test_active_original_flatten_never_creates_a_repair(unhedged, monkeypatch, status):
    bot, transport, plan = unhedged
    submit = bot.spot.place_ioc_limit_order
    def active(order):
        receipt = submit(order)
        transport.orders[order.client_order_id]["status"] = status.value
        return replace(receipt, status=status)
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", active)
    recover_unhedged_entry(bot, plan)
    finish(bot, plan)
    assert len(transport.orders) == 2 and not repairs(bot)
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    transport.orders[plan.spot_exit.client_order_id]["status"] = "FILLED"
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 2


@pytest.mark.parametrize("status", [OrderState.NEW, OrderState.PARTIALLY_FILLED])
def test_active_cached_entry_can_finish_then_abort_without_late_hedge(unhedged, status):
    bot, transport, plan = unhedged
    receipt = bot.spot.get_order(plan.symbol, plan.spot_entry.client_order_id)
    cached = replace(receipt, status=status, executed_quantity=D("0") if status is OrderState.NEW else D(".004"))
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, spot_entry=cached,
                       failure_code="SPOT_SUBMISSION_UNKNOWN")
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 2


def test_caller_decimal_precision_cannot_change_recovery_quantity(unhedged):
    bot, transport, plan = unhedged
    with localcontext() as context:
        context.prec = 2
        finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert bot.journal.intents(plan.attempt_id)[1].quantity == D(".01")
    assert transport.base[plan.symbol] == D("1")


@pytest.mark.parametrize("change", ["symbol", "side", "quantity", "ordinal", "original"])
def test_tampered_repair_journal_never_authorizes_another_sell(unhedged, monkeypatch, change):
    bot, transport, plan = unhedged
    partial_submission(bot, transport, monkeypatch, D(".002"))
    recover_unhedged_entry(bot, plan)
    recover_unhedged_entry(bot, plan)
    row = repairs(bot)[0]
    data = json.loads(row["order_json"])
    with bot.journal._connect() as db:
        if change in ("symbol", "side", "quantity"):
            data[change] = {"symbol": "BTCUSDT", "side": "BUY", "quantity": ".01"}[change]
            db.execute("UPDATE testnet_spot_exit_recoveries SET order_json=?", (json.dumps(data),))
        elif change == "ordinal":
            db.execute("UPDATE testnet_spot_exit_recoveries SET ordinal=3")
        else:
            original = bot.journal.get(plan.attempt_id).spot_exit
            db.execute("UPDATE testnet_spot_exit_recoveries SET original_exit_json=?",
                       (_receipt_json(replace(original, executed_quantity=D(".001"))),))
    finish(bot, plan)
    assert len(transport.orders) == 3 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
