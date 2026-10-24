"""Offline partial entry abort, native inventory proof and durable exit recovery."""

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
from crypto_arbitrage_bot.execution.aborted_entry_recovery import recover_aborted_entry
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.balance_ledger import balance_entry_blocker
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.execution.types import ExecutionMarket, OrderState
from crypto_arbitrage_bot.models.market import Market

D = Decimal


@pytest.fixture
def aborted(tmp_path, monkeypatch, request):
    symbol = getattr(request, "param", "ETHUSDT")
    quantity, done = (D(".002"), D(".001")) if symbol == "BTCUSDT" else (D(".01"), D(".004"))
    settings = load_settings(Path("docs/testnet-bot.toml"))
    settings = settings.model_copy(update={"testnet": settings.testnet.model_copy(update={
        "balance_evidence_enabled": True, "balance_evidence_directory": tmp_path / "frames"})})
    bot, transport = demo_bot(settings, tmp_path)
    transport.base[symbol] = D("1")
    frames = (BalanceSnapshot(ExecutionMarket.SPOT, AT, AT,
        (AssetBalance("USDT", D("10000")), AssetBalance(symbol[:-4], D("1")))),
        BalanceSnapshot(ExecutionMarket.USDTM, AT, AT, (AssetBalance("USDT", D("10000")),)))
    monkeypatch.setattr(balance_frame_store, "capture_balance_snapshot",
                        lambda client, **kw: frames[0 if client is bot.spot else 1])
    plan = bot._plan(symbol, quantity, "abort-" + symbol)
    submit = bot.perp.place_ioc_limit_order
    def partial_entry_lost_ack(order):
        submit(order)
        transport.positions[symbol] += quantity - done
        payload = transport.orders[order.client_order_id]
        payload.update(status="EXPIRED", executedQty=str(done), cummulativeQuoteQty=str(done * order.limit_price))
        raise TimeoutError("offline partial entry acknowledgement lost")
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", partial_entry_lost_ack)
    assert bot.coordinator.open_pair(plan).state is State.RECOVERY_REQUIRED
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", submit)
    assert len(transport.orders) == 2 and transport.positions[symbol] == -done
    return bot, transport, plan, done


def finish(bot, plan, rounds=7):
    for _ in range(rounds):
        bot._managed_exits()
        if bot.journal.get(plan.attempt_id).state is State.FLATTENED:
            break


def rows(bot, table):
    assert table in ("testnet_spot_exit_recoveries", "testnet_exit_recoveries")
    with bot.journal._connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            return []
        return db.execute(f"SELECT * FROM {table} ORDER BY ordinal").fetchall()


def partial_exit(bot, transport, monkeypatch, market, quantity):
    client = bot.spot if market is ExecutionMarket.SPOT else bot.perp
    submit = client.place_ioc_limit_order
    def partial(order):
        receipt = submit(order)
        if market is ExecutionMarket.SPOT:
            transport.base[order.symbol] += order.quantity - quantity
        else:
            transport.positions[order.symbol] -= order.quantity - quantity
        payload = transport.orders[order.client_order_id]
        payload.update(status="EXPIRED", executedQty=str(quantity), cummulativeQuoteQty=str(quantity * order.limit_price))
        if market is ExecutionMarket.SPOT:
            payload["fills"][0]["qty"] = str(quantity)
        return replace(receipt, status=OrderState.EXPIRED, executed_quantity=quantity,
                       cumulative_quote_quantity=quantity * order.limit_price)
    monkeypatch.setattr(client, "place_ioc_limit_order", partial)
    return submit


@pytest.mark.parametrize("aborted", ["BTCUSDT", "ETHUSDT"], indirect=True)
@pytest.mark.parametrize("restart", [False, True])
def test_confirmed_partial_entry_flattens_both_actual_legs(aborted, restart):
    bot, transport, plan, done = aborted
    if restart:
        bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    pair = bot.journal.get(plan.attempt_id)
    assert pair.state is State.FLATTENED and pair.failure_code == "ABORTED_ENTRY_RESTORED"
    assert pair.perp_entry.status is OrderState.EXPIRED and pair.perp_entry.executed_quantity == done
    assert pair.perp_exit.original_quantity == pair.perp_exit.executed_quantity == done
    assert pair.spot_exit.executed_quantity == plan.spot_entry.quantity
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0
    assert len(transport.orders) == 4
    intents = bot.journal.intents(plan.attempt_id)
    assert intents[-1].reduce_only and intents[-1].quantity == done and intents[-1].side.value == "BUY"
    assert len([i for i in intents if i.market is ExecutionMarket.USDTM and i.side.value == "SELL"]) == 1
    count = len(transport.requests)
    finish(bot, plan)
    assert len(transport.requests) == count
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"
    assert reconcile_closed_attempt(bot, plan.attempt_id).calculation is None


@pytest.mark.parametrize("market", [ExecutionMarket.SPOT, ExecutionMarket.USDTM])
def test_lost_original_exit_ack_is_queried_without_new_post(aborted, monkeypatch, market):
    bot, transport, plan, _ = aborted
    if market is ExecutionMarket.USDTM:
        recover_aborted_entry(bot, plan)
    client = bot.spot if market is ExecutionMarket.SPOT else bot.perp
    submit, seen = client.place_ioc_limit_order, []
    def lost(order):
        seen.append(order.client_order_id)
        submit(order)
        raise TimeoutError("offline original exit ACK lost")
    monkeypatch.setattr(client, "place_ioc_limit_order", lost)
    recover_aborted_entry(bot, plan)
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 4
    assert len(seen) == 1


@pytest.mark.parametrize("market,amount", [(ExecutionMarket.SPOT, ".004"), (ExecutionMarket.USDTM, ".001")])
def test_partial_original_exit_repairs_only_confirmed_residual(aborted, monkeypatch, market, amount):
    bot, transport, plan, done = aborted
    if market is ExecutionMarket.USDTM:
        recover_aborted_entry(bot, plan)
    submit = partial_exit(bot, transport, monkeypatch, market, D(amount))
    recover_aborted_entry(bot, plan)
    monkeypatch.setattr(bot.spot if market is ExecutionMarket.SPOT else bot.perp, "place_ioc_limit_order", submit)
    finish(bot, plan)
    table = "testnet_spot_exit_recoveries" if market is ExecutionMarket.SPOT else "testnet_exit_recoveries"
    repair = json.loads(rows(bot, table)[0]["order_json"])
    assert D(repair["quantity"]) == (plan.spot_entry.quantity if market is ExecutionMarket.SPOT else done) - D(amount)
    assert repair["reduce_only"] is (market is ExecutionMarket.USDTM)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 5
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0


def test_both_original_exits_can_be_partial_with_all_sources_preserved(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    ss = partial_exit(bot, transport, monkeypatch, ExecutionMarket.SPOT, D(".004"))
    recover_aborted_entry(bot, plan)
    monkeypatch.setattr(bot.spot, "place_ioc_limit_order", ss)
    recover_aborted_entry(bot, plan)
    ps = partial_exit(bot, transport, monkeypatch, ExecutionMarket.USDTM, D(".001"))
    recover_aborted_entry(bot, plan)
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", ps)
    finish(bot, plan)
    pair = bot.journal.get(plan.attempt_id)
    assert pair.state is State.FLATTENED and pair.spot_exit.status is pair.perp_exit.status is OrderState.EXPIRED
    assert len(transport.orders) == 6
    assert len(rows(bot, "testnet_spot_exit_recoveries")) == len(rows(bot, "testnet_exit_recoveries")) == 1


@pytest.mark.parametrize("market,amount", [(ExecutionMarket.SPOT, ".004"), (ExecutionMarket.USDTM, ".001")])
def test_lost_repair_ack_does_not_duplicate_exit_after_restart(aborted, monkeypatch, market, amount):
    bot, transport, plan, _ = aborted
    client = bot.spot if market is ExecutionMarket.SPOT else bot.perp
    if market is ExecutionMarket.USDTM:
        recover_aborted_entry(bot, plan)
    submit = partial_exit(bot, transport, monkeypatch, market, D(amount))
    recover_aborted_entry(bot, plan)
    seen = []
    def lost(order):
        seen.append(order.client_order_id)
        submit(order)
        raise TimeoutError("offline repair ACK lost")
    monkeypatch.setattr(client, "place_ioc_limit_order", lost)
    recover_aborted_entry(bot, plan)
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert len(seen) == 1 and len(transport.orders) == 5


def test_perp_retry_limit_survives_restart_without_busy_post_loop(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    recover_aborted_entry(bot, plan)
    partial_exit(bot, transport, monkeypatch, ExecutionMarket.USDTM, D(".001"))
    finish(bot, plan, 12)
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan, 12)
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    assert len(rows(bot, "testnet_exit_recoveries")) == 2 and len(transport.orders) == 6
    assert transport.positions[plan.symbol] == D("-.001")


@pytest.mark.parametrize("status", [OrderState.NEW, OrderState.PARTIALLY_FILLED])
def test_active_perp_entry_blocks_any_flatten_until_terminal(aborted, status):
    bot, transport, plan, _ = aborted
    transport.orders[plan.perp_entry.client_order_id]["status"] = status.value
    finish(bot, plan)
    assert len(transport.orders) == 2
    transport.orders[plan.perp_entry.client_order_id]["status"] = "EXPIRED"
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 4


def test_unknown_perp_entry_query_blocks_spot_sale(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    monkeypatch.setattr(bot.perp, "get_order", Mock(side_effect=TimeoutError("offline entry query failed")))
    finish(bot, plan)
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("status", [OrderState.EXPIRED, OrderState.REJECTED, OrderState.CANCELED])
def test_confirmed_zero_perp_entry_needs_only_spot_flatten(aborted, status):
    bot, transport, plan, _ = aborted
    payload = transport.orders[plan.perp_entry.client_order_id]
    payload.update(status=status.value, executedQty="0", cummulativeQuoteQty="0", avgPrice='0')
    transport.positions[plan.symbol] = D("0")
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED and len(transport.orders) == 3
    assert bot.journal.get(plan.attempt_id).perp_exit is None


@pytest.mark.parametrize("field,value", [("position_quantity", D("-.005")), ("position_quantity", D(".004")),
    ("open_order_count", 1), ("can_trade", False), ("one_way", False), ("single_asset", False),
    ("leverage", D("2")), ("symbol", "BTCUSDT"), ("market", ExecutionMarket.SPOT)])
def test_account_conflict_blocks_all_exit_posts(aborted, monkeypatch, field, value):
    bot, transport, plan, _ = aborted
    account = bot.perp.get_account_state(plan.symbol)
    monkeypatch.setattr(bot.perp, "get_account_state", lambda *a: replace(account, **{field: value}))
    finish(bot, plan)
    assert len(transport.orders) == 2


@pytest.mark.parametrize("delta", [".001", "-.001"])
def test_inventory_conflict_does_not_sell_old_spot(aborted, delta):
    bot, transport, plan, _ = aborted
    transport.base[plan.symbol] += D(delta)
    finish(bot, plan)
    assert len(transport.orders) == 2


@pytest.mark.parametrize("change", ["missing", "invalid", "tampered"])
def test_native_before_evidence_is_required(aborted, change):
    bot, transport, plan, _ = aborted
    p = bot.balance_directory / (plan.attempt_id + ".before.json")
    if change == "missing":
        p.unlink()
    elif change == "invalid":
        p.write_text("{", encoding="utf-8")
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        data["snapshots"][0]["balances"][1]["amount"] = "0"
        p.write_text(json.dumps(data), encoding="utf-8")
    count = len(transport.requests)
    recover_aborted_entry(bot, plan)
    assert len(transport.requests) == count and len(transport.orders) == 2


@pytest.mark.parametrize("change", ["order_id", "quantity", "status"])
def test_cached_terminal_entry_cannot_be_silently_replaced(aborted, monkeypatch, change):
    bot, transport, plan, _ = aborted
    entry = bot.perp.get_order(plan.symbol, plan.perp_entry.client_order_id)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, perp_entry=entry,
                       failure_code="PERP_SUBMISSION_UNKNOWN")
    query = bot.perp.get_order
    def changed(symbol, cid):
        receipt = query(symbol, cid)
        changes = {"order_id": {"order_id": "999"}, "quantity": {"executed_quantity": D(".005")},
                   "status": {"status": OrderState.CANCELED}}
        return replace(receipt, **changes[change])
    monkeypatch.setattr(bot.perp, "get_order", changed)
    finish(bot, plan)
    assert len(transport.orders) == 2 and bot.journal.get(plan.attempt_id).perp_entry == entry


@pytest.mark.parametrize("status", [OrderState.FILLED, OrderState.REJECTED])
def test_contradictory_terminal_perp_entry_blocks_flatten(aborted, status):
    bot, transport, plan, _ = aborted
    transport.orders[plan.perp_entry.client_order_id]["status"] = status.value
    finish(bot, plan)
    assert len(transport.orders) == 2


@pytest.mark.parametrize("code", [None, "MANUAL_REVIEW_REQUIRED"])
def test_manual_state_is_not_permission_to_abort(aborted, code):
    bot, transport, plan, _ = aborted
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, failure_code=code)
    count = len(transport.requests)
    finish(bot, plan)
    assert len(transport.requests) == count and len(transport.orders) == 2


@pytest.mark.parametrize("setting,value", [("balance_evidence_enabled", False), ("max_exit_recovery_attempts", 0)])
def test_disabled_native_abort_keeps_legacy_route(aborted, setting, value):
    bot, transport, plan, _ = aborted
    bot.config = bot.config.model_copy(update={setting: value})
    count = len(transport.requests)
    assert recover_aborted_entry(bot, plan) is None
    assert len(transport.requests) == count and len(transport.orders) == 2


def test_durable_intent_failure_blocks_exit_post(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    monkeypatch.setattr(bot.journal, "record_order", Mock(side_effect=ExecutionJournalError("offline disk error")))
    with pytest.raises(ExecutionJournalError):
        recover_aborted_entry(bot, plan)
    assert len(transport.orders) == 2


def test_recovery_uses_existing_quote_quality_and_depth_guards(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    monkeypatch.setattr(bot, "_exit_limit", Mock(side_effect=ExecutionGuardError("INSUFFICIENT_TESTNET_DEPTH")))
    with pytest.raises(ExecutionGuardError):
        recover_aborted_entry(bot, plan)
    assert len(transport.orders) == 2


def test_stale_quote_cannot_authorize_an_exit(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    assembly, now = bot.collector.collect_exit_depth(plan.symbol, markets=(Market.SPOT,)), bot.clock()
    monkeypatch.setattr(bot.collector, "collect_exit_depth", lambda *a, **kw: assembly)
    bot.clock = lambda: now + timedelta(minutes=1)
    with pytest.raises(ExecutionGuardError):
        recover_aborted_entry(bot, plan)
    assert len(transport.orders) == 2


def test_full_hedge_with_lost_ack_keeps_original_holding_path(aborted):
    bot, transport, plan, _ = aborted
    payload = transport.orders[plan.perp_entry.client_order_id]
    payload.update(status="FILLED", executedQty=str(plan.perp_entry.quantity),
                   cummulativeQuoteQty=str(plan.perp_entry.quantity * plan.perp_entry.limit_price))
    transport.positions[plan.symbol] = -plan.perp_entry.quantity
    assert recover_aborted_entry(bot, plan) is None
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.HEDGED and len(transport.orders) == 2


def test_low_decimal_context_cannot_change_partial_entry_quantities(aborted):
    bot, transport, plan, _ = aborted
    with localcontext() as context:
        context.prec = 2
        finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0


@pytest.mark.parametrize("field,value", [("symbol", "BTCUSDT"), ("side", "SELL"), ("quantity", ".01"),
                                        ("reduce_only", False), ("ordinal", 3)])
def test_invalid_perp_repair_evidence_never_creates_another_post(aborted, monkeypatch, field, value):
    bot, transport, plan, _ = aborted
    recover_aborted_entry(bot, plan)
    partial_exit(bot, transport, monkeypatch, ExecutionMarket.USDTM, D(".001"))
    recover_aborted_entry(bot, plan)
    recover_aborted_entry(bot, plan)
    row = rows(bot, "testnet_exit_recoveries")[0]
    with bot.journal._connect() as db:
        if field == "ordinal":
            db.execute("UPDATE testnet_exit_recoveries SET ordinal=?", (value,))
        else:
            data = json.loads(row["order_json"])
            data[field] = value
            db.execute("UPDATE testnet_exit_recoveries SET order_json=?", (json.dumps(data),))
    finish(bot, plan)
    assert len(transport.orders) == 5 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("market", [ExecutionMarket.SPOT, ExecutionMarket.USDTM])
def test_unknown_exit_id_cannot_be_replaced_even_when_account_looks_flat(aborted, monkeypatch, market):
    bot, transport, plan, _ = aborted
    recover_aborted_entry(bot, plan)
    if market is ExecutionMarket.USDTM:
        recover_aborted_entry(bot, plan)
    client = bot.spot if market is ExecutionMarket.SPOT else bot.perp
    query = client.get_order
    exit_id = plan.spot_exit.client_order_id if market is ExecutionMarket.SPOT else plan.perp_exit.client_order_id
    def unknown(symbol, cid):
        if cid == exit_id:
            raise TimeoutError("offline exit identity unavailable")
        return query(symbol, cid)
    monkeypatch.setattr(client, "get_order", unknown)
    count = len(transport.orders)
    finish(bot, plan)
    assert len(transport.orders) == count
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize("market,amount", [(ExecutionMarket.SPOT, ".004"), (ExecutionMarket.USDTM, ".001")])
def test_crash_after_repair_post_before_checkpoint_recovers_same_id(aborted, monkeypatch, market, amount):
    bot, transport, plan, _ = aborted
    client = bot.spot if market is ExecutionMarket.SPOT else bot.perp
    if market is ExecutionMarket.USDTM:
        recover_aborted_entry(bot, plan)
    submit = partial_exit(bot, transport, monkeypatch, market, D(amount))
    recover_aborted_entry(bot, plan)
    def crash(order):
        submit(order)
        raise KeyboardInterrupt("offline process interruption after exchange acceptance")
    monkeypatch.setattr(client, "place_ioc_limit_order", crash)
    with pytest.raises(KeyboardInterrupt):
        recover_aborted_entry(bot, plan)
    table = "testnet_spot_exit_recoveries" if market is ExecutionMarket.SPOT else "testnet_exit_recoveries"
    assert rows(bot, table)[0]["receipt_json"] is None
    monkeypatch.setattr(client, "place_ioc_limit_order", submit)
    bot = Bot(bot.settings, bot.spot, bot.perp)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert len(transport.orders) == 5 and len(rows(bot, table)) == 1


@pytest.mark.parametrize("kind", ["receipt", "original"])
def test_tampered_perp_repair_receipt_blocks_further_submissions(aborted, monkeypatch, kind):
    bot, transport, plan, _ = aborted
    recover_aborted_entry(bot, plan)
    partial_exit(bot, transport, monkeypatch, ExecutionMarket.USDTM, D(".001"))
    recover_aborted_entry(bot, plan)
    recover_aborted_entry(bot, plan)
    row = rows(bot, "testnet_exit_recoveries")[0]
    column = "receipt_json" if kind == "receipt" else "original_exit_json"
    data = json.loads(row[column])
    data["order_id"] = "999999"
    with bot.journal._connect() as db:
        db.execute(f"UPDATE testnet_exit_recoveries SET {column}=?", (json.dumps(data),))
    finish(bot, plan)
    assert len(transport.orders) == 5 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


def test_confirmed_spot_base_commission_reduces_flatten_quantity_once(aborted, monkeypatch):
    bot, transport, plan, _ = aborted
    payload = transport.orders[plan.spot_entry.client_order_id]
    payload["fills"][0].update(commission=".001", commissionAsset=plan.symbol[:-4])
    transport.base[plan.symbol] -= D(".001")
    request = bot.spot._request
    def with_entry_fee(operation, params=(), **kwargs):
        result = request(operation, params, **kwargs)
        if operation == "trades" and dict(params)["orderId"] == str(payload["orderId"]):
            result[0].update(commission=".001", commissionAsset=plan.symbol[:-4])
        return result
    monkeypatch.setattr(bot.spot, "_request", with_entry_fee)
    entry = bot.spot.get_order(plan.symbol, plan.spot_entry.client_order_id)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, spot_entry=entry,
                       failure_code="PERP_SUBMISSION_UNKNOWN")
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert bot.journal.get(plan.attempt_id).spot_exit.executed_quantity == D(".009")
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0


def test_perp_dust_is_never_rounded_up_to_a_new_order(aborted):
    bot, transport, plan, _ = aborted
    payload = transport.orders[plan.perp_entry.client_order_id]
    payload.update(executedQty=".0001", cummulativeQuoteQty=str(D(".0001") * plan.perp_entry.limit_price))
    transport.positions[plan.symbol] = D("-.0001")
    finish(bot, plan)
    assert len(transport.orders) == 3 and transport.base[plan.symbol] == 1
    assert transport.positions[plan.symbol] == D("-.0001")
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


def test_old_entry_deadline_does_not_block_confirmed_reduce_only_flatten(aborted):
    bot, transport, plan, _ = aborted
    now = bot.clock()
    bot.clock = lambda: now + timedelta(hours=2)
    transport.clock = bot.collector.clock = bot.clock
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.FLATTENED
    assert all(i.submit_before is None for i in bot.journal.intents(plan.attempt_id)
               if i.client_order_id in (plan.spot_exit.client_order_id, plan.perp_exit.client_order_id))
    assert transport.positions[plan.symbol] == 0
