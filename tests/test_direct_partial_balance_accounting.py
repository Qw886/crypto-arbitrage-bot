"""Direct IOC partial fills retain genuine coordinator states and wallet gates.

Every exchange response in this module is an explicit offline fixture.
"""

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from test_abort_balance_accounting import native_sources
from test_trade_accounting import AT, source_rows

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import balance_frame_store
from crypto_arbitrage_bot.execution.abort_balance_proof import ABORT_SCOPE, supported_abort
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.balance_check_repository import (
    BalanceCheckRepository,
    BalanceCheckStatus,
)
from crypto_arbitrage_bot.execution.balance_ledger import (
    balance_entry_blocker,
    reconcile_balance_ledger,
)
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.types import ExecutionMarket, OrderState

D = Decimal


@pytest.fixture(params=[("BTCUSDT", "SPOT"), ("ETHUSDT", "SPOT"),
                        ("BTCUSDT", "USDTM"), ("ETHUSDT", "USDTM")],
                ids=["btc-spot", "eth-spot", "btc-perp", "eth-perp"])
def direct_partial(tmp_path, monkeypatch, request):
    symbol, market = request.param
    quantity, done = (D(".002"), D(".001")) if symbol == "BTCUSDT" else (D(".01"), D(".004"))
    settings = load_settings(Path("docs/testnet-bot.toml"))
    settings = settings.model_copy(update={"testnet": settings.testnet.model_copy(update={
        "balance_evidence_enabled": True, "balance_evidence_directory": tmp_path / "frames"})})
    bot, transport = demo_bot(settings, tmp_path)
    transport.base[symbol] = D(1)
    frames = (BalanceSnapshot(ExecutionMarket.SPOT, AT, AT,
        (AssetBalance("USDT", D(10000)), AssetBalance(symbol[:-4], D(1)))),
        BalanceSnapshot(ExecutionMarket.USDTM, AT, AT, (AssetBalance("USDT", D(10000)),)))
    monkeypatch.setattr(balance_frame_store, "capture_balance_snapshot",
                        lambda client, **kw: frames[0 if client is bot.spot else 1])
    plan = bot._plan(symbol, quantity, "direct-partial-" + symbol + market)
    opener = transport.open
    entry = plan.spot_entry if market == "SPOT" else plan.perp_entry

    def partial_response(request, **kwargs):
        response = opener(request, **kwargs)
        params = parse_qs((request.data or b"").decode())
        if request.get_method() == "POST" and params.get("newClientOrderId") == [entry.client_order_id]:
            payload = json.loads(response.body)
            payload.update(status="EXPIRED", executedQty=str(done),
                           cummulativeQuoteQty=str(done * entry.limit_price))
            if market == "SPOT":
                transport.base[symbol] -= quantity - done
                payload["fills"][0]["qty"] = str(done)
            else:
                transport.positions[symbol] += quantity - done
            transport.orders[entry.client_order_id] = payload
            return DemoResponse(payload)
        return response

    monkeypatch.setattr(transport, "open", partial_response)
    pair = bot.coordinator.open_pair(plan)
    assert pair.state is State.FLATTENED
    assert pair.failure_code == ("SPOT_ENTRY_PARTIAL" if market == "SPOT" else "PERP_ENTRY_PARTIAL")
    receipt = pair.spot_entry if market == "SPOT" else pair.perp_entry
    assert receipt.status is OrderState.EXPIRED and receipt.executed_quantity == done
    assert receipt.original_quantity == quantity and transport.base[symbol] == 1
    assert transport.positions[symbol] == 0
    return bot, transport, plan, market


@pytest.mark.parametrize("restart", [False, True])
def test_direct_partial_wallet_archive_preserves_terminal_sources_and_null_profit(direct_partial, monkeypatch, restart):
    bot, transport, plan, market = direct_partial
    native_sources(bot, plan, monkeypatch)
    before, order_count = source_rows(bot), len(transport.orders)
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check is not None and check.status is BalanceCheckStatus.ARCHIVED
    assert check.balance_status == "MATCHED" and balance_entry_blocker(bot) is None
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["accounting_scope"] == ABORT_SCOPE and report["status"] == "INCOMPLETE"
    assert report["calculation"] is report["net_return"] is None
    assert source_rows(bot) == before and len(transport.orders) == order_count
    if market == "SPOT":
        bot.perp.get_order.assert_not_called()
        bot.perp.get_order_trades.assert_not_called()
        bot.perp.get_funding_payments.assert_not_called()
    calls = bot.spot.get_order.call_count + bot.perp.get_order.call_count
    if restart:
        bot = Bot(bot.settings, bot.spot, bot.perp)
    reconcile_balance_ledger(bot)
    assert bot.spot.get_order.call_count + bot.perp.get_order.call_count == calls
    assert balance_entry_blocker(bot) is None and bot.journal.get(plan.attempt_id).state is State.FLATTENED


@pytest.mark.parametrize("state", [State.FAILED, State.RECOVERY_REQUIRED, State.HEDGED])
def test_partial_code_never_admits_unfinished_or_failed_parent(direct_partial, monkeypatch, state):
    bot, _, plan, _ = direct_partial
    bot.journal.update(plan.attempt_id, state=state)
    native_sources(bot, plan, monkeypatch)
    assert not supported_abort(bot.journal.get(plan.attempt_id))
    assert BalanceCheckRepository(bot.journal).claim(plan.attempt_id, AT + timedelta(seconds=500),
        max_attempts=3, lease_seconds=60) is None
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id) is None
    bot.spot.get_order.assert_not_called()
    bot.perp.get_order.assert_not_called()


def test_conflicting_live_receipt_keeps_partial_review_and_original_records(direct_partial, monkeypatch):
    from dataclasses import replace

    bot, _, plan, market = direct_partial
    receipts, _, _, _ = native_sources(bot, plan, monkeypatch)
    cid = plan.spot_entry.client_order_id if market == "SPOT" else plan.perp_entry.client_order_id
    receipts[cid] = replace(receipts[cid], executed_quantity=plan.spot_entry.quantity)
    before = source_rows(bot)
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check is not None and check.status is BalanceCheckStatus.FAILED
    assert check.failure_code == "ABORT_EVIDENCE_CONFLICT" and check.audit_id is None
    assert source_rows(bot) == before


def test_mock_case_has_no_live_exchange_transport(direct_partial):
    bot, transport, _, _ = direct_partial
    assert not bot.spot.uses_default_transport and not bot.perp.uses_default_transport
    assert all(urlsplit(url).hostname in ("testnet.binance.vision", "demo-fapi.binance.com")
               for _, url in transport.requests)


@pytest.mark.parametrize("invalid", ["zero", "full", "live", "filled-label"])
def test_partial_label_without_terminal_partial_receipt_keeps_review(direct_partial, monkeypatch, invalid):
    from dataclasses import replace

    bot, _, plan, market = direct_partial
    pair = bot.journal.get(plan.attempt_id)
    field = "spot_entry" if market == "SPOT" else "perp_entry"
    entry = getattr(pair, field)
    updates = {
        "zero": {"executed_quantity": D(0)},
        "full": {"executed_quantity": entry.original_quantity},
        "live": {"status": OrderState.PARTIALLY_FILLED},
        "filled-label": {"status": OrderState.FILLED},
    }
    bot.journal.update(plan.attempt_id, state=pair.state, **{field: replace(entry, **updates[invalid])})
    native_sources(bot, plan, monkeypatch)
    before = source_rows(bot)
    assert not supported_abort(bot.journal.get(plan.attempt_id))
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id) is None
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"
    bot.spot.get_order.assert_not_called()
    bot.perp.get_order.assert_not_called()
    assert source_rows(bot) == before
