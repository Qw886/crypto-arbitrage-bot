"""Closed Demo fills use existing cash-flow math, never strategy estimates."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    PairExecutionState,
)
from crypto_arbitrage_bot.execution.reconciliation_repository import (
    AccountingPersistenceCode,
    ReconciliationRepository,
)
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as Error
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationCode as Code,
)
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationStatus as Status,
)
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    reconcile_closed_attempt,
)
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    OrderReceipt,
    OrderState,
)

D = Decimal
AT = datetime(2026, 10, 3, tzinfo=UTC)


@pytest.fixture
def evidence_bot(tmp_path, monkeypatch, request):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-funding.toml")), tmp_path)
    quantity = D(getattr(request, "param", ".01"))
    plan = bot._plan("ETHUSDT", quantity, "accounting-attempt")
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    assert bot.coordinator.close_pair(plan).state is PairExecutionState.CLOSED
    orders = (plan.spot_entry, plan.spot_exit, plan.perp_entry, plan.perp_exit)
    prices = (D("2000"), D("2001"), D("2010"), D("2002"))
    receipts, trades = {}, {}
    for index, (order, price) in enumerate(zip(orders, prices, strict=True)):
        oid = str(index + 1)
        fee_rate = D(".001") if order.market is ExecutionMarket.SPOT else D(".0005")
        receipts[order.client_order_id] = OrderReceipt(order.market, plan.symbol, oid, order.client_order_id,
            OrderState.FILLED, quantity, quantity, quantity * price, price,
            side=order.side, order_type='LIMIT', time_in_force='IOC', limit_price=order.limit_price,
            reduce_only=order.reduce_only if order.market is ExecutionMarket.USDTM else None,
            position_side='BOTH' if order.market is ExecutionMarket.USDTM else None)
        official_pnl = None if order.market is ExecutionMarket.SPOT else (D("0") if index == 2 else D("8") * quantity)
        trades[oid] = (AccountTrade(order.market, plan.symbol, oid, oid, order.side, price, quantity,
            price * quantity, price * quantity * fee_rate, "USDT",
            AT + timedelta(seconds=1 if index in (0, 2) else 300), official_pnl, '{"fixture":"official fill"}'),)
    bot.clock = lambda: AT + timedelta(seconds=400)
    for client in (bot.spot, bot.perp):
        monkeypatch.setattr(client, "get_order", lambda symbol, cid, **kwargs: receipts[cid])
        monkeypatch.setattr(client, "get_order_trades", lambda symbol, oid: trades[oid])
        monkeypatch.setattr(client, "place_ioc_limit_order", Mock(side_effect=AssertionError("accounting must never order")))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=(
        FundingPayment("account-payment", plan.symbol, AT + timedelta(seconds=200), D(".01")),)))
    return bot, plan, receipts, trades


def codes(report):
    return {issue.code for issue in report.issues}


def source_rows(bot):
    with bot.journal._connect() as db:
        return (tuple(tuple(row) for row in db.execute("SELECT * FROM testnet_pair_attempts ORDER BY attempt_id")),
                tuple(tuple(row) for row in db.execute("SELECT * FROM testnet_order_intents ORDER BY client_order_id")))


def test_hand_calculated_actual_fees_and_profit(evidence_bot):
    bot, plan, _, _ = evidence_bot
    before = source_rows(bot)
    report = reconcile_closed_attempt(bot, plan.attempt_id, capital=D("50"))
    assert report.status is Status.RECONCILED
    assert report.calculation.price_pnl == D(".09")
    assert report.fees.spot_entry == D(".02") and report.fees.spot_exit == D(".02001")
    assert report.fees.perp_entry == D(".01005") and report.fees.perp_exit == D(".01001")
    assert report.calculation.net_profit == D(".03993")
    assert report.net_return == D(".0007986")
    assert source_rows(bot) == before
    assert bot.spot.place_ioc_limit_order.call_count == bot.perp.place_ioc_limit_order.call_count == 0


@pytest.mark.parametrize("amount", [".01", "0", "-.01"])
def test_positive_zero_negative_actual_funding_counted_once(evidence_bot, monkeypatch, amount):
    bot, plan, _, _ = evidence_bot
    payment = FundingPayment("posted", plan.symbol, AT + timedelta(seconds=200), D(amount))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=(payment,)))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.RECONCILED
    assert report.calculation.net_profit == D(".02993") + D(amount)
    assert report.net_return is None  # No fictional occupied capital.


def test_missing_funding_is_null_not_zero(evidence_bot, monkeypatch):
    bot, plan, _, _ = evidence_bot
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=()))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.INCOMPLETE and report.calculation is None
    assert report.funding_total is None and Code.FUNDING_NOT_CONFIRMED in codes(report)
    assert report.price_pnl == D(".09") and report.fees.total == D(".06007")


@pytest.mark.parametrize("asset", ["ETH", "BNB"])
def test_non_quote_fees_not_guessed_or_double_charged(evidence_bot, asset):
    bot, plan, _, trades = evidence_bot
    trades["1"] = (replace(trades["1"][0], commission_asset=asset, commission=D(".00001")),)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.INCOMPLETE and report.calculation is None
    assert Code.UNVALUED_COMMISSION in codes(report)
    if asset == "ETH":
        assert report.spot_inventory_change == D("-.00001")


def test_fee_rebate_requires_explicit_accounting(evidence_bot):
    bot, plan, _, trades = evidence_bot
    trades["1"] = (replace(trades["1"][0], commission=D("-.01")),)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.REBATE_REQUIRES_ACCOUNTING in codes(report) and report.calculation is None


def test_missing_fills_not_replaced_by_limit_or_receipt_average(evidence_bot):
    bot, plan, receipts, trades = evidence_bot
    receipts[plan.perp_exit.client_order_id] = replace(receipts[plan.perp_exit.client_order_id], average_price=D("1"))
    trades["4"] = ()
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.INCOMPLETE_FILLS in codes(report) and report.calculation is None


@pytest.mark.parametrize("evidence_bot", [".03"], indirect=True)
def test_recurring_vwap_preserves_exact_cashflow_under_low_decimal_context(evidence_bot):
    bot, plan, receipts, trades = evidence_bot
    for oid, items in list(trades.items()):
        original = items[0]
        # .03 at three prices: 2000,2000,2001 -> recurring VWAP 2000+1/3.
        values = []
        for index, offset in enumerate((D("0"), D("0"), D("1"))):
            quantity, price = D(".01"), original.price + offset
            pnl = None if original.market is ExecutionMarket.SPOT else (D("0") if oid == "3" else D(".08"))
            values.append(replace(original, trade_id=str(int(oid) * 10 + index), price=price,
                quantity=quantity, quote_quantity=price * quantity, commission=D("0"), official_perp_pnl=pnl))
        trades[oid] = tuple(values)
    with localcontext() as context:
        context.prec = 6
        report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.RECONCILED
    assert report.calculation.net_profit == D(".28")
    assert str(report.legs[0].vwap).startswith("2000.333333333333333333333333333333")


@pytest.mark.parametrize("failure_leg", ["spot_entry", "perp_exit"])
def test_account_get_failure_retains_safe_incomplete_result(evidence_bot, monkeypatch, failure_leg):
    bot, plan, receipts, _ = evidence_bot
    blocked = getattr(plan, failure_leg).client_order_id
    def get_order(symbol, cid, **kwargs):
        if cid == blocked:
            raise Error(ExecutionErrorKind.NETWORK_ERROR, "TimeoutError")
        return receipts[cid]
    for client in (bot.spot, bot.perp):
        monkeypatch.setattr(client, "get_order", get_order)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.API_UNAVAILABLE in codes(report) and report.calculation is None


def test_official_perp_pnl_disagreement_invalidates_profit(evidence_bot):
    bot, plan, _, trades = evidence_bot
    trades["4"] = (replace(trades["4"][0], official_perp_pnl=D("999")),)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.INVALID and report.calculation is None
    assert Code.OFFICIAL_PERP_PNL_MISMATCH in codes(report)


@pytest.mark.parametrize("change,expected", [
    ({"symbol": "BTCUSDT"}, Code.IDENTITY_MISMATCH),
    ({"quote_quantity": D("999")}, Code.TRADE_NOTIONAL_MISMATCH),
    ({"traded_at": AT + timedelta(days=1)}, Code.TRADE_TIME_MISMATCH),
])
def test_conflicting_fill_identity_value_or_time_is_invalid(evidence_bot, change, expected):
    bot, plan, _, trades = evidence_bot
    trades["1"] = (replace(trades["1"][0], **change),)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert expected in codes(report) and report.status is Status.INVALID and report.calculation is None


def test_duplicate_trade_not_counted_twice(evidence_bot):
    bot, plan, _, trades = evidence_bot
    trades["2"] = (replace(trades["2"][0], trade_id=trades["1"][0].trade_id),)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.DUPLICATE_TRADE in codes(report) and report.calculation is None


@pytest.mark.parametrize("offset", [1, 301])
def test_outside_funding_window_invalidates_accounting(evidence_bot, monkeypatch, offset):
    bot, plan, _, _ = evidence_bot
    payment = FundingPayment("outside", plan.symbol, AT + timedelta(seconds=offset), D("1"))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=(payment,)))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.FUNDING_ATTRIBUTION_CONFLICT in codes(report) and report.calculation is None


def test_open_pair_never_triggers_accounting_network_requests(evidence_bot, monkeypatch):
    bot, plan, _, _ = evidence_bot
    bot.journal.update(plan.attempt_id, state=PairExecutionState.HEDGED)
    monkeypatch.setattr(bot.spot, "get_order", Mock(side_effect=AssertionError("open position")))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.ATTEMPT_NOT_CLOSED in codes(report) and report.calculation is None


def test_repository_exact_storage_idempotent_and_original_records_unchanged(evidence_bot):
    bot, plan, _, _ = evidence_bot
    before = source_rows(bot)
    report = reconcile_closed_attempt(bot, plan.attempt_id, capital=D("50"))
    archive = ReconciliationRepository(bot.journal)
    archive.initialize()
    archive.initialize()
    assert archive.save(report).success and archive.save(report).success
    restored = archive.get_report(plan.attempt_id)
    assert D(restored["calculation"]["net_profit"]) == D(".03993")
    assert D(restored["net_return"]) == D(".0007986")
    assert datetime.fromisoformat(restored["checked_at"]) == report.checked_at
    assert source_rows(bot) == before
    with bot.journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM testnet_trade_evidence").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM testnet_trade_reconciliations").fetchone()[0] == 1
        assert "fixture" in db.execute("SELECT evidence_json FROM testnet_trade_evidence LIMIT 1").fetchone()[0]


def test_repository_write_failure_does_not_change_source_or_result(evidence_bot, monkeypatch):
    bot, plan, _, _ = evidence_bot
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    before = repr(report)
    archive = ReconciliationRepository(bot.journal)
    monkeypatch.setattr(archive, "initialize", Mock(side_effect=ExecutionJournalError("locked")))
    saved = archive.save(report)
    assert not saved.success and saved.error_code is AccountingPersistenceCode.PERSISTENCE_FAILED
    assert repr(report) == before and bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED


def test_repository_conflicting_fill_rolls_back(evidence_bot):
    bot, plan, _, _ = evidence_bot
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    archive = ReconciliationRepository(bot.journal)
    assert archive.save(report).success
    original = archive.get_report(plan.attempt_id)
    first = report.legs[0]
    conflicting = replace(first, trades=(replace(first.trades[0], commission=D("123")),))
    saved = archive.save(replace(report, legs=(conflicting, *report.legs[1:])))
    assert saved.error_code is AccountingPersistenceCode.EVIDENCE_CONFLICT
    assert archive.get_report(plan.attempt_id) == original


def test_nullable_incomplete_report_can_be_saved(evidence_bot, monkeypatch):
    bot, plan, _, _ = evidence_bot
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=()))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    archive = ReconciliationRepository(bot.journal)
    assert archive.save(report).success
    restored = archive.get_report(plan.attempt_id)
    assert restored["calculation"] is restored["net_return"] is restored["funding_total"] is None
    assert restored["issues"][0]["code"] == Code.FUNDING_NOT_CONFIRMED.value


def test_all_recovery_exits_included_not_only_last_receipt(evidence_bot):
    bot, plan, receipts, trades = evidence_bot
    original = receipts[plan.perp_exit.client_order_id]
    receipts[plan.perp_exit.client_order_id] = replace(original, status=OrderState.EXPIRED,
        executed_quantity=D("0"), cumulative_quote_quantity=D('0'), average_price=D('0'))
    exit_trade = trades.pop("4")[0]
    rows = []
    for index, price in enumerate((D("2002"), D("2004")), start=1):
        cid, oid = f"recovery-rx{index}", str(10 + index)
        receipts[cid] = replace(original, order_id=oid, client_order_id=cid,
                                original_quantity=D(".005"), executed_quantity=D(".005"),
                                limit_price=D('2005'), cumulative_quote_quantity=price * D('.005'),
                                average_price=price)
        trades[oid] = (replace(exit_trade, trade_id=oid, order_id=oid, quantity=D(".005"), price=price,
            quote_quantity=price * D(".005"), commission=price * D(".005") * D(".0005"),
            official_perp_pnl=(D("2010") - price) * D(".005")),)
        data = {"market": "USDTM", "symbol": plan.symbol, "side": "BUY", "quantity": ".005",
                "limit_price": "2005", "client_order_id": cid, "reduce_only": True}
        rows.append((cid, plan.attempt_id, index, json.dumps(data)))
    with bot.journal._connect() as db:
        db.execute("CREATE TABLE testnet_exit_recoveries (client_order_id TEXT,attempt_id TEXT,ordinal INTEGER,order_json TEXT)")
        db.executemany("INSERT INTO testnet_exit_recoveries VALUES(?,?,?,?)", rows)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.RECONCILED
    assert report.calculation.price_pnl == D(".08")
    assert report.perp_pnl == D(".07")
    assert report.legs[-1].quantity == D(".01") and len(report.legs[-1].trades) == 2


def test_incomplete_extra_recovery_order_cannot_be_ignored(evidence_bot):
    bot, plan, receipts, _ = evidence_bot
    receipts[plan.perp_exit.client_order_id] = replace(receipts[plan.perp_exit.client_order_id], status=OrderState.NEW)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.ORDER_UNRESOLVED in codes(report) and report.calculation is None


def test_duplicate_funding_does_not_generate_profit(evidence_bot, monkeypatch):
    bot, plan, _, _ = evidence_bot
    payment = FundingPayment("same", plan.symbol, AT + timedelta(seconds=200), D("1"))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=(payment, payment)))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert Code.FUNDING_ATTRIBUTION_CONFLICT in codes(report) and report.calculation is None


def test_funding_owned_by_another_attempt_is_not_reallocated(evidence_bot):
    bot, plan, _, _ = evidence_bot
    with bot.journal._connect() as db:
        db.execute("CREATE TABLE testnet_funding_payments (transaction_id TEXT PRIMARY KEY,attempt_id TEXT)")
        db.execute("INSERT INTO testnet_funding_payments VALUES('account-payment','other-attempt')")
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.INVALID and report.calculation is None


def test_funding_network_failure_keeps_known_price_fees_but_no_net_profit(evidence_bot, monkeypatch):
    bot, plan, _, _ = evidence_bot
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(side_effect=Error(ExecutionErrorKind.NETWORK_ERROR, "TimeoutError")))
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is Status.INCOMPLETE and report.calculation is None
    assert report.price_pnl == D(".09") and report.funding_total is None


def test_fee_and_other_costs_not_deducted_twice(evidence_bot):
    bot, plan, _, _ = evidence_bot
    report = reconcile_closed_attempt(bot, plan.attempt_id, other_costs=D(".02"))
    assert report.calculation.net_profit == D(".01993")


@pytest.mark.parametrize("capital", [D("0"), D("-1"), 1.5])
def test_fictional_or_float_capital_rejected(evidence_bot, capital):
    bot, plan, _, _ = evidence_bot
    with pytest.raises((ValueError, TypeError)):
        reconcile_closed_attempt(bot, plan.attempt_id, capital=capital)


def test_repository_missing_parent_and_wrong_symbol_are_rejected(evidence_bot):
    bot, plan, _, _ = evidence_bot
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    archive = ReconciliationRepository(bot.journal)
    assert archive.save(replace(report, attempt_id="missing")).error_code is AccountingPersistenceCode.MISSING_PARENT
    assert archive.save(replace(report, symbol="BTCUSDT")).error_code is AccountingPersistenceCode.SYMBOL_MISMATCH


def test_repository_duplicate_evidence_rejected_before_any_write(evidence_bot):
    bot, plan, _, _ = evidence_bot
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    first = report.legs[0]
    report = replace(report, legs=(replace(first, trades=(*first.trades, first.trades[0])), *report.legs[1:]))
    archive = ReconciliationRepository(bot.journal)
    assert archive.save(report).error_code is AccountingPersistenceCode.EVIDENCE_CONFLICT
    with bot.journal._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM testnet_trade_evidence").fetchone()[0] == 0


def test_cli_accounting_does_not_enable_orders_or_take_execution_lock(evidence_bot, monkeypatch, capsys):
    from crypto_arbitrage_bot.execution import main as cli
    bot, plan, _, _ = evidence_bot
    monkeypatch.setattr(cli, "create_testnet_bot", lambda *args, **kwargs: bot)
    monkeypatch.setattr(cli, "execution_environment", lambda *args, **kwargs: {})
    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    monkeypatch.setattr("sys.argv", ["run_testnet.py", "--reconcile", "--attempt-id", plan.attempt_id, "--capital", "50"])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["net_profit"] == ".03993" or D(output["net_profit"]) == D(".03993")
    assert output["orders_submitted"] is False and output["status"] == "RECONCILED"
