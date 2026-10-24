"""Native abort evidence closes review only after actual sources/account audit match."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from unittest.mock import Mock

import pytest
from test_aborted_entry_recovery import aborted as aborted
from test_aborted_entry_recovery import finish, partial_exit, rows
from test_trade_accounting import AT, source_rows
from test_unhedged_entry_recovery import finish as finish_unhedged
from test_unhedged_entry_recovery import unhedged as unhedged

from crypto_arbitrage_bot.execution import abort_balance_accounting as accounting
from crypto_arbitrage_bot.execution.abort_balance_accounting import reconcile_abort_balance_evidence
from crypto_arbitrage_bot.execution.abort_balance_proof import (
    ABORT_SCOPE,
    AbortBalanceProofRepository,
)
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import (
    BalanceCheckRepository,
    BalanceCheckStatus,
)
from crypto_arbitrage_bot.execution.balance_ledger import (
    balance_entry_blocker,
    reconcile_balance_ledger,
)
from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    PairExecutionState,
    _plan_json,
)
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    OrderState,
)
from crypto_arbitrage_bot.models.market import Side

D = Decimal


def native_sources(bot, plan, monkeypatch, *, funding=".01"):
    intents = list(bot.journal.intents(plan.attempt_id))
    for table in ("testnet_spot_exit_recoveries", "testnet_exit_recoveries"):
        for row in rows(bot, table):
            data = json.loads(row["order_json"])
            from crypto_arbitrage_bot.execution.types import IocLimitOrder
            intents.append(IocLimitOrder(ExecutionMarket(data["market"]), data["symbol"], Side(data["side"]),
                                         D(data["quantity"]), D(data["limit_price"]), data["client_order_id"], data["reduce_only"]))
    receipts, trades, wallet = {}, {}, {ExecutionMarket.SPOT: D(0), ExecutionMarket.USDTM: D(0)}
    for index, order in enumerate(intents):
        client = bot.spot if order.market is ExecutionMarket.SPOT else bot.perp
        receipt = client.get_order(plan.symbol, order.client_order_id)
        receipts[order.client_order_id] = receipt
        entries = order.side is (Side.BUY if order.market is ExecutionMarket.SPOT else Side.SELL)
        when = AT + timedelta(seconds=20 + index if entries else 200 + index)
        fee = (receipt.cumulative_quote_quantity or D(0)) * D(".001")
        pnl = None if order.market is ExecutionMarket.SPOT else (D(0) if entries else -receipt.executed_quantity)
        commission, asset = (receipt.base_commission, plan.symbol[:-4]) if (
            order.market is ExecutionMarket.SPOT and receipt.base_commission) else (fee, "USDT")
        values = () if receipt.executed_quantity == 0 else (AccountTrade(order.market, plan.symbol,
            str(index + 100), receipt.order_id, order.side, order.limit_price, receipt.executed_quantity,
            receipt.cumulative_quote_quantity, commission, asset, when, pnl, '{"fixture":"native abort fill"}'),)
        trades[(order.market, receipt.order_id)] = values
        if order.market is ExecutionMarket.SPOT:
            wallet[order.market] += (-1 if entries else 1) * (receipt.cumulative_quote_quantity or D(0))
        else:
            wallet[order.market] += pnl or D(0)
        if asset == "USDT":
            wallet[order.market] -= commission
    payments = () if funding is None else (FundingPayment("abort-funding", plan.symbol, AT + timedelta(seconds=100), D(funding)),)
    if not any(order.market is ExecutionMarket.USDTM and receipts[order.client_order_id].executed_quantity for order in intents):
        payments = ()
    wallet[ExecutionMarket.USDTM] += sum((p.amount for p in payments), D(0))
    bot.clock = lambda: AT + timedelta(seconds=500)
    after = (BalanceSnapshot(ExecutionMarket.SPOT, bot.clock(), bot.clock(),
                            (AssetBalance("USDT", D("10000") + wallet[ExecutionMarket.SPOT]),
                             AssetBalance(plan.symbol[:-4], D(1)))),
             BalanceSnapshot(ExecutionMarket.USDTM, bot.clock(), bot.clock(),
                             (AssetBalance("USDT", D("10000") + wallet[ExecutionMarket.USDTM]),)))
    for client in (bot.spot, bot.perp):
        monkeypatch.setattr(client, "get_order", Mock(side_effect=lambda symbol, cid, **kwargs: receipts[cid]))
        monkeypatch.setattr(client, "get_order_trades", Mock(side_effect=lambda symbol, oid, market=client.market: trades[(market, oid)]))
        monkeypatch.setattr(client, "place_ioc_limit_order", Mock(side_effect=AssertionError("accounting cannot submit")))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=payments))
    monkeypatch.setattr(accounting, "capture_balance_snapshot", Mock(side_effect=lambda client, **kwargs: after[0 if client is bot.spot else 1]))
    return receipts, trades, after, payments


@pytest.fixture
def evidence(aborted, monkeypatch):
    bot, transport, plan, done = aborted
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.FLATTENED
    sources = native_sources(bot, plan, monkeypatch)
    return bot, transport, plan, sources


@pytest.mark.parametrize("aborted", ["BTCUSDT", "ETHUSDT"], indirect=True)
@pytest.mark.parametrize("restart", [False, True])
def test_native_partial_entry_archive_releases_review_without_normal_profit(evidence, restart):
    bot, transport, plan, _ = evidence
    original = source_rows(bot)
    count = len(transport.orders)
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.ARCHIVED and check.balance_status == "MATCHED"
    assert balance_entry_blocker(bot) is None
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["accounting_scope"] == ABORT_SCOPE
    assert report["status"] == "INCOMPLETE" and report["calculation"] is report["net_return"] is None
    assert source_rows(bot) == original and len(transport.orders) == count
    audit = BalanceAuditRepository(bot.journal).get(check.audit_id)
    assert all(D(m["unexplained_delta"]) == 0 for m in audit["movements"])
    before_count = bot.spot.get_order.call_count + bot.perp.get_order.call_count
    if restart:
        bot = Bot(bot.settings, bot.spot, bot.perp)
    reconcile_balance_ledger(bot)
    assert bot.spot.get_order.call_count + bot.perp.get_order.call_count == before_count
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.FLATTENED


@pytest.mark.parametrize("unhedged", ["BTCUSDT", "ETHUSDT"], indirect=True)
def test_unhedged_spot_only_archive_has_no_perp_orders_or_invented_funding(unhedged, monkeypatch):
    bot, transport, plan = unhedged
    finish_unhedged(bot, plan)
    native_sources(bot, plan, monkeypatch)
    original = source_rows(bot)
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    assert source_rows(bot) == original and len(transport.orders) == 2
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["funding_total"] == "0" and report["calculation"] is None
    bot.perp.get_order.assert_not_called()
    bot.perp.get_order_trades.assert_not_called()
    bot.perp.get_funding_payments.assert_not_called()


@pytest.mark.parametrize("funding", [".01", "0", "-.01", None])
def test_native_funding_signed_or_unknown_with_null_profit(aborted, monkeypatch, funding):
    bot, _, plan, _ = aborted
    finish(bot, plan)
    native_sources(bot, plan, monkeypatch, funding=funding)
    reconcile_balance_ledger(bot)
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["funding_total"] == (None if funding is None else str(D(funding)))
    assert report["calculation"] is None
    assert balance_entry_blocker(bot) is None
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.balance_status == ("INCOMPLETE" if funding is None else "MATCHED")


@pytest.mark.parametrize("market", [ExecutionMarket.SPOT, ExecutionMarket.USDTM])
def test_repair_trades_are_accounted_once_and_originals_retained(aborted, monkeypatch, market):
    bot, transport, plan, _ = aborted
    if market is ExecutionMarket.USDTM:
        bot._managed_exits()
    amount = D(".004") if market is ExecutionMarket.SPOT else D(".001")
    submit = partial_exit(bot, transport, monkeypatch, market, amount)
    bot._managed_exits()
    monkeypatch.setattr(bot.spot if market is ExecutionMarket.SPOT else bot.perp, "place_ioc_limit_order", submit)
    finish(bot, plan)
    native_sources(bot, plan, monkeypatch)
    original = source_rows(bot)
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    audit = BalanceAuditRepository(bot.journal).get(check.audit_id)
    assert len(audit["trades"]) == 5
    assert len({(t["market"], t["trade_id"]) for t in audit["trades"]}) == 5
    assert all(D(m["unexplained_delta"]) == 0 for m in audit["movements"])
    assert source_rows(bot) == original


def test_balance_checkpoint_crash_recovers_committed_abort_proof_without_network(evidence, monkeypatch):
    bot, _, plan, _ = evidence
    finish_check = BalanceCheckRepository.finish
    monkeypatch.setattr(BalanceCheckRepository, "finish", Mock(return_value=False))
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).status is BalanceCheckStatus.RUNNING
    count = bot.spot.get_order.call_count + bot.perp.get_order.call_count
    assert balance_entry_blocker(bot) == "BALANCE_CHECK_UNRESOLVED"
    monkeypatch.setattr(BalanceCheckRepository, "finish", finish_check)
    restarted = Bot(bot.settings, bot.spot, bot.perp)
    reconcile_balance_ledger(restarted)
    assert balance_entry_blocker(restarted) is None
    assert bot.spot.get_order.call_count + bot.perp.get_order.call_count == count


@pytest.mark.parametrize("which", ["spot_order", "perp_trades", "funding", "balance"])
def test_api_failure_retains_gate_and_source_records_with_bounded_retry(evidence, monkeypatch, which):
    bot, _, plan, _ = evidence
    original = source_rows(bot)
    error = GatewayError(ExecutionErrorKind.NETWORK_ERROR, "private-payload-never-log", market=ExecutionMarket.SPOT)
    obj, field = {"spot_order": (bot.spot, "get_order"), "perp_trades": (bot.perp, "get_order_trades"),
                  "funding": (bot.perp, "get_funding_payments"), "balance": (accounting, "capture_balance_snapshot")}[which]
    failure = Mock(side_effect=error)
    monkeypatch.setattr(obj, field, failure)
    for _ in range(5):
        reconcile_balance_ledger(bot)
    first = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert first.status is BalanceCheckStatus.PENDING and first.attempts == 1
    assert first.failure_code == "NETWORK_ERROR" and failure.call_count == 1
    assert source_rows(bot) == original and balance_entry_blocker(bot) is not None
    for _ in range(bot.config.balance_check_max_attempts):
        at = bot.clock()
        bot.clock = lambda at=at: at + timedelta(seconds=bot.config.balance_check_interval_seconds)
        reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).status is BalanceCheckStatus.FAILED
    assert failure.call_count == bot.config.balance_check_max_attempts


@pytest.mark.parametrize("case", ["missing_fill", "wrong_order", "wrong_symbol", "changed_quote", "outside_before", "duplicate_fill", "active_order"])
def test_corrupt_trade_or_receipt_evidence_never_clears_review(evidence, case):
    bot, _, plan, (receipts, trades, _, _) = evidence
    key = (ExecutionMarket.SPOT, receipts[plan.spot_entry.client_order_id].order_id)
    trade = trades[key][0]
    if case == "missing_fill":
        trades[key] = ()
    elif case == "wrong_order":
        trades[key] = (replace(trade, order_id="999"),)
    elif case == "wrong_symbol":
        trades[key] = (replace(trade, symbol="BTCUSDT"),)
    elif case == "changed_quote":
        trades[key] = (replace(trade, quote_quantity=trade.quote_quantity + 1),)
    elif case == "outside_before":
        trades[key] = (replace(trade, traded_at=AT),)
    elif case == "duplicate_fill":
        trades[key] = (replace(trade, quantity=trade.quantity / 2, quote_quantity=trade.quote_quantity / 2),) * 2
    else:
        cid = plan.spot_entry.client_order_id
        receipts[cid] = replace(receipts[cid], status=OrderState.NEW)
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is not None
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.FLATTENED


@pytest.mark.parametrize("case", ["native_position", "other_symbol_position", "open_orders", "missing_asset", "unexplained_quote"])
def test_native_conflicts_cannot_be_hidden_by_flat_journal(evidence, monkeypatch, case):
    bot, _, plan, (_, _, after, _) = evidence
    if case in ("native_position", "other_symbol_position", "open_orders"):
        get_state = bot.perp.get_account_state
        def altered(symbol):
            state = get_state(symbol)
            if case == "open_orders":
                return replace(state, open_order_count=1)
            if symbol == (plan.symbol if case == "native_position" else "BTCUSDT"):
                return replace(state, position_quantity=D("-.01"))
            return state
        monkeypatch.setattr(bot.perp, "get_account_state", altered)
    else:
        balances = (AssetBalance("USDT", D("9000")),) if case == "unexplained_quote" else after[0].balances[:1]
        bad = replace(after[0], balances=balances)
        monkeypatch.setattr(accounting, "capture_balance_snapshot", lambda c, **kw: bad if c is bot.spot else after[1])
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is not None


@pytest.mark.parametrize("change", ["parent", "audit", "source_report", "marker", "repair"])
def test_archived_proof_invalidated_by_source_mutation(evidence, change):
    bot, _, plan, _ = evidence
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    with bot.journal._connect() as db:
        if change == "parent":
            db.execute("UPDATE testnet_pair_attempts SET failure_reason='changed' WHERE attempt_id=?", (plan.attempt_id,))
        elif change == "audit":
            db.execute("UPDATE testnet_balance_audits SET report_json='{}'")
        elif change == "source_report":
            db.execute("UPDATE testnet_trade_reconciliations SET report_json='{}'")
        elif change == "marker":
            db.execute("UPDATE testnet_balance_frame_intents SET frame_sha256='changed'")
        else:
            db.execute("INSERT INTO testnet_exit_recoveries VALUES('fake',?,1,'{}','{}',NULL,?)",
                       (plan.attempt_id, bot.clock().isoformat()))
    assert balance_entry_blocker(bot) is not None


def test_old_currency_archive_alone_cannot_release_abort_review(evidence):
    bot, _, plan, _ = evidence
    outcome = reconcile_abort_balance_evidence(bot, bot.balance_directory, plan.attempt_id)
    assert outcome.status.value == "ARCHIVED"
    with bot.journal._connect() as db:
        db.execute("DELETE FROM testnet_abort_balance_proofs")
    BalanceCheckRepository(bot.journal).recover_archive(plan.attempt_id, outcome.audit_id, bot.clock())
    assert balance_entry_blocker(bot) == "BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW"


def test_persistence_failure_does_not_alter_abort_or_submit_orders(evidence, monkeypatch):
    bot, _, plan, _ = evidence
    original = source_rows(bot)
    monkeypatch.setattr(AbortBalanceProofRepository, "save", Mock(side_effect=ExecutionJournalError("locked")))
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).status is BalanceCheckStatus.PENDING
    assert source_rows(bot) == original and balance_entry_blocker(bot) is not None
    bot.spot.place_ioc_limit_order.assert_not_called()
    bot.perp.place_ioc_limit_order.assert_not_called()


def test_low_decimal_context_retains_exact_currency_accounting(evidence):
    bot, _, plan, _ = evidence
    with localcontext() as context:
        context.prec = 3
        reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["funding_total"] == "0.01" and report["calculation"] is None


def test_stop_and_active_exposure_precede_optional_abort_reads(evidence):
    bot, _, plan, _ = evidence
    bot.stop.set()
    reconcile_balance_ledger(bot)
    assert bot.spot.get_order.call_count == 0
    bot.stop.clear()
    bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED)
    reconcile_balance_ledger(bot)
    assert bot.spot.get_order.call_count == 0


def test_schema_idempotence_and_foreign_keys(evidence):
    bot, _, plan, _ = evidence
    repo = AbortBalanceProofRepository(bot.journal)
    repo.initialize()
    repo.initialize()
    with bot.journal._connect() as db:
        # SQLite foreign-key settings are per connection. Repository writes
        # enable them explicitly; this separate inspection connection does too.
        db.execute("PRAGMA foreign_keys=ON")
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert len(db.execute("PRAGMA foreign_key_list(testnet_abort_balance_proofs)").fetchall()) == 2
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id) is None


@pytest.mark.parametrize("kind", ["spot_no_fill", "perp_no_fill", "base_fee"])
def test_zero_and_base_asset_fee_abort_paths_are_auditable(request, monkeypatch, kind):
    if kind == "perp_no_fill":
        bot, transport, plan, _ = request.getfixturevalue("aborted")
        transport.positions[plan.symbol] = D(0)
        transport.orders[plan.perp_entry.client_order_id].update(status="EXPIRED", executedQty="0", cummulativeQuoteQty="0", avgPrice='0')
        finish(bot, plan)
    else:
        bot, transport, plan = request.getfixturevalue("unhedged")
        row = transport.orders[plan.spot_entry.client_order_id]
        if kind == "spot_no_fill":
            transport.base[plan.symbol] = D(1)
            row.update(status="EXPIRED", executedQty="0", cummulativeQuoteQty="0", avgPrice='0', fills=[])
        else:
            transport.base[plan.symbol] -= D(".001")
            row["fills"][0].update(commission=".001", commissionAsset=plan.symbol[:-4])
            original_query = bot.spot.get_order
            def query(symbol, client_order_id, **kwargs):
                receipt = original_query(symbol, client_order_id, **kwargs)
                return replace(receipt, base_commission=D(".001")) if client_order_id == plan.spot_entry.client_order_id else receipt
            monkeypatch.setattr(bot.spot, "get_order", query)
        finish_unhedged(bot, plan)
    native_sources(bot, plan, monkeypatch)
    original = source_rows(bot)
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["funding_total"] == "0" and report["calculation"] is None
    assert source_rows(bot) == original
    bot.perp.get_funding_payments.assert_not_called()
    if kind == "spot_no_fill":
        assert bot.journal.get(plan.attempt_id).state is PairExecutionState.FAILED
        bot.spot.get_order_trades.assert_not_called()


def replace_stored_plan(bot, plan):
    encoded = _plan_json(plan)
    with bot.journal._connect() as db:
        db.execute("UPDATE testnet_pair_attempts SET plan_json=?,plan_sha256=? WHERE attempt_id=?",
                   (encoded, hashlib.sha256(encoded.encode()).hexdigest(), plan.attempt_id))


def test_planned_settlement_missing_evidence_retries_then_accepts_explicit_zero(aborted, monkeypatch):
    bot, _, original, _ = aborted
    finish(bot, original)
    plan = replace(original, funding_settlement_at=AT + timedelta(seconds=100),
                   scheduled_exit_at=AT + timedelta(seconds=101), opportunity_id="synthetic-opportunity")
    replace_stored_plan(bot, plan)
    native_sources(bot, plan, monkeypatch, funding=None)
    reconcile_balance_ledger(bot)
    first = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert first.status is BalanceCheckStatus.PENDING and first.failure_code == "FUNDING_EVIDENCE_PENDING"
    assert balance_entry_blocker(bot) is not None
    payment = FundingPayment("explicit-zero", plan.symbol, AT + timedelta(seconds=100), D(0))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=(payment,)))
    bot.clock = lambda: AT + timedelta(seconds=500 + bot.config.balance_check_interval_seconds)
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.ARCHIVED and check.attempts == 2
    assert balance_entry_blocker(bot) is None
    report = ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    assert report["funding_total"] == "0" and len(report["funding_payments"]) == 1
    assert report["calculation"] is report["net_return"] is None


@pytest.mark.parametrize("settlement", [100, 400])
def test_actual_holding_window_decides_if_planned_funding_must_be_confirmed(aborted, monkeypatch, settlement):
    bot, _, original, _ = aborted
    finish(bot, original)
    plan = replace(original, funding_settlement_at=AT + timedelta(seconds=settlement),
                   scheduled_exit_at=AT + timedelta(seconds=settlement + 1), opportunity_id="synthetic-opportunity")
    replace_stored_plan(bot, plan)
    native_sources(bot, plan, monkeypatch, funding=None)
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    if settlement == 100:
        assert check.status is BalanceCheckStatus.PENDING and check.failure_code == "FUNDING_EVIDENCE_PENDING"
        assert balance_entry_blocker(bot) is not None
    else:
        assert check.status is BalanceCheckStatus.ARCHIVED and check.balance_status == "INCOMPLETE"
        assert balance_entry_blocker(bot) is None


@pytest.mark.parametrize("case", ["wrong_symbol", "at_entry", "after_exit", "duplicate", "owned_elsewhere", "report_owned_elsewhere"])
def test_funding_cannot_be_misattributed_to_aborted_hold(evidence, monkeypatch, case):
    bot, _, plan, (_, trades, _, payments) = evidence
    payment = payments[0]
    if case == "wrong_symbol":
        payments = (replace(payment, symbol="BTCUSDT"),)
    elif case == "at_entry":
        at = next(t[0].traded_at for (market, _), t in trades.items() if market is ExecutionMarket.USDTM and t[0].side is Side.SELL)
        payments = (replace(payment, posted_at=at),)
    elif case == "after_exit":
        payments = (replace(payment, posted_at=AT + timedelta(seconds=400)),)
    elif case == "duplicate":
        payments = payments * 2
    elif case == "owned_elsewhere":
        with bot.journal._connect() as db:
            db.execute("CREATE TABLE testnet_funding_payments (transaction_id TEXT PRIMARY KEY,attempt_id TEXT)")
            db.execute("INSERT INTO testnet_funding_payments VALUES(?, 'other-attempt')", (payment.transaction_id,))
    else:
        ReconciliationRepository(bot.journal).initialize()
        with bot.journal._connect() as db:
            db.execute("INSERT INTO testnet_trade_reconciliations VALUES(?,1,?,?,?,?)", (
                "other-attempt", plan.symbol, "INCOMPLETE", AT.isoformat(),
                json.dumps({"funding_payments": [{"transaction_id": payment.transaction_id}]})))
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=payments))
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is not None
    assert not BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)


@pytest.mark.parametrize("column,body", [("account_states_json", "[]"), ("journal_sha256", "changed"),
                                       ("audit_sha256", "changed"), ("reconciliation_sha256", "changed")])
def test_durable_proof_mutation_never_releases_review(evidence, column, body):
    bot, _, plan, _ = evidence
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    with bot.journal._connect() as db:
        db.execute(f"UPDATE testnet_abort_balance_proofs SET {column}=?", (body,))
    assert balance_entry_blocker(bot) is not None


def test_native_trade_source_mutation_invalidates_existing_proof(evidence):
    bot, _, _, _ = evidence
    reconcile_balance_ledger(bot)
    assert balance_entry_blocker(bot) is None
    with bot.journal._connect() as db:
        db.execute("UPDATE testnet_trade_evidence SET evidence_json='{}'")
    assert balance_entry_blocker(bot) is not None
