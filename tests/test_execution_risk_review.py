"""Native-shaped synthetic source graph tests; NOT actual engineering proof."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.arbitrage.profit import net_return, round_trip_from_fills
from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import risk_review as module
from crypto_arbitrage_bot.execution.risk_review import RiskReviewError
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    LegTradeEvidence,
    ReconciliationStatus,
    TradeReconciliation,
)
from crypto_arbitrage_bot.execution.types import AccountTrade, ExecutionMarket
from crypto_arbitrage_bot.models.calculation import FillSlice, FillStatus, SimulatedFill, TradeLeg
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.models.strategy_admission import AdmissionReason as R
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase as K
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

D = Decimal
AT = OPENED + timedelta(days=4)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("risk review is offline"))
    )


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Graph:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.path = tmp_path / "synthetic-risk.sqlite"
        base = load_settings(Path("docs/testnet-bot.toml"))
        self.settings = base.model_copy(update={"symbols": ("BTCUSDT",)})
        self.policy = module.execution_policy_fingerprint(self.settings)
        self.implementation = module.implementation_fingerprint()
        self.data = {
            "schema_version": 1,
            "environment": "TESTNET",
            "registered_at": AT - timedelta(seconds=10),
            "reviewed_at": AT + timedelta(seconds=150),
            "expires_at": AT + timedelta(days=1),
            "policy_sha256": self.policy,
            "implementation_sha256": self.implementation,
            "journal_path": str(self.path.resolve()),
            "qualified_quantities": (("BTCUSDT", D(".001")),),
            "cases": [],
        }
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
                CREATE TABLE testnet_pair_attempts(attempt_id TEXT PRIMARY KEY,plan_json TEXT,plan_sha256 TEXT,
                    state TEXT,spot_entry TEXT,perp_entry TEXT,spot_exit TEXT,perp_exit TEXT,
                    failure_code TEXT,failure_reason TEXT,created_at TEXT,updated_at TEXT);
                CREATE TABLE testnet_order_intents(client_order_id TEXT,attempt_id TEXT,order_json TEXT,created_at TEXT);
                CREATE TABLE testnet_trade_reconciliations(attempt_id TEXT,status TEXT,report_json TEXT);
                CREATE TABLE testnet_balance_audits(audit_id TEXT,attempt_id TEXT,status TEXT,report_json TEXT);
                CREATE TABLE testnet_trade_evidence(market TEXT,symbol TEXT,trade_id TEXT,attempt_id TEXT,evidence_json TEXT);
                CREATE TABLE testnet_funding_payments(transaction_id TEXT,attempt_id TEXT,symbol TEXT,posted_at TEXT,amount TEXT);
            """)
        for index, kind in enumerate(K):
            self.add_case(kind, index)

    def add_case(self, kind, index):
        aid = f"synthetic-case-{index}"
        legs = ("spot_entry", "perp_entry", "spot_exit", "perp_exit")
        orders, receipts, entries, trade_rows = {}, {}, [], []
        repair = None
        for i, leg in enumerate(legs):
            market = "SPOT" if leg.startswith("spot") else "USDTM"
            side = "BUY" if leg in ("spot_entry", "perp_exit") else "SELL"
            order = {
                "market": market,
                "side": side,
                "symbol": "BTCUSDT",
                "quantity": "0.001",
                "limit_price": "50000",
                "client_order_id": f"{aid}-{i}",
                "reduce_only": leg == "perp_exit",
            }
            orders[leg] = order
            qty = (
                D(".0005") if kind is K.PARTIAL_ENTRY_RECOVERY and market == "USDTM" else D(".001")
            )
            if kind is K.SPOT_ACK_RECOVERY and market == "USDTM":
                receipts[leg] = None
                entries.append({"leg": leg.upper(), "trades": []})
                continue
            receipt = {
                "market": market,
                "symbol": "BTCUSDT",
                "client_order_id": order["client_order_id"],
                "order_id": str(index * 10 + i + 1),
                "original_quantity": "0.001",
                "executed_quantity": str(qty),
                "status": "FILLED" if qty == D(".001") else "EXPIRED",
                "cumulative_quote_quantity": str(qty * 50000),
                "average_price": "50000",
                "base_commission": "0",
            }
            receipts[leg] = receipt
            parts = [(receipt["order_id"], qty)]
            if kind is K.PARTIAL_EXIT_RECOVERY and leg == "spot_exit":
                parts = [(receipt["order_id"], qty / 2), (str(index * 10 + 9), qty / 2)]
                receipt.update(
                    executed_quantity="0.0005", cumulative_quote_quantity="25", status="EXPIRED"
                )
                repair_order = {**order, "quantity": "0.0005", "client_order_id": aid + "-repair"}
                repair_receipt = {
                    **receipt,
                    "client_order_id": repair_order["client_order_id"],
                    "order_id": str(index * 10 + 9),
                    "original_quantity": "0.0005",
                    "status": "FILLED",
                }
                repair = (repair_order, repair_receipt)
            trades = []
            for part, (oid, amount) in enumerate(parts):
                raw = {
                    "id": index * 100 + i * 10 + part + 1,
                    "orderId": int(oid),
                    "price": "50000",
                    "qty": str(amount),
                    "commission": "0.0005" if len(parts) == 2 else "0.001",
                    "commissionAsset": "USDT",
                    "quoteQty": str(amount * 50000),
                    "symbol": "BTCUSDT",
                    "side": side,
                    "realizedPnl": "0",
                    "isBuyer": side == "BUY",
                    "buyer": side == "BUY",
                    "time": (
                        AT
                        + timedelta(seconds=1 if leg.endswith("entry") else 90)
                        - datetime(1970, 1, 1, tzinfo=UTC)
                    )
                    // timedelta(milliseconds=1),
                }
                trade = {
                    "market": market,
                    "symbol": "BTCUSDT",
                    "trade_id": str(raw["id"]),
                    "order_id": oid,
                    "side": side,
                    "price": "50000",
                    "quantity": str(amount),
                    "quote_quantity": str(amount * 50000),
                    "commission": raw["commission"],
                    "commission_asset": "USDT",
                    "traded_at": (
                        AT + timedelta(seconds=1 if leg.endswith("entry") else 90)
                    ).isoformat(timespec="microseconds"),
                    "official_perp_pnl": "0" if market == "USDTM" else None,
                    "raw_json": exact_json(raw),
                }
                trades.append(trade)
                trade_rows.append((market, "BTCUSDT", trade["trade_id"], aid, exact_json(trade)))
            entries.append({"leg": leg.upper(), "trades": trades})
        plan = exact_json({"attempt_id": aid, **orders})
        events = []
        if kind is K.SPOT_ACK_RECOVERY:
            events.append(
                {
                    "kind": "LOCAL_ACK_DELIVERY_INTERRUPT",
                    "market": "SPOT",
                    "client_order_id": orders["spot_entry"]["client_order_id"],
                }
            )
        elif kind is K.PERP_UNKNOWN_RECOVERY:
            events.extend(
                (
                    {
                        "kind": "POST_RESULT_UNKNOWN",
                        "market": "USDTM",
                        "error_kind": "TIMEOUT",
                        "client_order_id": orders["perp_entry"]["client_order_id"],
                    },
                    {
                        "kind": "FIXED_ID_QUERY_RESOLVED",
                        "market": "USDTM",
                        "order_id": receipts["perp_entry"]["order_id"],
                        "client_order_id": orders["perp_entry"]["client_order_id"],
                    },
                )
            )
        elif kind is K.PARTIAL_EXIT_RECOVERY:
            events.append(
                {
                    "kind": "PARTIAL_EXIT_TERMINAL",
                    "market": "SPOT",
                    "native_order": {
                        "orderId": int(receipts["spot_exit"]["order_id"]),
                        "clientOrderId": orders["spot_exit"]["client_order_id"],
                        "origQty": "0.001",
                        "executedQty": "0.0005",
                        "status": "EXPIRED",
                    },
                }
            )
        payments = (
            []
            if kind is K.SPOT_ACK_RECOVERY
            else [
                {
                    "transaction_id": aid + "-funding",
                    "symbol": "BTCUSDT",
                    "posted_at": (AT + timedelta(seconds=45)).isoformat(timespec="microseconds"),
                    "amount": "0.01",
                    "asset": "USDT",
                }
            ]
        )
        with localcontext() as ctx:
            ctx.prec = 50
            grouped = []
            for entry in entries:
                decoded = tuple(
                    AccountTrade(
                        ExecutionMarket(t["market"]),
                        t["symbol"],
                        t["trade_id"],
                        t["order_id"],
                        Side(t["side"]),
                        D(t["price"]),
                        D(t["quantity"]),
                        D(t["quote_quantity"]),
                        D(t["commission"]),
                        t["commission_asset"],
                        datetime.fromisoformat(t["traded_at"]),
                        None if t["official_perp_pnl"] is None else D(t["official_perp_pnl"]),
                        t["raw_json"],
                    )
                    for t in entry["trades"]
                )
                quantity = sum((t.quantity for t in decoded), D(0))
                quote = sum((t.quote_quantity for t in decoded), D(0))
                grouped.append(
                    LegTradeEvidence(
                        TradeLeg(entry["leg"]),
                        decoded,
                        quantity,
                        quote,
                        None if not quantity else quote / quantity,
                        ()
                        if not decoded
                        else (("USDT", sum((t.commission for t in decoded), D(0))),),
                    )
                )
            grouped.sort(key=lambda g: list(TradeLeg).index(g.leg))
            full = kind not in (K.SPOT_ACK_RECOVERY, K.PARTIAL_ENTRY_RECOVERY)
            fees = (
                FourLegFees(*(dict(g.commissions).get("USDT", D(0)) for g in grouped))
                if full
                else None
            )
            calc = None
            if full:
                fills = [
                    SimulatedFill(
                        "BTCUSDT",
                        g.leg,
                        g.quantity,
                        g.quantity,
                        g.quote_quantity,
                        g.vwap,
                        FillStatus.FILLED,
                        tuple(FillSlice(t.price, t.quantity) for t in g.trades),
                        g.trades[0].price,
                    )
                    for g in grouped
                ]
                calc = round_trip_from_fills(*fills, fees, D(".01"))
            accounting = json.loads(
                exact_json(
                    TradeReconciliation(
                        aid,
                        "BTCUSDT",
                        AT + timedelta(seconds=125),
                        ReconciliationStatus.RECONCILED
                        if full
                        else ReconciliationStatus.INCOMPLETE,
                        (),
                        legs=tuple(grouped),
                        funding_payments=tuple(payments),
                        funding_total=D(".01") if full else None,
                        official_perp_pnl=D(0) if full else None,
                        spot_inventory_change=grouped[0].quantity - grouped[1].quantity,
                        fees=fees,
                        calculation=calc,
                        capital=D(100),
                        net_return=None if calc is None else net_return(calc.net_profit, D(100)),
                        price_pnl=calc.price_pnl if full else None,
                        spot_pnl=grouped[1].quote_quantity - grouped[0].quote_quantity
                        if full
                        else None,
                        perp_pnl=grouped[2].quote_quantity - grouped[3].quote_quantity
                        if full
                        else None,
                        entry_time=AT + timedelta(seconds=1) if full else None,
                        exit_time=AT + timedelta(seconds=90) if full else None,
                    )
                )
            )
            entries = json.loads(exact_json(grouped))
            # Match the real repository: raw fills occur once in source rows,
            # while the accounting report contains exact numeric/ID projections.
            for leg in accounting["legs"]:
                for trade in leg["trades"]:
                    trade.pop("raw_json")
        frames = {}
        for phase in ("before", "after"):
            values = []
            for market in ("SPOT", "USDTM"):
                usdt = (
                    "1000"
                    if phase == "before"
                    else (
                        "999.998" if market == "SPOT" else ("1000" if not payments else "1000.008")
                    )
                )
                balances = [{"asset": "USDT", "amount": usdt}] + (
                    [{"asset": "BTC", "amount": "1"}] if market == "SPOT" else []
                )
                raw = (
                    {
                        "balances": [
                            {"asset": b["asset"], "free": b["amount"], "locked": "0"}
                            for b in balances
                        ]
                    }
                    if market == "SPOT"
                    else {"assets": [{"asset": "USDT", "walletBalance": usdt}]}
                )
                when = AT + timedelta(seconds=-1 if phase == "before" else 120)
                values.append(
                    {
                        "market": market,
                        "requested_at": when,
                        "received_at": when,
                        "balances": balances,
                        "raw_balances_json": exact_json(raw),
                    }
                )
            frames[phase] = values
        balance = {
            "status": "MATCHED",
            "issues": [],
            "funding_evidence_complete": True,
            "scope": "TESTNET_CURRENCY_BALANCE_V1_NOT_PROFIT",
            "movements": [{"unexplained_delta": "0"}],
            **frames,
            "trades": [t for e in entries for t in e["trades"]],
            "funding": payments,
            "adjustments": [],
        }
        native = exact_json(balance)
        report = exact_json(accounting)
        audit_id = aid + "-audit"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO testnet_pair_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    aid,
                    plan,
                    digest(plan),
                    "FLATTENED"
                    if kind in (K.SPOT_ACK_RECOVERY, K.PARTIAL_ENTRY_RECOVERY)
                    else "CLOSED",
                    *(None if receipts[leg] is None else exact_json(receipts[leg]) for leg in legs),
                    None,
                    None,
                    AT.isoformat(timespec="microseconds"),
                    (AT + timedelta(seconds=100)).isoformat(timespec="microseconds"),
                ),
            )
            for name, order in orders.items():
                if receipts[name] is None:
                    continue
                db.execute(
                    "INSERT INTO testnet_order_intents VALUES(?,?,?,?)",
                    (order["client_order_id"], aid, exact_json(order), AT.isoformat()),
                )
            db.execute(
                "INSERT INTO testnet_trade_reconciliations VALUES(?,?,?)",
                (aid, accounting["status"], report),
            )
            db.execute(
                "INSERT INTO testnet_balance_audits VALUES(?,?,?,?)",
                (audit_id, aid, "MATCHED", native),
            )
            db.executemany("INSERT INTO testnet_trade_evidence VALUES(?,?,?,?,?)", trade_rows)
            for payment in payments:
                db.execute(
                    "INSERT INTO testnet_funding_payments VALUES(?,?,?,?,?)",
                    (
                        payment["transaction_id"],
                        aid,
                        "BTCUSDT",
                        payment["posted_at"],
                        payment["amount"],
                    ),
                )
            if repair:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS testnet_spot_exit_recoveries(attempt_id TEXT,order_json TEXT,receipt_json TEXT,created_at TEXT)"
                )
                db.execute(
                    "INSERT INTO testnet_spot_exit_recoveries VALUES(?,?,?,?)",
                    (aid, exact_json(repair[0]), exact_json(repair[1]), AT.isoformat()),
                )
            db.row_factory = sqlite3.Row
            attempt = dict(
                db.execute(
                    "SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (aid,)
                ).fetchone()
            )
            intents = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM testnet_order_intents WHERE attempt_id=? ORDER BY client_order_id",
                    (aid,),
                )
            ]
        if kind is K.RESTART_IDEMPOTENCY:
            hashed = digest(exact_json(intents))
            events.append(
                {
                    "kind": "TERMINAL_RESTART_NO_NEW_INTENTS",
                    "before_intents_sha256": hashed,
                    "after_intents_sha256": hashed,
                    "entry_post_count": 0,
                }
            )
        proof = {
            "environment": "TESTNET",
            "evidence_origin": "NATIVE_TESTNET",
            "attempt_id": aid,
            "case": kind.value,
            "policy_sha256": self.policy,
            "implementation_sha256": self.implementation,
            "started_at": AT - timedelta(seconds=2),
            "finished_at": AT + timedelta(seconds=130),
            "attempt_sha256": digest(exact_json(attempt)),
            "audit_id": audit_id,
            "audit_sha256": digest(native),
            "accounting_sha256": digest(report),
            "events": events,
        }
        path = self.root / f"synthetic-{index}.json"
        path.write_text(exact_json(proof), encoding="utf-8")
        self.data["cases"].append(
            {
                "case": kind.value,
                "symbol": "BTCUSDT",
                "attempt_id": aid,
                "proof_path": str(path.resolve()),
                "proof_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    def reference(self):
        path = self.root / "synthetic-review.json"
        path.write_text(exact_json(self.data), encoding="utf-8")
        return StudyRegistrationSettings(
            path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def read(self):
        return module.read_execution_risk_review(
            self.reference(), self.settings, evaluated_at=AT + timedelta(seconds=180)
        )

    def change_proof(self, kind, change):
        item = next(item for item in self.data["cases"] if item["case"] == kind.value)
        path = Path(item["proof_path"])
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        path.write_text(exact_json(value), encoding="utf-8")
        item["proof_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def graph(tmp_path):
    return Graph(tmp_path)


def test_complete_synthetic_graph_verifies_each_case_and_leaves_database_unchanged(graph):
    before = graph.path.read_bytes()
    result = graph.read()
    assert len(result.cases) == len(K) and result.qualified_quantities == (("BTCUSDT", D(".001")),)
    assert graph.path.read_bytes() == before
    assert result.reviewed_at.tzinfo is not None and isinstance(
        result.qualified_quantities[0][1], D
    )


@pytest.mark.parametrize("case", list(K))
def test_missing_one_case_is_not_complete(graph, case):
    graph.data["cases"] = [item for item in graph.data["cases"] if item["case"] != case.value]
    with pytest.raises(RiskReviewError) as error:
        graph.read()
    assert error.value.reason is R.RISK_REVIEW_INCOMPLETE


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("environment", "MOCK", R.RISK_REVIEW_UNAVAILABLE),
        ("policy_sha256", "0" * 64, R.RISK_POLICY_MISMATCH),
        ("implementation_sha256", "0" * 64, R.IMPLEMENTATION_MISMATCH),
        ("expires_at", AT + timedelta(seconds=180), R.RISK_REVIEW_EXPIRED),
        ("schema_version", 99, R.RISK_REVIEW_UNAVAILABLE),
    ],
)
def test_review_version_and_bindings_must_match(graph, field, value, reason):
    graph.data[field] = value
    with pytest.raises(RiskReviewError) as error:
        graph.read()
    assert error.value.reason is reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence_origin", "MOCK"),
        ("policy_sha256", "0" * 64),
        ("implementation_sha256", "0" * 64),
        ("attempt_sha256", "0" * 64),
        ("audit_sha256", "0" * 64),
        ("accounting_sha256", "0" * 64),
        ("started_at", AT + timedelta(seconds=1)),
        ("events", []),
    ],
)
def test_case_claims_cannot_replace_verified_native_sources(graph, field, value):
    graph.change_proof(K.PERP_UNKNOWN_RECOVERY, lambda proof: proof.update({field: value}))
    with pytest.raises(RiskReviewError):
        graph.read()


def test_spot_ack_loss_is_not_perp_post_unknown(graph):
    graph.change_proof(
        K.PERP_UNKNOWN_RECOVERY,
        lambda proof: proof.update(
            {
                "events": [
                    {
                        "kind": "LOCAL_ACK_DELIVERY_INTERRUPT",
                        "market": "SPOT",
                        "client_order_id": "unrelated",
                    }
                ]
            }
        ),
    )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_full_exit_cannot_be_called_partial(graph):
    graph.change_proof(
        K.PARTIAL_EXIT_RECOVERY,
        lambda proof: proof["events"][0]["native_order"].update({"status": "FILLED"}),
    )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_restart_with_new_entry_intents_is_not_idempotent(graph):
    graph.change_proof(
        K.RESTART_IDEMPOTENCY, lambda proof: proof["events"][0].update({"entry_post_count": 1})
    )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_summary_pass_without_case_proofs_is_rejected(graph):
    graph.data["cases"] = []
    graph.data["status"] = "PASS"
    graph.data["pytest_passed"] = 99999
    with pytest.raises(RiskReviewError) as error:
        graph.read()
    assert error.value.reason is R.RISK_REVIEW_INCOMPLETE


def test_original_case_proof_modified_without_new_pin_is_rejected(graph):
    item = graph.data["cases"][0]
    Path(item["proof_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(RiskReviewError):
        graph.read()


def test_readonly_reader_missing_db_never_initializes_it(graph):
    graph.path.unlink()
    with pytest.raises(RiskReviewError):
        graph.read()
    assert not graph.path.exists()


def test_all_reads_are_ro_not_signed_or_order_requests(graph, monkeypatch):
    connect = sqlite3.connect
    statements = []

    def read_only(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"]
        db = connect(database, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", read_only)
    graph.read()
    assert not any(
        s.upper().startswith(("INSERT", "UPDATE", "CREATE", "DELETE")) for s in statements
    )


def test_wrong_native_funding_amount_does_not_count_as_accounted(graph):
    with closing(sqlite3.connect(graph.path)) as db, db:
        db.execute(
            "UPDATE testnet_funding_payments SET amount='999' WHERE attempt_id=?",
            (
                next(
                    c["attempt_id"]
                    for c in graph.data["cases"]
                    if c["case"] == K.FEES_FUNDING_BALANCE.value
                ),
            ),
        )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_case_cannot_qualify_larger_quantity_than_was_exercised(graph):
    graph.data["qualified_quantities"] = (("BTCUSDT", D(".002")),)
    with pytest.raises(RiskReviewError):
        graph.read()


def test_duplicate_case_cannot_make_up_for_missing_case(graph):
    graph.data["cases"][-1] = graph.data["cases"][0]
    with pytest.raises(RiskReviewError):
        graph.read()


def test_manually_claimed_matched_audit_with_inconsistent_native_wallet_fails(graph):
    item = graph.data["cases"][0]
    path = Path(item["proof_path"])
    proof = json.loads(path.read_text(encoding="utf-8"))
    with closing(sqlite3.connect(graph.path)) as db, db:
        report = json.loads(
            db.execute(
                "SELECT report_json FROM testnet_balance_audits WHERE audit_id=?",
                (proof["audit_id"],),
            ).fetchone()[0]
        )
        wallet = report["after"][1]
        wallet["balances"][0]["amount"] = "900"
        wallet["raw_balances_json"] = exact_json(
            {"assets": [{"asset": "USDT", "walletBalance": "900"}]}
        )
        text = exact_json(report)
        db.execute(
            "UPDATE testnet_balance_audits SET report_json=? WHERE audit_id=?",
            (text, proof["audit_id"]),
        )
    graph.change_proof(
        K.NORMAL_ENTRY_EXIT, lambda value: value.update({"audit_sha256": digest(text)})
    )
    with pytest.raises(RiskReviewError):
        graph.read()
