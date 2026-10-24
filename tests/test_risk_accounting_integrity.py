"""Offline source graph contradictions must never qualify native risk cases."""

import json
import sqlite3
from contextlib import closing
from decimal import Decimal

import pytest
from test_execution_risk_review import AT, Graph, digest

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.arbitrage.profit import net_profit, net_return
from crypto_arbitrage_bot.execution.risk_review import RiskReviewError
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase as K
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


@pytest.fixture
def graph(tmp_path):
    return Graph(tmp_path)


def case(graph, kind=K.NORMAL_ENTRY_EXIT):
    return next(item for item in graph.data["cases"] if item["case"] == kind.value)


def change_report(graph, change, kind=K.NORMAL_ENTRY_EXIT):
    aid = case(graph, kind)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        report = json.loads(
            db.execute(
                "SELECT report_json FROM testnet_trade_reconciliations WHERE attempt_id=?", (aid,)
            ).fetchone()[0]
        )
        change(report)
        text = exact_json(report)
        db.execute(
            "UPDATE testnet_trade_reconciliations SET report_json=? WHERE attempt_id=?", (text, aid)
        )
    graph.change_proof(kind, lambda proof: proof.update(accounting_sha256=digest(text)))


@pytest.mark.parametrize(
    "fault",
    [
        "net_profit",
        "fees",
        "deleted_trade",
        "duplicate_trade",
        "misgrouped_trade",
        "native_fee",
        "fake_quantity",
        "fake_quote",
        "fake_vwap",
        "funding_total",
        "official_pnl",
        "spot_inventory",
        "return",
        "future_check",
    ],
)
def test_repinning_inconsistent_accounting_does_not_make_it_valid(graph, fault):
    def change(report):
        if fault == "net_profit":
            report["calculation"]["net_profit"] = "999"
        elif fault == "fees":
            report["fees"]["spot_entry"] = "0"
        elif fault == "deleted_trade":
            report["legs"][0]["trades"] = []
        elif fault == "duplicate_trade":
            report["legs"][0]["trades"].append(report["legs"][0]["trades"][0])
        elif fault == "misgrouped_trade":
            report["legs"][0]["trades"], report["legs"][1]["trades"] = (
                report["legs"][1]["trades"],
                report["legs"][0]["trades"],
            )
        elif fault == "native_fee":
            report["legs"][0]["trades"][0]["commission"] = "0"
        elif fault.startswith("fake_"):
            report["legs"][0]["quote_quantity" if fault == "fake_quote" else fault[5:]] = "999"
        elif fault == "future_check":
            report["checked_at"] = (AT.replace(year=2099)).isoformat()
        else:
            field = {
                "return": "net_return",
                "official_pnl": "official_perp_pnl",
                "spot_inventory": "spot_inventory_change",
            }.get(fault, fault)
            report[field] = "999"

    change_report(graph, change)
    before = graph.path.read_bytes()
    with pytest.raises(RiskReviewError):
        graph.read()
    assert graph.path.read_bytes() == before


@pytest.mark.parametrize("leg", ["spot_exit", "perp_exit"])
def test_exit_receipt_must_identify_the_native_fill_group(graph, leg):
    aid = case(graph)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        receipt = json.loads(
            db.execute(
                f"SELECT {leg} FROM testnet_pair_attempts WHERE attempt_id=?", (aid,)
            ).fetchone()[0]
        )
        receipt["order_id"] = "unobserved-order"
        db.execute(
            f"UPDATE testnet_pair_attempts SET {leg}=? WHERE attempt_id=?",
            (exact_json(receipt), aid),
        )
        db.row_factory = sqlite3.Row
        attempt = dict(
            db.execute("SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (aid,)).fetchone()
        )
    graph.change_proof(
        K.NORMAL_ENTRY_EXIT, lambda proof: proof.update(attempt_sha256=digest(exact_json(attempt)))
    )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_normal_case_funding_requires_original_ledger_owner_and_amount(graph):
    with closing(sqlite3.connect(graph.path)) as db, db:
        db.execute(
            "UPDATE testnet_funding_payments SET amount='999' WHERE attempt_id=?",
            (case(graph)["attempt_id"],),
        )
    with pytest.raises(RiskReviewError):
        graph.read()


@pytest.mark.parametrize("fault", ["missing", "wrong_side", "wrong_quantity"])
def test_native_fill_receipt_requires_the_original_durable_intent(graph, fault):
    aid = case(graph)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        row = db.execute(
            "SELECT client_order_id,order_json FROM testnet_order_intents WHERE attempt_id=? ORDER BY client_order_id",
            (aid,),
        ).fetchone()
        if fault == "missing":
            db.execute("DELETE FROM testnet_order_intents WHERE client_order_id=?", (row[0],))
        else:
            order = json.loads(row[1])
            order["side" if fault == "wrong_side" else "quantity"] = (
                "SELL" if fault == "wrong_side" else "9"
            )
            db.execute(
                "UPDATE testnet_order_intents SET order_json=? WHERE client_order_id=?",
                (exact_json(order), row[0]),
            )
    with pytest.raises(RiskReviewError):
        graph.read()


@pytest.mark.parametrize(
    "field",
    [
        "symbol",
        "quantity",
        "entry_basis",
        "exit_basis",
        "price_pnl",
        "funding_total",
        "spot_fees",
        "perp_fees",
        "total_fees",
        "other_costs",
    ],
)
def test_every_cash_flow_component_is_recomputed_from_native_sources(graph, field):
    change_report(
        graph, lambda r: r["calculation"].update({field: "ETHUSDT" if field == "symbol" else "999"})
    )
    with pytest.raises(RiskReviewError):
        graph.read()


@pytest.mark.parametrize("fault", ["late_intent", "violated_limit", "not_reduce_only"])
def test_native_exit_is_bound_to_its_original_intent_and_price_limit(graph, fault):
    aid = case(graph)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        row = db.execute(
            "SELECT client_order_id,order_json FROM testnet_order_intents WHERE attempt_id=? AND client_order_id LIKE '%-3'",
            (aid,),
        ).fetchone()
        if fault == "late_intent":
            db.execute(
                "UPDATE testnet_order_intents SET created_at=? WHERE client_order_id=?",
                (AT.replace(year=2099).isoformat(), row[0]),
            )
        else:
            order = json.loads(row[1])
            order.update(
                {"limit_price": "1"} if fault == "violated_limit" else {"reduce_only": False}
            )
            db.execute(
                "UPDATE testnet_order_intents SET order_json=? WHERE client_order_id=?",
                (exact_json(order), row[0]),
            )
    with pytest.raises(RiskReviewError):
        graph.read()


@pytest.mark.parametrize("fault", ["missing", "unresolved", "late", "wrong_side"])
def test_native_partial_exit_must_include_every_durable_repair(graph, fault):
    aid = case(graph, K.PARTIAL_EXIT_RECOVERY)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        if fault == "missing":
            db.execute("DELETE FROM testnet_spot_exit_recoveries WHERE attempt_id=?", (aid,))
        elif fault == "unresolved":
            db.execute(
                "UPDATE testnet_spot_exit_recoveries SET receipt_json=NULL WHERE attempt_id=?",
                (aid,),
            )
        elif fault == "late":
            db.execute(
                "UPDATE testnet_spot_exit_recoveries SET created_at=? WHERE attempt_id=?",
                (AT.replace(year=2099).isoformat(), aid),
            )
        else:
            order = json.loads(
                db.execute(
                    "SELECT order_json FROM testnet_spot_exit_recoveries WHERE attempt_id=?", (aid,)
                ).fetchone()[0]
            )
            order["side"] = "BUY"
            db.execute(
                "UPDATE testnet_spot_exit_recoveries SET order_json=? WHERE attempt_id=?",
                (exact_json(order), aid),
            )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_unresolved_original_intent_cannot_qualify_as_spot_ack_recovery(graph):
    aid = case(graph, K.SPOT_ACK_RECOVERY)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        plan = json.loads(
            db.execute(
                "SELECT plan_json FROM testnet_pair_attempts WHERE attempt_id=?", (aid,)
            ).fetchone()[0]
        )
        order = plan["perp_entry"]
        db.execute(
            "INSERT INTO testnet_order_intents VALUES(?,?,?,?)",
            (order["client_order_id"], aid, exact_json(order), AT.isoformat()),
        )
    with pytest.raises(RiskReviewError):
        graph.read()


def test_wallet_capture_must_precede_original_witness_completion(graph):
    item = case(graph)
    with closing(sqlite3.connect(graph.path)) as db, db:
        row = db.execute(
            "SELECT audit_id,report_json FROM testnet_balance_audits WHERE attempt_id=?",
            (item["attempt_id"],),
        ).fetchone()
        audit = json.loads(row[1])
        for snapshot in audit["after"]:
            snapshot["requested_at"] = snapshot["received_at"] = AT.replace(year=2099).isoformat()
        text = exact_json(audit)
        db.execute(
            "UPDATE testnet_balance_audits SET report_json=? WHERE audit_id=?", (text, row[0])
        )
    graph.change_proof(K.NORMAL_ENTRY_EXIT, lambda proof: proof.update(audit_sha256=digest(text)))
    with pytest.raises(RiskReviewError):
        graph.read()


@pytest.mark.parametrize("capital,costs", [(None, "0"), ("100", "0.1"), ("100.001", "0")])
def test_original_explicit_capital_and_other_costs_are_preserved(graph, capital, costs):
    def change(report):
        fees = FourLegFees(**{k: Decimal(v) for k, v in report["fees"].items()})
        calculation = report["calculation"]
        profit = net_profit(
            Decimal(calculation["price_pnl"]),
            Decimal(calculation["funding_total"]),
            fees,
            Decimal(costs),
        )
        calculation.update(other_costs=costs, net_profit=str(profit))
        report.update(
            capital=capital,
            net_return=None if capital is None else str(net_return(profit, Decimal(capital))),
        )

    change_report(graph, change)
    before = graph.path.read_bytes()
    assert len(graph.read().cases) == len(K)
    assert graph.path.read_bytes() == before


@pytest.mark.parametrize("matches", [True, False])
def test_optional_raw_copy_in_report_must_match_the_verified_source(graph, matches):
    aid = case(graph)["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db:
        source = json.loads(
            db.execute(
                "SELECT evidence_json FROM testnet_trade_evidence WHERE attempt_id=? ORDER BY trade_id",
                (aid,),
            ).fetchone()[0]
        )

    def change(report):
        target = next(
            t
            for leg in report["legs"]
            for t in leg["trades"]
            if t["trade_id"] == source["trade_id"] and t["market"] == source["market"]
        )
        target["raw_json"] = source["raw_json"] if matches else "{}"

    change_report(graph, change)
    if matches:
        assert len(graph.read().cases) == len(K)
    else:
        with pytest.raises(RiskReviewError):
            graph.read()


def test_row_status_cannot_contradict_its_pinned_accounting_report(graph):
    with closing(sqlite3.connect(graph.path)) as db, db:
        db.execute(
            "UPDATE testnet_trade_reconciliations SET status='INCOMPLETE' WHERE attempt_id=?",
            (case(graph)["attempt_id"],),
        )
    with pytest.raises(RiskReviewError):
        graph.read()
