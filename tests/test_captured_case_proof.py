"""Synthetic native-shaped temporary graphs, never actual exchange evidence."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_execution_risk_review import AT, Graph

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.execution import main
from crypto_arbitrage_bot.execution.acceptance_proof import ProofExportFailure as F
from crypto_arbitrage_bot.execution.captured_case_proof import (
    _unknown_kind,
    export_captured_case_proof,
)
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase as K
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Capture:
    def __init__(self, tmp_path, case):
        graph = Graph(tmp_path)
        self.graph = graph
        self.settings = graph.settings.model_copy(update={"testnet": graph.settings.testnet.model_copy(update={"journal_path": graph.path})})
        self.aid = next(item["attempt_id"] for item in graph.data["cases"] if item["case"] == case.value)
        self.folder = tmp_path / "synthetic-capture"
        self.folder.mkdir()
        self.reg = {"schema_version": 1, "environment": "TESTNET", "attempt_id": self.aid,
            "symbol": "BTCUSDT", "case": case.value, "created_at": AT - timedelta(seconds=2),
            "expires_at": AT + timedelta(days=1), "origin": "NATIVE_TESTNET",
            "journal_path": str(graph.path), "policy_sha256": graph.policy,
            "implementation_sha256": graph.implementation}
        self.raw = [{"kind": "SESSION_BEGIN", "at": AT - timedelta(seconds=2)}]
        with closing(sqlite3.connect(graph.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (self.aid,)).fetchone()
            plan = json.loads(row["plan_json"])
            receipts = [(plan[leg], json.loads(row[leg]), leg) for leg in ("spot_entry", "perp_entry", "spot_exit", "perp_exit") if row[leg] is not None]
            repairs = db.execute("SELECT order_json,receipt_json FROM testnet_spot_exit_recoveries WHERE attempt_id=?", (self.aid,)).fetchall()
            receipts.extend((json.loads(r[0]), json.loads(r[1]), "spot_exit") for r in repairs)
        for order, receipt, leg in receipts:
            when = AT + timedelta(seconds=1 if leg.endswith("entry") else 90)
            cid = order["client_order_id"]
            self.raw.append({"kind": "POST_BEGIN", "at": when, **{key: order[key] for key in ("market", "symbol", "side", "quantity", "client_order_id")}})
            unknown = case is K.PERP_UNKNOWN_RECOVERY and leg == "perp_entry"
            if unknown:
                self.raw.extend([
                    {"kind": "POST_FAILED", "at": when, "market": "USDTM", "client_order_id": cid,
                     "diagnostic": {"kind": "SUBMISSION_UNKNOWN", "transport_error_type": "TimeoutError", "http_status": None, "api_code": None, "market": "USDTM", "method": "POST", "submission_may_have_reached_exchange": True}},
                    {"kind": "QUERY_BEGIN", "at": when, "symbol": "BTCUSDT", "client_order_id": cid},
                ])
            self.raw.append({"kind": "QUERY_RESULT" if unknown else "POST_RESULT", "at": when,
                "receipt": {key: receipt[key] for key in ("market", "symbol", "status", "order_id", "client_order_id", "original_quantity", "executed_quantity")}})
            if case is K.SPOT_ACK_RECOVERY and leg == "spot_entry":
                self.raw.append({"kind": "LOCAL_ACK_DELIVERY_INTERRUPT", "at": when, "market": "SPOT", "client_order_id": cid})
        self.raw.append({"kind": "SESSION_END", "at": AT + timedelta(seconds=130), "capture_complete": True})
        self.out = tmp_path / "synthetic-proof.json"
        self.publish()

    def publish(self):
        text = exact_json(self.reg)
        path = self.folder / "registration.json"
        path.write_text(text, encoding="utf-8")
        self.reference = StudyRegistrationSettings(path=path, sha256=sha(text))
        directory = self.folder / "events"
        directory.mkdir(exist_ok=True)
        for file in directory.iterdir():
            file.unlink()
        previous = self.reference.sha256
        events = []
        for index, value in enumerate(self.raw):
            event = {"sequence": index, "previous_sha256": previous, "registration_sha256": self.reference.sha256, **value}
            body = exact_json(event)
            directory.joinpath(f"{index:05d}.json").write_text(body, encoding="utf-8")
            events.append(json.loads(body))
            previous = sha(body)
        body = exact_json({"schema_version": 1, "registration": self.reference.model_dump(mode="json"), "origin": self.reg["origin"], "events": events})
        path = self.folder / "trace.json"
        path.write_text(body, encoding="utf-8")
        self.trace = StudyRegistrationSettings(path=path, sha256=sha(body))

    def export(self):
        return export_captured_case_proof(self.settings, self.reference, self.trace, self.out, clock=lambda: AT + timedelta(seconds=180))


@pytest.mark.parametrize("case", [kind for kind in K if kind is not K.RESTART_IDEMPOTENCY])
def test_observed_branch_and_all_native_accounting_export_without_writes(tmp_path, case):
    capture = Capture(tmp_path, case)
    before = capture.graph.path.read_bytes()
    result = capture.export()
    assert result.success, result.failure
    assert capture.graph.path.read_bytes() == before
    proof = json.loads(capture.out.read_text(encoding="utf-8"))
    assert proof["case"] == case.value
    assert proof["witness_trace"] == capture.trace.model_dump(mode="json")
    assert "trade_allowed" not in proof and "qualified_quantities" not in proof
    assert result.reference.sha256 == sha(capture.out.read_text(encoding="utf-8"))
    assert capture.export().failure is F.OUTPUT_EXISTS


@pytest.mark.parametrize("change", ["nested", "unclosed", "outside", "regressed", "unfinished_post", "unobserved_post", "unowned", "wrong_quantity", "wrong_market", "wrong_symbol", "receipt_status", "receipt_size", "receipt_order", "interrupted", "manual_branch", "false_tail", "lost_repair"])
def test_causal_and_receipt_changes_are_refused_even_if_repinned(tmp_path, change):
    capture = Capture(tmp_path, K.PARTIAL_EXIT_RECOVERY)
    raw = capture.raw
    if change == "nested":
        raw.insert(1, {"kind": "SESSION_BEGIN", "at": AT})
    elif change == "unclosed":
        raw.pop()
    elif change == "outside":
        raw[0]["kind"] = "SESSION_END"
        raw[0]["capture_complete"] = True
    elif change == "regressed":
        raw[2]["at"] = AT - timedelta(seconds=1)
    elif change == "unfinished_post":
        raw.pop(-2)
    elif change == "unobserved_post":
        raw.pop(1)
    elif change in ("unowned", "wrong_quantity", "wrong_market", "wrong_symbol"):
        key, value = {"unowned": ("client_order_id", "other"), "wrong_quantity": ("quantity", "1"), "wrong_market": ("market", "USDTM"), "wrong_symbol": ("symbol", "ETHUSDT")}[change]
        raw[1][key] = value
    elif change.startswith("receipt_"):
        key, value = {"receipt_status": ("status", "EXPIRED"), "receipt_size": ("executed_quantity", "0"), "receipt_order": ("order_id", "999")}[change]
        raw[2]["receipt"][key] = value
    elif change == "interrupted":
        raw[2]["kind"] = "POST_INTERRUPTED"
    elif change == "manual_branch":
        raw.insert(-1, {"kind": "TERMINAL_RESTART_NO_NEW_INTENTS", "at": AT + timedelta(seconds=100)})
    elif change == "false_tail":
        raw[-1]["capture_complete"] = False
    elif change == "lost_repair":
        del raw[-3:-1]
    capture.publish()
    before = capture.graph.path.read_bytes()
    assert not capture.export().success
    assert not capture.out.exists()
    assert capture.graph.path.read_bytes() == before


@pytest.mark.parametrize("change", ["wrong_cid", "before_send", "no_query", "wrong_method", "wrong_kind", "unconfirmed", "manual_unknown"])
def test_unknown_result_requires_actual_possible_post_and_later_fixed_id_query(tmp_path, change):
    capture = Capture(tmp_path, K.PERP_UNKNOWN_RECOVERY)
    failure = next(event for event in capture.raw if event["kind"] == "POST_FAILED")
    query = next(event for event in capture.raw if event["kind"] == "QUERY_BEGIN")
    if change == "wrong_cid":
        query["client_order_id"] = "other"
    elif change == "before_send":
        failure["diagnostic"]["submission_may_have_reached_exchange"] = False
    elif change == "no_query":
        capture.raw.remove(query)
    elif change == "wrong_method":
        failure["diagnostic"]["method"] = "GET"
    elif change == "wrong_kind":
        failure["diagnostic"]["kind"] = "ORDER_GUARD_REJECTED"
    elif change == "unconfirmed":
        failure["diagnostic"]["submission_may_have_reached_exchange"] = None
    else:
        failure["kind"] = "POST_RESULT_UNKNOWN"
    capture.publish()
    assert not capture.export().success and not capture.out.exists()


@pytest.mark.parametrize("change", ["offline", "policy", "code", "late_registration", "different_journal", "restart", "registration_digest", "trace_digest", "event_limit", "byte_limit", "duplicate_audit", "wrong_accounting", "late_intent"])
def test_source_constraints_and_bounds_preserve_failure(tmp_path, change):
    capture = Capture(tmp_path, K.NORMAL_ENTRY_EXIT)
    if change in ("offline", "policy", "code", "late_registration", "different_journal", "restart"):
        key, value = {"offline": ("origin", "OFFLINE_FIXTURE"), "policy": ("policy_sha256", "0" * 64), "code": ("implementation_sha256", "0" * 64), "late_registration": ("created_at", AT + timedelta(seconds=1)), "different_journal": ("journal_path", str(tmp_path / "other.sqlite")), "restart": ("case", K.RESTART_IDEMPOTENCY.value)}[change]
        capture.reg[key] = value
        capture.publish()
    elif change == "registration_digest":
        capture.reference = capture.reference.model_copy(update={"sha256": "0" * 64})
    elif change == "trace_digest":
        capture.trace = capture.trace.model_copy(update={"sha256": "0" * 64})
    elif change == "event_limit":
        for index in range(201):
            capture.folder.joinpath(f"events/{index:05d}.json").write_text("{}", encoding="utf-8")
    elif change == "byte_limit":
        capture.folder.joinpath("events/00000.json").write_bytes(b" " * 16385)
    else:
        with closing(sqlite3.connect(capture.graph.path)) as db, db:
            if change == "duplicate_audit":
                db.execute("INSERT INTO testnet_balance_audits SELECT * FROM testnet_balance_audits WHERE attempt_id=?", (capture.aid,))
            elif change == "wrong_accounting":
                db.execute("UPDATE testnet_trade_reconciliations SET report_json='{}' WHERE attempt_id=?", (capture.aid,))
            else:
                db.execute("UPDATE testnet_order_intents SET created_at=? WHERE attempt_id=?", ((AT + timedelta(seconds=120)).isoformat(), capture.aid))
    before = capture.graph.path.read_bytes()
    assert not capture.export().success and not capture.out.exists()
    assert capture.graph.path.read_bytes() == before


def test_cli_reads_pinned_case_without_constructing_clients_or_credentials(tmp_path, monkeypatch, capsys):
    capture = Capture(tmp_path, K.NORMAL_ENTRY_EXIT)
    monkeypatch.setattr(main, "load_settings", lambda path: capture.settings)
    monkeypatch.setattr(main, "execution_environment", Mock(side_effect=AssertionError("no credentials")))
    monkeypatch.setattr(main, "create_testnet_bot", Mock(side_effect=AssertionError("no clients")))
    monkeypatch.setattr("sys.argv", ["bot", "--export-captured-case", "--capture-registration", str(capture.reference.path), "--capture-registration-sha256", capture.reference.sha256, "--capture-trace", str(capture.trace.path), "--capture-trace-sha256", capture.trace.sha256, "--risk-proof-output", str(capture.out)])
    assert main.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["credentials_read"] is False and result["orders_submitted"] is False
    assert result["risk_qualification_issued"] is False


@pytest.mark.parametrize("args", [["--export-captured-case"], ["--check", "--capture-trace", "some.json"], ["--export-captured-case", "--risk-proof-output", "new.json"]])
def test_cli_missing_or_unrelated_pins_fail_before_clients(monkeypatch, args):
    monkeypatch.setattr(main, "load_settings", Mock(side_effect=AssertionError("no settings before parse")))
    monkeypatch.setattr("sys.argv", ["bot", *args])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2


@pytest.mark.parametrize("transport,http_status,api_code,expected", [
    ("TimeoutError", None, None, "TIMEOUT"),
    ("ConnectionResetError", None, None, "NETWORK_ERROR"),
    ("SSLEOFError", None, None, "NETWORK_ERROR"),
    (None, 503, None, "HTTP_ERROR"),
    (None, 400, -1007, "HTTP_ERROR"),
    (None, 400, -1006, "HTTP_ERROR"),
    (None, 400, -2010, None), (None, None, None, None),
    ("SECRET_CUSTOM_ERROR", None, None, None),
])
def test_branch_vocabulary_uses_actual_safe_gateway_diagnostics(transport, http_status, api_code, expected):
    error = GatewayError(ExecutionErrorKind.SUBMISSION_UNKNOWN, "not exported",
        market="USDTM", http_status=http_status, api_code=api_code,
        submission_may_have_reached_exchange=True)
    error.transport_error_type = transport
    assert _unknown_kind(error.diagnostic_fields()) == expected


def test_repeated_post_cannot_be_approved_as_fixed_id_recovery(tmp_path):
    capture = Capture(tmp_path, K.NORMAL_ENTRY_EXIT)
    capture.raw[3:3] = [dict(capture.raw[1]), dict(capture.raw[2])]
    capture.publish()
    assert not capture.export().success and not capture.out.exists()


def test_unbracketed_wallet_is_not_exported(tmp_path):
    capture = Capture(tmp_path, K.NORMAL_ENTRY_EXIT)
    with closing(sqlite3.connect(capture.graph.path)) as db, db:
        row = db.execute("SELECT report_json FROM testnet_balance_audits WHERE attempt_id=?", (capture.aid,)).fetchone()
        value = json.loads(row[0])
        value["before"][0]["received_at"] = (AT + timedelta(seconds=2)).isoformat()
        db.execute("UPDATE testnet_balance_audits SET report_json=? WHERE attempt_id=?", (exact_json(value), capture.aid))
    assert not capture.export().success and not capture.out.exists()


def test_io_error_is_sanitized_and_sources_remain_unchanged(tmp_path, monkeypatch):
    capture = Capture(tmp_path, K.NORMAL_ENTRY_EXIT)
    before = capture.graph.path.read_bytes()
    original = Path.open

    def fail(path, mode="r", *args, **kwargs):
        if path == capture.out and mode == "xb":
            raise OSError("SECRET_REQUEST")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    result = capture.export()
    assert result.failure is F.IO_ERROR and not capture.out.exists()
    assert capture.graph.path.read_bytes() == before
