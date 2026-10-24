"""Synthetic native-shaped graphs test linkage only; not actual execution evidence."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_execution_risk_review import AT, Graph

from crypto_arbitrage_bot.execution import main
from crypto_arbitrage_bot.execution.acceptance_proof import ProofExportFailure as F
from crypto_arbitrage_bot.execution.acceptance_proof import export_acceptance_proof
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase as K
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("offline proof test"))
    )
    graph = Graph(tmp_path)
    settings = graph.settings.model_copy(
        update={"testnet": graph.settings.testnet.model_copy(update={"journal_path": graph.path})}
    )
    case = next(c for c in graph.data["cases"] if c["case"] == K.FEES_FUNDING_BALANCE.value)
    aid = case["attempt_id"]
    # Deliberately manufactured local native-shaped data in this temporary DB.
    # No real exchange source, qualification output or trading is involved.
    registration = {
        "schema_version": 1,
        "environment": "TESTNET",
        "attempt_id": aid,
        "symbol": "BTCUSDT",
        "case": K.FEES_FUNDING_BALANCE,
        "created_at": AT - timedelta(seconds=2),
        "expires_at": AT + timedelta(days=1),
        "origin": "NATIVE_TESTNET",
        "journal_path": str(graph.path),
        "policy_sha256": graph.policy,
        "implementation_sha256": graph.implementation,
    }
    folder = tmp_path / "synthetic-witness"
    folder.mkdir()
    path = folder / "registration.json"
    text = exact_json(registration)
    path.write_text(text, encoding="utf-8")
    ref = {"path": str(path), "sha256": sha(text)}
    directory = folder / "events"
    directory.mkdir()
    with closing(sqlite3.connect(graph.path)) as db:
        db.row_factory = sqlite3.Row
        attempt = db.execute(
            "SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (aid,)
        ).fetchone()
    raw_events = [{"kind": "SESSION_BEGIN", "at": AT - timedelta(seconds=2)}]
    for leg in ("spot_entry", "perp_entry", "spot_exit", "perp_exit"):
        value = json.loads(attempt[leg])
        raw_events.append(
            {
                "kind": "POST_RESULT",
                "at": AT + timedelta(seconds=1 if leg.endswith("entry") else 90),
                "receipt": {
                    key: value[key]
                    for key in (
                        "market",
                        "symbol",
                        "order_id",
                        "client_order_id",
                        "executed_quantity",
                    )
                },
            }
        )
    raw_events.append(
        {"kind": "SESSION_END", "at": AT + timedelta(seconds=130), "capture_complete": True}
    )
    previous = ref["sha256"]
    events = []
    for index, value in enumerate(raw_events):
        event = {
            "sequence": index,
            "previous_sha256": previous,
            "registration_sha256": ref["sha256"],
            **value,
        }
        body = exact_json(event)
        directory.joinpath(f"{index:05d}.json").write_text(body, encoding="utf-8")
        previous = sha(body)
        events.append(json.loads(body))
    trace = {"schema_version": 1, "registration": ref, "origin": "NATIVE_TESTNET", "events": events}
    body = exact_json(trace)
    trace_path = folder / "trace.json"
    trace_path.write_text(body, encoding="utf-8")
    report = {
        "sample_type": "TESTNET_CONNECTIVITY_ACCEPTANCE",
        "status": "VERIFIED",
        "witness_status": "EXPORTED",
        "journal_path": str(graph.path),
        "symbol": "BTCUSDT",
        "attempt_id": aid,
        "witness_registration": ref,
        "witness_trace": {"path": str(trace_path), "sha256": sha(body)},
    }
    report_path = tmp_path / "synthetic-report.json"
    report_path.write_text(exact_json(report), encoding="utf-8")
    return (
        graph,
        settings,
        report_path,
        tmp_path / "synthetic-case-export.json",
        lambda: AT + timedelta(seconds=180),
    )


def test_links_original_native_archive_and_leaves_database_unchanged(source):
    graph, settings, report, out, clock = source
    before = graph.path.read_bytes()
    result = export_acceptance_proof(settings, report, out, clock=clock)
    assert result.success and result.failure is None
    proof = json.loads(out.read_text())
    assert (
        proof["case"] == K.FEES_FUNDING_BALANCE.value
        and proof["attempt_id"] == json.loads(report.read_text())["attempt_id"]
    )
    assert result.reference.sha256 == sha(out.read_text())
    assert graph.path.read_bytes() == before
    assert "qualified_quantities" not in proof and "trade_allowed" not in proof
    original = out.read_bytes()
    assert export_acceptance_proof(settings, report, out, clock=clock).failure is F.OUTPUT_EXISTS
    assert out.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        "offline",
        "wrong_symbol",
        "different_journal",
        "missing_trace",
        "trace_changed",
        "deleted_tail",
        "unclosed_session",
        "wrong_policy",
        "wrong_code",
        "status",
        "capture_status",
        "missing_registration",
    ],
)
def test_incomplete_or_inconsistent_capture_refused(source, change):
    graph, settings, path, out, clock = source
    report = json.loads(path.read_text())
    reg_path = Path(report["witness_registration"]["path"])
    reg = json.loads(reg_path.read_text())
    if change in ("offline", "wrong_policy", "wrong_code"):
        key = {
            "offline": "origin",
            "wrong_policy": "policy_sha256",
            "wrong_code": "implementation_sha256",
        }[change]
        reg[key] = "OFFLINE_FIXTURE" if change == "offline" else "0" * 64
        body = exact_json(reg)
        reg_path.write_text(body, encoding="utf-8")
        report["witness_registration"]["sha256"] = sha(body)
        # The original pinned trace is not rewritten into approval.
    elif change == "wrong_symbol":
        report["symbol"] = "ETHUSDT"
    elif change == "different_journal":
        report["journal_path"] = str(path.parent / "other.sqlite")
    elif change == "status":
        report["status"] = "NEEDS_ATTENTION"
    elif change == "capture_status":
        report["witness_status"] = "FAILED"
    elif change == "missing_registration":
        reg_path.unlink()
    elif change == "missing_trace":
        Path(report["witness_trace"]["path"]).unlink()
    elif change == "trace_changed":
        Path(report["witness_trace"]["path"]).write_text("{}", encoding="utf-8")
    else:
        reg_path.parent.joinpath("events/00005.json").unlink()
    path.write_text(exact_json(report), encoding="utf-8")
    result = export_acceptance_proof(settings, path, out, clock=clock)
    assert not result.success and result.reference is None and not out.exists()


@pytest.mark.parametrize(
    "change",
    [
        "audit_missing",
        "accounting_missing",
        "fill_missing",
        "native_trade_changed",
        "wallet_changed",
        "before_after_wrong",
        "attempt_changed",
        "funding_missing",
    ],
)
def test_missing_or_contradictory_native_sources_not_approved(source, change):
    graph, settings, path, out, clock = source
    aid = json.loads(path.read_text())["attempt_id"]
    with closing(sqlite3.connect(graph.path)) as db, db:
        if change == "audit_missing":
            db.execute("DELETE FROM testnet_balance_audits WHERE attempt_id=?", (aid,))
        elif change == "accounting_missing":
            db.execute("DELETE FROM testnet_trade_reconciliations WHERE attempt_id=?", (aid,))
        elif change == "fill_missing":
            db.execute("DELETE FROM testnet_trade_evidence WHERE attempt_id=?", (aid,))
        elif change == "native_trade_changed":
            db.execute(
                "UPDATE testnet_trade_evidence SET evidence_json='{}' WHERE attempt_id=?", (aid,)
            )
        elif change == "attempt_changed":
            db.execute("UPDATE testnet_pair_attempts SET state='HEDGED' WHERE attempt_id=?", (aid,))
        elif change == "funding_missing":
            db.execute("DELETE FROM testnet_funding_payments WHERE attempt_id=?", (aid,))
        else:
            value = json.loads(
                db.execute(
                    "SELECT report_json FROM testnet_balance_audits WHERE attempt_id=?", (aid,)
                ).fetchone()[0]
            )
            if change == "wallet_changed":
                value["before"][0]["raw_balances_json"] = "{}"
            else:
                value["after"][0]["requested_at"] = AT.isoformat()
            db.execute(
                "UPDATE testnet_balance_audits SET report_json=? WHERE attempt_id=?",
                (exact_json(value), aid),
            )
    result = export_acceptance_proof(settings, path, out, clock=clock)
    assert result.failure is F.SOURCE_INCOMPLETE and result.reference is None and not out.exists()


def test_current_source_change_is_explicit_failure(source, monkeypatch):
    graph, settings, path, out, clock = source
    monkeypatch.setattr(
        "crypto_arbitrage_bot.execution.acceptance_proof.implementation_fingerprint",
        lambda: "0" * 64,
    )
    assert export_acceptance_proof(settings, path, out, clock=clock).failure is F.POLICY_CHANGED


def test_even_internally_consistent_offline_capture_cannot_be_native_proof(source):
    graph, settings, path, out, clock = source
    report = json.loads(path.read_text())
    registration_path = Path(report["witness_registration"]["path"])
    registration = json.loads(registration_path.read_text())
    registration["origin"] = "OFFLINE_FIXTURE"
    body = exact_json(registration)
    registration_path.write_text(body, encoding="utf-8")
    report["witness_registration"]["sha256"] = previous = sha(body)
    events = []
    for event_path in sorted(registration_path.parent.joinpath("events").iterdir()):
        event = json.loads(event_path.read_text())
        event.update(
            previous_sha256=previous, registration_sha256=report["witness_registration"]["sha256"]
        )
        body = exact_json(event)
        event_path.write_text(body, encoding="utf-8")
        previous = sha(body)
        events.append(event)
    body = exact_json(
        {
            "schema_version": 1,
            "registration": report["witness_registration"],
            "origin": "OFFLINE_FIXTURE",
            "events": events,
        }
    )
    Path(report["witness_trace"]["path"]).write_text(body, encoding="utf-8")
    report["witness_trace"]["sha256"] = sha(body)
    path.write_text(exact_json(report), encoding="utf-8")
    assert export_acceptance_proof(settings, path, out, clock=clock).failure is F.NON_NATIVE_CAPTURE
    assert not out.exists()


def test_output_failure_does_not_change_native_sources(source, monkeypatch):
    graph, settings, path, out, clock = source
    before = graph.path.read_bytes()
    original = Path.open

    def open_path(p, mode="r", *args, **kwargs):
        if p == out and mode == "xb":
            raise OSError("SECRET_REQUEST_BODY")
        return original(p, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    result = export_acceptance_proof(settings, path, out, clock=clock)
    assert result.failure is F.IO_ERROR and graph.path.read_bytes() == before and not out.exists()


def test_cli_export_never_creates_client_or_reads_credentials(source, monkeypatch, capsys):
    graph, settings, path, out, clock = source
    monkeypatch.setattr(
        "sys.argv",
        [
            "bot",
            "--export-risk-proof",
            "--acceptance-report",
            str(path),
            "--risk-proof-output",
            str(out),
        ],
    )
    monkeypatch.setattr(main, "load_settings", lambda p: settings)
    monkeypatch.setattr(
        main, "execution_environment", Mock(side_effect=AssertionError("no credential read"))
    )
    monkeypatch.setattr(
        main, "create_testnet_bot", Mock(side_effect=AssertionError("no network client"))
    )
    actual = export_acceptance_proof
    monkeypatch.setattr(
        "crypto_arbitrage_bot.execution.acceptance_proof.export_acceptance_proof",
        lambda s, p, o: actual(s, p, o, clock=clock),
    )
    assert main.main() == 0
    assert '"risk_qualification_issued":false' in capsys.readouterr().out


@pytest.mark.parametrize(
    "args", [["--export-risk-proof"], ["--once", "--risk-proof-output", "bad"]]
)
def test_export_option_requires_explicit_separate_output(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["bot", *args])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
