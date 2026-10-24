"""Synthetic exchange graphs and real offline processes, never native trade proof."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_captured_case_proof import Capture
from test_execution_risk_review import AT
from test_risk_witness import example as example
from test_risk_witness import plan_for

from crypto_arbitrage_bot.execution import main, restart_capture
from crypto_arbitrage_bot.execution.acceptance_proof import ProofExportFailure as F
from crypto_arbitrage_bot.execution.risk_witness import (
    RiskWitness,
    WitnessError,
    read_witness_events,
)
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase as K
from crypto_arbitrage_bot.utils.native_identity import NativeIdentity, NativeState


def identity(pid, when):
    ticks = (when - datetime(1601, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1) * 10
    return NativeIdentity(pid, ticks)


def captured(tmp_path):
    capture = Capture(tmp_path, K.RESTART_IDEMPOTENCY)
    old = identity(101, AT - timedelta(seconds=20))
    new = identity(102, AT + timedelta(seconds=131))
    snapshot = restart_capture.execution_snapshot(capture.graph.path, capture.aid)
    capture.raw[0]["process_identity"] = old.payload()
    capture.raw[-1].update(process_identity=old.payload(), terminal_snapshot=snapshot)
    capture.raw.extend(
        [
            {
                "kind": "SESSION_BEGIN",
                "at": AT + timedelta(seconds=132),
                "process_identity": new.payload(),
                "phase": "TERMINAL_RESTART",
            },
            {
                "kind": "TERMINAL_REPLAY_BEGIN",
                "at": AT + timedelta(seconds=132),
                "previous_process": old.payload(),
                "previous_observation": "MISSING",
                "before": snapshot,
            },
            {
                "kind": "TERMINAL_REPLAY_END",
                "at": AT + timedelta(seconds=133),
                "after": snapshot,
                "request_call_count": 0,
                "result_state": snapshot["state"],
            },
            {
                "kind": "SESSION_END",
                "at": AT + timedelta(seconds=134),
                "process_identity": new.payload(),
                "terminal_snapshot": snapshot,
                "capture_complete": True,
            },
        ]
    )
    capture.publish()
    return capture


def test_two_bound_sessions_and_original_terminal_replay_export(tmp_path):
    capture = captured(tmp_path)
    before = capture.graph.path.read_bytes()
    result, calls = restart_capture.replay_terminal(capture.graph.path, capture.aid)
    assert result.state.value == "CLOSED" and calls == 0
    assert capture.export().success
    value = json.loads(capture.out.read_text(encoding="utf-8"))
    event = next(e for e in value["events"] if e["kind"] == "TERMINAL_RESTART_NO_NEW_INTENTS")
    assert event["before_execution_sha256"] == event["after_execution_sha256"]
    assert event["entry_post_count"] == 0
    assert capture.graph.path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        "same_process",
        "old_running",
        "old_unknown",
        "born_before_terminal",
        "birth_projection",
        "wrong_prior",
        "changed_before",
        "changed_after",
        "changed_checkpoint",
        "changed_ledger",
        "boolean_calls",
        "one_call",
        "wrong_result",
        "empty_restart",
        "query_during_restart",
        "unclosed",
        "missing_native",
        "wrong_phase",
    ],
)
def test_labels_or_inconsistent_restart_source_never_export(tmp_path, change):
    capture = captured(tmp_path)
    raw = capture.raw
    start, before, after, end = raw[-4:]
    if change == "same_process":
        start["process_identity"] = raw[0]["process_identity"]
        end["process_identity"] = raw[0]["process_identity"]
    elif change in ("old_running", "old_unknown"):
        before["previous_observation"] = "RUNNING" if change == "old_running" else "UNKNOWN"
    elif change == "born_before_terminal":
        new = identity(102, AT + timedelta(seconds=129)).payload()
        start["process_identity"] = end["process_identity"] = new
    elif change == "birth_projection":
        start["process_identity"]["created_at"] = AT.isoformat()
    elif change == "wrong_prior":
        before["previous_process"] = start["process_identity"]
    elif change in ("changed_before", "changed_after", "changed_checkpoint"):
        target = (
            before if change == "changed_before" else after if change == "changed_after" else end
        )
        key = (
            "before"
            if change == "changed_before"
            else "after"
            if change == "changed_after"
            else "terminal_snapshot"
        )
        target[key] = {**target[key], "execution_sha256": "0" * 64}
    elif change == "changed_ledger":
        with closing(sqlite3.connect(capture.graph.path)) as db, db:
            db.execute(
                "UPDATE testnet_pair_attempts SET failure_reason='changed other attempt' WHERE attempt_id<>?",
                (capture.aid,),
            )
    elif change in ("boolean_calls", "one_call"):
        after["request_call_count"] = False if change == "boolean_calls" else 1
    elif change == "wrong_result":
        after["result_state"] = "HEDGED"
    elif change == "empty_restart":
        del raw[-3:-1]
    elif change == "query_during_restart":
        raw.insert(
            -1,
            {
                "kind": "QUERY_BEGIN",
                "at": AT + timedelta(seconds=134),
                "symbol": "BTCUSDT",
                "client_order_id": raw[1]["client_order_id"],
            },
        )
    elif change == "unclosed":
        raw.pop()
    elif change == "missing_native":
        raw[0].pop("process_identity")
    else:
        start["phase"] = "NORMAL"
    capture.publish()
    assert not capture.export().success and not capture.out.exists()


@pytest.mark.parametrize("origin", ["OFFLINE_FIXTURE", "NATIVE_TESTNET"])
def test_native_exchange_origin_cannot_be_inferred_from_real_process_metadata(tmp_path, origin):
    capture = captured(tmp_path)
    capture.reg["origin"] = origin
    capture.publish()
    if origin == "OFFLINE_FIXTURE":
        assert capture.export().failure is F.NON_NATIVE_CAPTURE
    else:
        assert capture.export().success  # Explicitly synthetic native-shaped graph.


@pytest.mark.parametrize("limit,value", [("MAX_ROWS", 1), ("MAX_BYTES", 1), ("MAX_SECONDS", 0)])
def test_snapshot_limits_are_enforced_without_writes(tmp_path, monkeypatch, limit, value):
    capture = captured(tmp_path)
    before = capture.graph.path.read_bytes()
    monkeypatch.setattr(restart_capture, limit, value)
    with pytest.raises((ValueError, sqlite3.Error)):
        restart_capture.execution_snapshot(capture.graph.path, capture.aid)
    assert capture.graph.path.read_bytes() == before


def test_missing_or_live_attempt_cannot_be_replayed(tmp_path):
    capture = captured(tmp_path)
    with pytest.raises(ValueError):
        restart_capture.replay_terminal(capture.graph.path, "missing")
    with closing(sqlite3.connect(capture.graph.path)) as db, db:
        db.execute(
            "UPDATE testnet_pair_attempts SET state='HEDGED' WHERE attempt_id=?", (capture.aid,)
        )
    before = capture.graph.path.read_bytes()
    with pytest.raises(ValueError):
        restart_capture.replay_terminal(capture.graph.path, capture.aid)
    assert capture.graph.path.read_bytes() == before


def test_requests_and_journal_writes_are_blocked_in_original_replay(tmp_path, monkeypatch):
    capture = captured(tmp_path)
    before = capture.graph.path.read_bytes()
    original = restart_capture.TestnetPairCoordinator.reconcile

    def modified(coordinator, plan):
        with coordinator.journal._connect() as db:
            with pytest.raises(sqlite3.OperationalError):
                db.execute("DELETE FROM testnet_order_intents")
        with pytest.raises(ValueError):
            coordinator.spot.get_order(plan.symbol, plan.spot_entry.client_order_id)
        return original(coordinator, plan)

    monkeypatch.setattr(restart_capture.TestnetPairCoordinator, "reconcile", modified)
    with pytest.raises(ValueError):
        restart_capture.replay_terminal(capture.graph.path, capture.aid)
    assert capture.graph.path.read_bytes() == before


def test_existing_connection_snapshot_stays_in_original_transaction(tmp_path):
    capture = captured(tmp_path)
    with closing(sqlite3.connect(capture.graph.path)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        expected = restart_capture.execution_snapshot(capture.graph.path, capture.aid)
        assert (
            restart_capture.execution_snapshot(capture.graph.path, capture.aid, connection=db)
            == expected
        )
        assert db.in_transaction


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows identity capture")
def test_same_process_restart_is_refused_without_extending_trace(example, tmp_path):
    bot, transport, old = example
    witness = RiskWitness.register(
        tmp_path / "restart",
        bot.settings,
        bot.spot,
        bot.perp,
        attempt_id="offline-terminal-restart",
        symbol="ETHUSDT",
        case=K.RESTART_IDEMPOTENCY,
        duration=timedelta(minutes=10),
    )
    with witness.observe(bot.spot, bot.perp):
        plan = plan_for(bot, witness)
        bot.coordinator.open_pair(plan)
        bot.coordinator.close_pair(plan)
    events = read_witness_events(witness.registration)
    requests = len(transport.requests)
    with pytest.raises(WitnessError):
        witness.capture_terminal_restart()
    assert read_witness_events(witness.registration) == events
    assert len(transport.requests) == requests


@pytest.mark.parametrize(
    "args",
    [
        ["--capture-terminal-restart"],
        ["--capture-terminal-restart", "--capture-trace-output", "new.json"],
        ["--check", "--capture-trace-output", "new.json"],
    ],
)
def test_restart_cli_requires_original_registration_and_separate_output(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["bot", *args])
    monkeypatch.setattr(main, "load_settings", Mock(side_effect=AssertionError("parse first")))
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2


def test_restart_cli_refuses_missing_capture_without_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bot",
            "--capture-terminal-restart",
            "--capture-registration",
            str(tmp_path / "missing.json"),
            "--capture-registration-sha256",
            "0" * 64,
            "--capture-trace-output",
            str(tmp_path / "new.json"),
        ],
    )
    monkeypatch.setattr(
        main, "execution_environment", Mock(side_effect=AssertionError("no credentials"))
    )
    monkeypatch.setattr(main, "create_testnet_bot", Mock(side_effect=AssertionError("no clients")))
    assert main.main() == 1
    value = json.loads(capsys.readouterr().out)
    assert value["failure"] == "TERMINAL_RESTART_CAPTURE_REJECTED"
    assert value["orders_submitted"] is False and not (tmp_path / "new.json").exists()


def first_capture(tmp_path):
    capture = captured(tmp_path)
    del capture.raw[-4:]
    capture.publish()
    from crypto_arbitrage_bot.execution.risk_witness import load_witness

    return capture, RiskWitness(
        load_witness(capture.reference), capture.settings, clock=lambda: AT + timedelta(seconds=140)
    )


def test_terminal_capture_invokes_original_replay_with_no_exchange_clients(tmp_path, monkeypatch):
    capture, witness = first_capture(tmp_path)
    from crypto_arbitrage_bot.execution import risk_witness

    monkeypatch.setattr(
        risk_witness, "current_native_identity", lambda: identity(102, AT + timedelta(seconds=131))
    )
    monkeypatch.setattr(risk_witness, "observe_native_identity", lambda old: NativeState.MISSING)
    before = capture.graph.path.read_bytes()
    result = witness.capture_terminal_restart()
    assert result["request_call_count"] == 0 and result["state"] == "CLOSED"
    trace = witness.export_trace(tmp_path / "captured-restart-trace.json")
    capture.trace = trace
    assert capture.export().success  # Synthetic native-shaped source only.
    assert capture.graph.path.read_bytes() == before


@pytest.mark.parametrize(
    "condition", ["RUNNING", "UNKNOWN", "same", "old_birth", "snapshot_changed"]
)
def test_owner_or_snapshot_failures_do_not_start_new_capture(tmp_path, monkeypatch, condition):
    capture, witness = first_capture(tmp_path)
    from crypto_arbitrage_bot.execution import risk_witness

    current = identity(102, AT + timedelta(seconds=131))
    if condition == "same":
        current = identity(101, AT - timedelta(seconds=20))
    if condition == "old_birth":
        current = identity(102, AT + timedelta(seconds=129))
    monkeypatch.setattr(risk_witness, "current_native_identity", lambda: current)
    monkeypatch.setattr(
        risk_witness,
        "observe_native_identity",
        lambda old: (
            NativeState(condition) if condition in ("RUNNING", "UNKNOWN") else NativeState.MISSING
        ),
    )
    if condition == "snapshot_changed":
        with closing(sqlite3.connect(capture.graph.path)) as db, db:
            db.execute(
                "UPDATE testnet_pair_attempts SET failure_reason='changed' WHERE attempt_id<>?",
                (capture.aid,),
            )
    prior = read_witness_events(witness.registration)
    with pytest.raises(WitnessError):
        witness.capture_terminal_restart()
    assert read_witness_events(witness.registration) == prior


def test_replay_failure_closes_an_incomplete_capture_and_never_exports(tmp_path, monkeypatch):
    capture, witness = first_capture(tmp_path)
    from crypto_arbitrage_bot.execution import risk_witness

    monkeypatch.setattr(
        risk_witness, "current_native_identity", lambda: identity(102, AT + timedelta(seconds=131))
    )
    monkeypatch.setattr(risk_witness, "observe_native_identity", lambda old: NativeState.MISSING)
    monkeypatch.setattr(
        risk_witness, "replay_terminal", Mock(side_effect=OSError("PRIVATE_ERROR_BODY"))
    )
    with pytest.raises(WitnessError):
        witness.capture_terminal_restart()
    events = read_witness_events(witness.registration)
    assert events[-1]["capture_complete"] is False and witness.broken
    assert "PRIVATE_ERROR_BODY" not in json.dumps(events)
    with pytest.raises(WitnessError):
        witness.export_trace(tmp_path / "never.json")


@pytest.mark.skipif(sys.platform != "win32", reason="actual native Windows restart identity")
def test_two_actual_processes_capture_offline_restart_and_cannot_issue_native_proof(tmp_path):
    from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.execution.captured_case_proof import export_captured_case_proof

    folder = tmp_path / "two-processes"
    helper = Path("work/restart-witness-20261011/offline_process_probe.py").resolve()
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    first = subprocess.run(
        [sys.executable, str(helper), "first", str(folder)],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert first.returncode == 0, first.stderr
    value = json.loads(first.stdout)
    assert value["origin"] == "OFFLINE_FIXTURE" and value["actual_exchange_orders"] == 0
    before = hashlib.sha256((folder / "execution.sqlite").read_bytes()).hexdigest()
    second = subprocess.run(
        [sys.executable, str(helper), "second", str(folder)],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    result = json.loads(second.stdout)
    assert result["success"] and result["result"]["request_call_count"] == 0
    assert result["result"]["origin"] == "OFFLINE_FIXTURE"
    trace = json.loads((folder / "restart-trace.json").read_text(encoding="utf-8"))
    sessions = [e for e in trace["events"] if e["kind"] == "SESSION_BEGIN"]
    assert len(sessions) == 2 and sessions[0]["process_identity"] != sessions[1]["process_identity"]
    assert hashlib.sha256((folder / "execution.sqlite").read_bytes()).hexdigest() == before
    denied = export_captured_case_proof(
        load_settings(folder / "bound-offline.toml"),
        StudyRegistrationSettings(**value["registration"]),
        StudyRegistrationSettings(**result["trace"]),
        folder / "no-native-proof.json",
    )
    assert denied.failure is F.NON_NATIVE_CAPTURE and not (folder / "no-native-proof.json").exists()


@pytest.mark.parametrize(
    "change", ["registration_bytes", "event_bytes", "event_count", "invalid_utf8"]
)
def test_core_witness_readers_bound_untrusted_local_archives(tmp_path, change):
    from crypto_arbitrage_bot.execution.risk_witness import load_witness

    capture = captured(tmp_path)
    if change == "registration_bytes":
        capture.reference.path.write_bytes(b" " * 256001)
        with pytest.raises(WitnessError):
            load_witness(capture.reference)
    else:
        registration = load_witness(capture.reference)
        if change == "event_count":
            for index in range(201):
                registration.events_directory.joinpath(f"{index:05d}.json").write_text(
                    "{}", encoding="utf-8"
                )
        else:
            registration.events_directory.joinpath("00000.json").write_bytes(
                b" " * 16385 if change == "event_bytes" else b"\xff"
            )
        with pytest.raises(WitnessError):
            read_witness_events(registration)


def test_oversized_event_is_refused_before_any_observed_entry(tmp_path):
    capture, witness = first_capture(tmp_path)
    original = read_witness_events(witness.registration)
    with pytest.raises(WitnessError):
        witness._event("OVERSIZED", value="x" * 16385)
    assert read_witness_events(witness.registration) == original
