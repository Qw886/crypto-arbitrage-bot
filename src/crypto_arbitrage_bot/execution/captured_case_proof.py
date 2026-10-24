"""Read-only case export from a preregistered, complete original witness.

No clients, credential reads, journal writes or qualification issuance. Derived
branch events carry original sequence references; labels alone prove nothing.
Restart requires a distinct native process and a captured original terminal replay.
"""

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.acceptance_proof import ProofExportFailure, ProofExportResult
from crypto_arbitrage_bot.execution.restart_capture import derive_restart_event, execution_snapshot
from crypto_arbitrage_bot.execution.risk_review import (
    _case_matches,
    _decimal,
    _native_accounting,
    _time,
    execution_policy_fingerprint,
    implementation_fingerprint,
)
from crypto_arbitrage_bot.execution.risk_witness import (
    MAX_EVENTS,
    load_witness,
    read_witness_events,
)
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_EVENT_BYTES = 16384


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unknown_kind(diagnostic: dict[str, Any]) -> str | None:
    # The real gateway uses SUBMISSION_UNKNOWN, not a TIMEOUT enum. Derive
    # the review vocabulary only from its safe transport/HTTP metadata.
    if diagnostic.get("kind") != "SUBMISSION_UNKNOWN":
        return None
    transport = diagnostic.get("transport_error_type")
    if transport in ("TimeoutError", "timeout"):
        return "TIMEOUT"
    if transport in ("URLError", "OSError", "ConnectionError", "ConnectionResetError",
                     "ConnectionAbortedError", "BrokenPipeError", "SSLError", "SSLEOFError"):
        return "NETWORK_ERROR"
    status = diagnostic.get("http_status")
    if (type(status) is int and status >= 500) or diagnostic.get("api_code") in (-1006, -1007):
        return "HTTP_ERROR"
    return None


def _document(reference: StudyRegistrationSettings) -> dict[str, Any]:
    if reference.path.is_symlink():
        raise ValueError("regular pinned document required")
    with reference.path.open("rb") as stream:
        raw = stream.read(MAX_DOCUMENT_BYTES + 1)
    text = raw.decode("utf-8")
    if len(raw) > MAX_DOCUMENT_BYTES or _sha(text) != reference.sha256:
        raise ValueError("bounded pinned document required")
    value = json.loads(text)
    if not isinstance(value, dict) or exact_json(value) != text:
        raise ValueError("canonical object required")
    return value


def _branch_events(
    events: tuple[dict[str, Any], ...], intents: dict[str, dict[str, Any]],
    *, allow_restart: bool = False,
) -> list[dict[str, Any]]:
    """Check causal gateway observations and derive the original case vocabulary."""
    result: list[dict[str, Any]] = []
    in_session = False
    pending: tuple[str, str] | None = None
    previous_at: datetime | None = None
    unknown: dict[str, dict[str, Any]] = {}
    posted: set[str] = set()
    last_receipt: dict[str, Any] | None = None
    for event in events:
        at = _time(event["at"])
        if previous_at is not None and at < previous_at:
            raise ValueError("observation time regressed")
        previous_at = at
        kind = event["kind"]
        if kind == "SESSION_BEGIN":
            if in_session or pending is not None:
                raise ValueError("nested capture session")
            in_session = True
            last_receipt = None
        elif kind == "SESSION_END":
            if not in_session or pending is not None or event["capture_complete"] is not True:
                raise ValueError("incomplete capture session")
            in_session = False
        else:
            if not in_session:
                raise ValueError("event outside capture session")
            if kind in ("POST_BEGIN", "QUERY_BEGIN"):
                cid = event["client_order_id"]
                order = intents[cid]
                if pending is not None or event["symbol"] != order["symbol"]:
                    raise ValueError("unowned or overlapping operation")
                if kind == "POST_BEGIN" and (
                    event["market"] != order["market"] or event["side"] != order["side"]
                    or _decimal(event["quantity"]) != _decimal(order["quantity"])
                ):
                    raise ValueError("observed submission differs from durable intent")
                if kind == "POST_BEGIN":
                    if cid in posted:
                        raise ValueError("captured POST must not replay a consumed intent")
                    posted.add(cid)
                pending = (kind.split("_")[0], cid)
                last_receipt = None
            elif kind in ("POST_RESULT", "QUERY_RESULT"):
                receipt = event["receipt"]
                cid = receipt["client_order_id"]
                order = intents[cid]
                if pending != (kind.split("_")[0], cid) or any(
                    receipt[key] != order[key] for key in ("market", "symbol")
                ) or _decimal(receipt["original_quantity"]) != _decimal(order["quantity"]):
                    raise ValueError("receipt lacks matching original operation")
                pending = None
                last_receipt = receipt
                if kind == "QUERY_RESULT" and cid in unknown:
                    result.extend([
                        unknown.pop(cid),
                        {"kind": "FIXED_ID_QUERY_RESOLVED", "market": receipt["market"],
                         "order_id": receipt["order_id"], "client_order_id": cid,
                         "source_sequence": event["sequence"]},
                    ])
                if order["side"] == ("SELL" if order["market"] == "SPOT" else "BUY") and (
                    receipt["status"] in ("EXPIRED", "CANCELED", "EXPIRED_IN_MATCH")
                    and 0 < _decimal(receipt["executed_quantity"]) < _decimal(order["quantity"])
                ):
                    result.append({"kind": "PARTIAL_EXIT_TERMINAL", "market": receipt["market"],
                        "source_sequence": event["sequence"], "native_order": {
                            "orderId": receipt["order_id"], "clientOrderId": cid,
                            "origQty": receipt["original_quantity"],
                            "executedQty": receipt["executed_quantity"], "status": receipt["status"]}})
            elif kind in ("POST_FAILED", "QUERY_FAILED"):
                cid = event["client_order_id"]
                if pending != (kind.split("_")[0], cid):
                    raise ValueError("failure lacks matching operation")
                diagnostic = event["diagnostic"]
                order = intents[cid]
                if kind == "POST_FAILED" and event["market"] != order["market"]:
                    raise ValueError("failure market differs")
                error_kind = _unknown_kind(diagnostic)
                if kind == "POST_FAILED" and (
                    error_kind is not None
                    and diagnostic["submission_may_have_reached_exchange"] is True
                    and diagnostic["method"] == "POST" and diagnostic["market"] == order["market"]
                ):
                    unknown[cid] = {"kind": "POST_RESULT_UNKNOWN", "market": order["market"],
                        "client_order_id": cid, "error_kind": error_kind,
                        "source_sequence": event["sequence"]}
                pending = None
                last_receipt = None
            elif kind == "LOCAL_ACK_DELIVERY_INTERRUPT":
                if pending is not None or last_receipt is None or (
                    last_receipt["market"] != "SPOT" or event["market"] != "SPOT"
                    or event["client_order_id"] != last_receipt["client_order_id"]
                    or intents[event["client_order_id"]]["side"] != "BUY"
                ):
                    raise ValueError("ack interruption lacks observed entry receipt")
                result.append(event)
                last_receipt = None
            elif allow_restart and kind in ("TERMINAL_REPLAY_BEGIN", "TERMINAL_REPLAY_END"):
                if pending is not None:
                    raise ValueError("restart replay overlaps a gateway operation")
            else:
                raise ValueError("unsupported or interrupted gateway event")
    if in_session or pending is not None:
        raise ValueError("unclosed capture")
    return result


def export_captured_case_proof(
    settings: Settings, registration_ref: StudyRegistrationSettings,
    trace_ref: StudyRegistrationSettings, output_path: Path,
    *, clock: Callable[[], datetime] = utc_now,
) -> ProofExportResult:
    failure = ProofExportFailure.INVALID_CAPTURE
    try:
        at = utc_timestamp(clock())
        _document(registration_ref)
        registration = load_witness(registration_ref)
        trace = _document(trace_ref)
        if registration.events_directory.is_symlink():
            raise ValueError("regular event directory required")
        files: list[Path] = []
        for path in registration.events_directory.iterdir():
            if len(files) >= MAX_EVENTS or path.is_symlink() or not path.is_file():
                raise ValueError("bounded regular event files required")
            with path.open("rb") as stream:
                if len(stream.read(MAX_EVENT_BYTES + 1)) > MAX_EVENT_BYTES:
                    raise ValueError("event byte limit exceeded")
            files.append(path)
        events = read_witness_events(registration)
        if (
            settings.testnet is None or registration.symbol not in settings.symbols
            or registration.journal_path != settings.testnet.journal_path.resolve()
            or type(trace.get("schema_version")) is not int or trace["schema_version"] != 1
            or trace["registration"] != registration_ref.model_dump(mode="json")
            or trace["origin"] != registration.origin or trace["events"] != list(events)
            or not events or events[0]["kind"] != "SESSION_BEGIN" or events[-1]["kind"] != "SESSION_END"
        ):
            raise ValueError("identity, completeness or restart source unsupported")
        failure = ProofExportFailure.NON_NATIVE_CAPTURE
        if registration.origin != "NATIVE_TESTNET":
            raise ValueError("offline capture is not native evidence")
        failure = ProofExportFailure.POLICY_CHANGED
        if (registration.policy_sha256 != execution_policy_fingerprint(settings)
                or registration.implementation_sha256 != implementation_fingerprint()):
            raise ValueError("current implementation and policy required")
        failure = ProofExportFailure.SOURCE_INCOMPLETE
        with closing(sqlite3.connect(registration.journal_path.as_uri() + "?mode=ro", uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            attempt = db.execute("SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (registration.attempt_id,)).fetchone()
            audits = db.execute("SELECT * FROM testnet_balance_audits WHERE attempt_id=? AND status='MATCHED'", (registration.attempt_id,)).fetchall()
            accounting = db.execute("SELECT report_json FROM testnet_trade_reconciliations WHERE attempt_id=?", (registration.attempt_id,)).fetchall()
            if attempt is None or len(audits) != 1 or len(accounting) != 1:
                raise ValueError("unique complete accounting required")
            plan = json.loads(attempt["plan_json"])
            if (attempt["state"] not in ("CLOSED", "FLATTENED")
                    or _sha(attempt["plan_json"]) != attempt["plan_sha256"]
                    or any(plan[leg]["symbol"] != registration.symbol for leg in ("spot_entry", "perp_entry"))
                    or not registration.created_at <= _time(events[0]["at"]) <= _time(attempt["created_at"])
                    <= _time(attempt["updated_at"]) <= _time(events[-1]["at"]) <= at):
                raise ValueError("journal identity or chronology differs")
            rows = db.execute("SELECT * FROM testnet_order_intents WHERE attempt_id=?", (registration.attempt_id,)).fetchall()
            intents = {row["client_order_id"]: json.loads(row["order_json"]) for row in rows}
            if len(intents) != len(rows):
                raise ValueError("ambiguous durable intents")
            receipts = [json.loads(attempt[leg]) for leg in ("spot_entry", "perp_entry", "spot_exit", "perp_exit") if attempt[leg] is not None]
            intent_times = {row["client_order_id"]: _time(row["created_at"]) for row in rows}
            for table in ("testnet_spot_exit_recoveries", "testnet_exit_recoveries"):
                if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                    continue
                for row in db.execute(f"SELECT order_json,receipt_json,created_at FROM {table} WHERE attempt_id=?", (registration.attempt_id,)):
                    order = json.loads(row["order_json"])
                    cid = order["client_order_id"]
                    if cid in intents or row["receipt_json"] is None:
                        raise ValueError("ambiguous or unresolved repair intent")
                    intents[cid] = order
                    intent_times[cid] = _time(row["created_at"])
                    receipts.append(json.loads(row["receipt_json"]))
            if any(event["kind"] == "POST_BEGIN" and intent_times[event["client_order_id"]] > _time(event["at"]) for event in events):
                raise ValueError("observed submission preceded durable intent")
            restart = registration.case is ExecutionRiskCase.RESTART_IDEMPOTENCY
            derived = _branch_events(events, intents, allow_restart=restart)
            if restart:
                derived.append(derive_restart_event(events, execution_snapshot(registration.journal_path, registration.attempt_id, connection=db)))
            for receipt in receipts:
                if not any(event["kind"] in ("POST_RESULT", "QUERY_RESULT") and all(
                    event["receipt"][key] == receipt[key] for key in ("market", "symbol", "client_order_id", "order_id", "status")
                ) and all(_decimal(event["receipt"][key]) == _decimal(receipt[key]) for key in ("original_quantity", "executed_quantity")) for event in events):
                    raise ValueError("terminal receipt was not observed")
            proof = {"environment": "TESTNET", "evidence_origin": "NATIVE_TESTNET",
                "attempt_id": registration.attempt_id, "case": registration.case.value,
                "policy_sha256": registration.policy_sha256, "implementation_sha256": registration.implementation_sha256,
                "started_at": registration.created_at, "finished_at": at,
                "attempt_sha256": _sha(exact_json(dict(attempt))), "audit_id": audits[0]["audit_id"],
                "audit_sha256": _sha(audits[0]["report_json"]), "accounting_sha256": _sha(accounting[0][0]),
                "events": derived, "witness_registration": registration_ref.model_dump(mode="json"),
                "witness_trace": trace_ref.model_dump(mode="json"), "capture_schema_version": 1}
            native = _native_accounting(db, attempt, proof)
            if not _case_matches(registration.case, attempt, proof, native, db):
                raise ValueError("captured native branch not demonstrated")
            balance = json.loads(audits[0]["report_json"])
            times = [_time(trade["traded_at"]) for trade in balance["trades"]]
            if any(_time(frame["received_at"]) > min(times) for frame in balance["before"]) or any(
                not max(times) <= _time(frame["requested_at"]) <= _time(frame["received_at"]) <= at for frame in balance["after"]
            ):
                raise ValueError("wallet captures must bracket native trades")
        text = exact_json(proof)
        failure = ProofExportFailure.IO_ERROR
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        return ProofExportResult(True, StudyRegistrationSettings(path=output_path.resolve(), sha256=_sha(text)))
    except FileExistsError:
        return ProofExportResult(False, failure=ProofExportFailure.OUTPUT_EXISTS)
    except Exception:
        return ProofExportResult(False, failure=failure)
