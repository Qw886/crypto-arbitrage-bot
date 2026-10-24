"""Read-only export linking a captured run to native archived accounting.

One case proof is not an engineering qualification, strategy PASS or order permit.
No network, new audit, journal write or independently implemented PnL formula.
"""

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.risk_review import (
    _case_matches,
    _decimal,
    _native_accounting,
    _pinned,
    _time,
    execution_policy_fingerprint,
    implementation_fingerprint,
)
from crypto_arbitrage_bot.execution.risk_witness import load_witness, read_witness_events
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class ProofExportFailure(StrEnum):
    INVALID_CAPTURE = "INVALID_ACCEPTANCE_CAPTURE"
    NON_NATIVE_CAPTURE = "NON_NATIVE_CAPTURE"
    SOURCE_INCOMPLETE = "NATIVE_ACCOUNTING_INCOMPLETE"
    POLICY_CHANGED = "CAPTURE_POLICY_CHANGED"
    OUTPUT_EXISTS = "PROOF_OUTPUT_EXISTS"
    IO_ERROR = "PROOF_IO_ERROR"


@dataclass(frozen=True)
class ProofExportResult:
    success: bool
    reference: StudyRegistrationSettings | None = None
    failure: ProofExportFailure | None = None


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def export_acceptance_proof(
    settings: Settings,
    report_path: Path,
    output_path: Path,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> ProofExportResult:
    failure = ProofExportFailure.INVALID_CAPTURE
    try:
        at = utc_timestamp(clock())
        with report_path.open("rb") as stream:
            body = stream.read(256001)
        if len(body) > 256000:
            raise ValueError("bounded report required")
        report = json.loads(body)
        if (
            settings.testnet is None
            or report["sample_type"] != "TESTNET_CONNECTIVITY_ACCEPTANCE"
            or report["status"] != "VERIFIED"
            or report["witness_status"] != "EXPORTED"
            or Path(report["journal_path"]) != settings.testnet.journal_path.resolve()
        ):
            raise ValueError("complete captured acceptance required")
        reference = StudyRegistrationSettings(**report["witness_registration"])
        registration = load_witness(reference)
        trace_ref = StudyRegistrationSettings(**report["witness_trace"])
        trace = _pinned(trace_ref.path, trace_ref.sha256)
        events = read_witness_events(registration)
        if (
            type(trace.get("schema_version")) is not int
            or trace["schema_version"] != 1
            or registration.attempt_id != report["attempt_id"]
            or registration.symbol != report["symbol"]
            or registration.symbol not in settings.symbols
            or registration.journal_path != settings.testnet.journal_path.resolve()
            or registration.case is not ExecutionRiskCase.FEES_FUNDING_BALANCE
            or trace["registration"] != {"path": str(reference.path), "sha256": reference.sha256}
            or trace["origin"] != registration.origin
            or trace["events"] != list(events)
            or not events
            or events[0]["kind"] != "SESSION_BEGIN"
            or events[-1]["kind"] != "SESSION_END"
            or events[-1]["capture_complete"] is not True
            or any(e.get("capture_complete") is False for e in events)
            or sum(e["kind"] == "SESSION_BEGIN" for e in events)
            != sum(e["kind"] == "SESSION_END" for e in events)
        ):
            raise ValueError("capture identity/completeness differs")
        failure = ProofExportFailure.NON_NATIVE_CAPTURE
        if registration.origin != "NATIVE_TESTNET":
            raise ValueError("offline fixture cannot become native proof")
        failure = ProofExportFailure.POLICY_CHANGED
        if (
            registration.policy_sha256 != execution_policy_fingerprint(settings)
            or registration.implementation_sha256 != implementation_fingerprint()
        ):
            raise ValueError("current source/policy differs")
        failure = ProofExportFailure.SOURCE_INCOMPLETE
        with closing(
            sqlite3.connect(registration.journal_path.as_uri() + "?mode=ro", uri=True, timeout=0)
        ) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            attempt = db.execute(
                "SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (registration.attempt_id,)
            ).fetchone()
            audits = db.execute(
                "SELECT * FROM testnet_balance_audits WHERE attempt_id=? AND status='MATCHED' ORDER BY audit_id",
                (registration.attempt_id,),
            ).fetchall()
            accounting = db.execute(
                "SELECT report_json FROM testnet_trade_reconciliations WHERE attempt_id=?",
                (registration.attempt_id,),
            ).fetchone()
            if attempt is None or accounting is None or len(audits) != 1:
                raise ValueError("one unambiguous native audit and accounting required")
            plan = json.loads(attempt["plan_json"])
            if (
                attempt["state"] != "CLOSED"
                or _sha(attempt["plan_json"]) != attempt["plan_sha256"]
                or plan["spot_entry"]["symbol"] != registration.symbol
                or plan["perp_entry"]["symbol"] != registration.symbol
                or not registration.created_at
                <= _time(events[0]["at"])
                <= _time(attempt["created_at"])
                <= _time(attempt["updated_at"])
                <= _time(events[-1]["at"])
                <= at
            ):
                raise ValueError("original journal identity/timing differs")
            for leg in ("spot_entry", "perp_entry", "spot_exit", "perp_exit"):
                receipt = json.loads(attempt[leg])
                if not any(
                    e["kind"] in ("POST_RESULT", "QUERY_RESULT")
                    and all(
                        e["receipt"][name] == receipt[name]
                        for name in ("market", "symbol", "order_id", "client_order_id")
                    )
                    and _decimal(e["receipt"]["executed_quantity"])
                    == _decimal(receipt["executed_quantity"])
                    for e in events
                ):
                    raise ValueError("an original fill receipt was not observed")
            audit = audits[0]
            proof = {
                "environment": "TESTNET",
                "evidence_origin": "NATIVE_TESTNET",
                "attempt_id": registration.attempt_id,
                "case": registration.case.value,
                "policy_sha256": registration.policy_sha256,
                "implementation_sha256": registration.implementation_sha256,
                "started_at": registration.created_at,
                "finished_at": at,
                "attempt_sha256": _sha(exact_json(dict(attempt))),
                "audit_id": audit["audit_id"],
                "audit_sha256": _sha(audit["report_json"]),
                "accounting_sha256": _sha(accounting[0]),
                "events": list(events),
                "witness_registration": report["witness_registration"],
                "witness_trace": report["witness_trace"],
            }
            native = _native_accounting(db, attempt, proof)
            if not _case_matches(registration.case, attempt, proof, native, db):
                raise ValueError("native case not demonstrated")
            balance = json.loads(audit["report_json"])
            trade_times = [_time(t["traded_at"]) for t in balance["trades"]]
            if any(_time(s["received_at"]) > min(trade_times) for s in balance["before"]) or any(
                not max(trade_times) <= _time(s["requested_at"]) <= _time(s["received_at"]) <= at
                for s in balance["after"]
            ):
                raise ValueError("wallet captures do not bracket native fills")
        text = exact_json(proof)
        failure = ProofExportFailure.IO_ERROR
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as target:
            target.write(text.encode("utf-8"))
            target.flush()
            os.fsync(target.fileno())
        return ProofExportResult(
            True, StudyRegistrationSettings(path=output_path.resolve(), sha256=_sha(text))
        )
    except FileExistsError:
        return ProofExportResult(False, failure=ProofExportFailure.OUTPUT_EXISTS)
    except Exception:
        return ProofExportResult(False, failure=failure)
