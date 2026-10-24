"""Additive, immutable account-balance audit archive; no execution updates."""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from crypto_arbitrage_bot.execution.account_balance_audit import BalanceAudit, BalanceAuditStatus
from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    SQLitePairExecutionJournal,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now


class BalancePersistenceCode(StrEnum):
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    MISSING_PARENT = "MISSING_PARENT"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    MISSING_SOURCE_EVIDENCE = "MISSING_SOURCE_EVIDENCE"


@dataclass(frozen=True)
class BalanceSaveResult:
    success: bool
    audit_id: str | None = None
    error_code: BalancePersistenceCode | None = None


class BalanceAuditRepository:
    """Foreign-key-linked audit of supplied evidence, not a trading decision."""

    def __init__(self, journal: SQLitePairExecutionJournal):
        self.journal = journal

    def initialize(self) -> None:
        with self.journal._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_balance_audits (
                audit_id TEXT PRIMARY KEY NOT NULL,
                attempt_id TEXT NOT NULL REFERENCES testnet_pair_attempts(attempt_id),
                schema_version INTEGER NOT NULL CHECK(schema_version=1),
                status TEXT NOT NULL, created_at TEXT NOT NULL, report_json TEXT NOT NULL)""")

    def save(self, attempt_id: str, audit: BalanceAudit) -> BalanceSaveResult:
        if audit.status is BalanceAuditStatus.MATCHED and (
            audit.issues or not audit.funding_evidence_complete or any(
                m.unexplained_delta is None or abs(m.unexplained_delta) > m.tolerance for m in audit.movements)
        ):
            return BalanceSaveResult(False, error_code=BalancePersistenceCode.EVIDENCE_CONFLICT)
        encoded = exact_json(audit)
        # Snapshot times define a frame. Same frame + differing evidence is a
        # conflict, never an overwrite of the earlier accounting observation.
        frame = [(label, snapshot.market.value, snapshot.requested_at.isoformat(), snapshot.received_at.isoformat())
                 for label, snapshots in (("before", audit.before), ("after", audit.after))
                 for snapshot in snapshots]
        audit_id = hashlib.sha256(exact_json((attempt_id, sorted(frame))).encode()).hexdigest()
        try:
            self.initialize()
            with self.journal._connect() as db:
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("BEGIN IMMEDIATE")
                parent = db.execute("SELECT plan_json FROM testnet_pair_attempts WHERE attempt_id=?",
                                    (attempt_id,)).fetchone()
                if parent is None:
                    return BalanceSaveResult(False, error_code=BalancePersistenceCode.MISSING_PARENT)
                symbol = json.loads(parent[0])["spot_entry"]["symbol"]
                if (any(trade.symbol != symbol for trade in audit.trades)
                        or any(payment.symbol != symbol for payment in audit.funding)):
                    return BalanceSaveResult(False, error_code=BalancePersistenceCode.SYMBOL_MISMATCH)
                if audit.trades:
                    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_trade_evidence'").fetchone():
                        return BalanceSaveResult(False, error_code=BalancePersistenceCode.MISSING_SOURCE_EVIDENCE)
                    for trade in audit.trades:
                        source = db.execute("SELECT attempt_id,evidence_json FROM testnet_trade_evidence "
                                            "WHERE market=? AND symbol=? AND trade_id=?",
                                            (trade.market.value, trade.symbol, trade.trade_id)).fetchone()
                        if source is None:
                            return BalanceSaveResult(False, error_code=BalancePersistenceCode.MISSING_SOURCE_EVIDENCE)
                        if source[0] != attempt_id or source[1] != exact_json(trade):
                            return BalanceSaveResult(False, error_code=BalancePersistenceCode.EVIDENCE_CONFLICT)
                if audit.funding:
                    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_trade_reconciliations'").fetchone():
                        return BalanceSaveResult(False, error_code=BalancePersistenceCode.MISSING_SOURCE_EVIDENCE)
                    source_report = db.execute("SELECT report_json FROM testnet_trade_reconciliations WHERE attempt_id=?",
                                               (attempt_id,)).fetchone()
                    if source_report is None:
                        return BalanceSaveResult(False, error_code=BalancePersistenceCode.MISSING_SOURCE_EVIDENCE)
                    payments = {exact_json(p) for p in json.loads(source_report[0]).get("funding_payments", [])}
                    if any(exact_json(p) not in payments for p in audit.funding):
                        return BalanceSaveResult(False, error_code=BalancePersistenceCode.EVIDENCE_CONFLICT)
                old = db.execute("SELECT report_json FROM testnet_balance_audits WHERE audit_id=?",
                                 (audit_id,)).fetchone()
                if old is not None and old[0] != encoded:
                    return BalanceSaveResult(False, error_code=BalancePersistenceCode.EVIDENCE_CONFLICT)
                db.execute("INSERT OR IGNORE INTO testnet_balance_audits VALUES(?,?,1,?,?,?)",
                           (audit_id, attempt_id, audit.status.value, utc_now().isoformat(), encoded))
            return BalanceSaveResult(True, audit_id)
        except (ExecutionJournalError, OSError):
            return BalanceSaveResult(False, error_code=BalancePersistenceCode.PERSISTENCE_FAILED)

    def get(self, audit_id: str) -> dict[str, Any] | None:
        with self.journal._connect() as db:
            row = db.execute("SELECT report_json FROM testnet_balance_audits WHERE audit_id=?",
                             (audit_id,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def list_for_attempt(self, attempt_id: str) -> tuple[dict[str, Any], ...]:
        with self.journal._connect() as db:
            rows = db.execute("SELECT audit_id,status,created_at FROM testnet_balance_audits "
                              "WHERE attempt_id=? ORDER BY created_at,audit_id", (attempt_id,)).fetchall()
        return tuple(dict(row) for row in rows)
