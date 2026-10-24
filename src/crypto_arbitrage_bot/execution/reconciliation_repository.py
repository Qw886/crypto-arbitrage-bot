"""Separate, additive test-account accounting archive; never changes orders."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError,
    SQLitePairExecutionJournal,
)
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationStatus,
    TradeReconciliation,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


class AccountingPersistenceCode(StrEnum):
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    MISSING_PARENT = "MISSING_PARENT"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"


@dataclass(frozen=True)
class AccountingSaveResult:
    success: bool
    error_code: AccountingPersistenceCode | None = None


class AccountingReportOrigin(StrEnum):
    RECONCILIATION = "RECONCILIATION"
    LEGACY_IMPORT = "LEGACY_IMPORT"
    LEGACY_UNMIGRATED = "LEGACY_UNMIGRATED"


class ReconciliationRepository:
    """Version-1 archive uses Decimal TEXT and UTC ISO strings, including raw fills."""

    def __init__(self, journal: SQLitePairExecutionJournal) -> None:
        self.journal = journal

    def initialize(self) -> None:
        with self.journal._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_trade_evidence (
                market TEXT NOT NULL, symbol TEXT NOT NULL, trade_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL REFERENCES testnet_pair_attempts(attempt_id),
                leg TEXT NOT NULL, evidence_json TEXT NOT NULL,
                PRIMARY KEY(market,symbol,trade_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_trade_reconciliations (
                attempt_id TEXT PRIMARY KEY REFERENCES testnet_pair_attempts(attempt_id),
                schema_version INTEGER NOT NULL CHECK(schema_version=1),
                symbol TEXT NOT NULL, status TEXT NOT NULL, checked_at TEXT NOT NULL,
                report_json TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_trade_reconciliation_history (
                attempt_id TEXT NOT NULL REFERENCES testnet_pair_attempts(attempt_id),
                report_sha256 TEXT NOT NULL,
                schema_version INTEGER NOT NULL CHECK(schema_version=1),
                symbol TEXT NOT NULL, status TEXT NOT NULL, checked_at TEXT NOT NULL,
                archived_at TEXT NOT NULL, origin TEXT NOT NULL,
                report_json TEXT NOT NULL, PRIMARY KEY(attempt_id,report_sha256))""")
            # Import the exact old body, not a newly computed or backdated result.
            # The old schema/current row is retained for existing readers.
            for row in db.execute("SELECT attempt_id,symbol,status,checked_at,report_json "
                                  "FROM testnet_trade_reconciliations").fetchall():
                db.execute("INSERT OR IGNORE INTO testnet_trade_reconciliation_history VALUES(?,?,1,?,?,?,?,?,?)",
                    (row["attempt_id"], hashlib.sha256(row["report_json"].encode("utf-8")).hexdigest(),
                     row["symbol"], row["status"], row["checked_at"], datetime.now(UTC).isoformat(),
                     AccountingReportOrigin.LEGACY_IMPORT.value, row["report_json"]))

    def save(self, result: TradeReconciliation) -> AccountingSaveResult:
        """Atomic evidence + report; conflicts roll back, source journal untouched."""
        try:
            self.initialize()
            with self.journal._connect() as db:
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("BEGIN IMMEDIATE")
                parent = db.execute("SELECT plan_json,state FROM testnet_pair_attempts WHERE attempt_id=?",
                                    (result.attempt_id,)).fetchone()
                if parent is None:
                    return AccountingSaveResult(False, AccountingPersistenceCode.MISSING_PARENT)
                if json.loads(parent["plan_json"])["spot_entry"]["symbol"] != result.symbol:
                    return AccountingSaveResult(False, AccountingPersistenceCode.SYMBOL_MISMATCH)
                if result.status is ReconciliationStatus.RECONCILED and (
                    parent["state"] != "CLOSED" or result.calculation is None or result.issues
                ):
                    return AccountingSaveResult(False, AccountingPersistenceCode.EVIDENCE_CONFLICT)
                staged = []
                staged_keys: set[tuple[str, str, str]] = set()
                for leg in result.legs:
                    for trade in leg.trades:
                        if trade.symbol != result.symbol:
                            return AccountingSaveResult(False, AccountingPersistenceCode.SYMBOL_MISMATCH)
                        encoded = exact_json(trade)
                        key = (trade.market.value, trade.symbol, trade.trade_id)
                        if key in staged_keys:
                            return AccountingSaveResult(False, AccountingPersistenceCode.EVIDENCE_CONFLICT)
                        staged_keys.add(key)
                        old = db.execute("SELECT attempt_id,leg,evidence_json FROM testnet_trade_evidence "
                                         "WHERE market=? AND symbol=? AND trade_id=?",
                                         (trade.market.value, trade.symbol, trade.trade_id)).fetchone()
                        if old is not None and tuple(old) != (result.attempt_id, leg.leg.value, encoded):
                            return AccountingSaveResult(False, AccountingPersistenceCode.EVIDENCE_CONFLICT)
                        staged.append((trade.market.value, trade.symbol, trade.trade_id,
                                       result.attempt_id, leg.leg.value, encoded))
                # Check ALL conflicts before inserting any rows.
                db.executemany("INSERT OR IGNORE INTO testnet_trade_evidence VALUES(?,?,?,?,?,?)", staged)
                data = asdict(result)
                # Raw fill bytes are kept once in evidence_json; report retains
                # numeric fill details and IDs for standalone inspection.
                for leg_data in data["legs"]:
                    for trade_data in leg_data["trades"]:
                        trade_data.pop("raw_json", None)
                encoded_report = exact_json(data)
                db.execute("INSERT OR IGNORE INTO testnet_trade_reconciliation_history VALUES(?,?,1,?,?,?,?,?,?)",
                    (result.attempt_id, hashlib.sha256(encoded_report.encode("utf-8")).hexdigest(), result.symbol,
                     result.status.value, result.checked_at.isoformat(), datetime.now(UTC).isoformat(),
                     AccountingReportOrigin.RECONCILIATION.value, encoded_report))
                db.execute("INSERT INTO testnet_trade_reconciliations VALUES(?,1,?,?,?,?) "
                    "ON CONFLICT(attempt_id) DO UPDATE SET status=excluded.status,"
                    "checked_at=excluded.checked_at,report_json=excluded.report_json",
                    (result.attempt_id, result.symbol, result.status.value, result.checked_at.isoformat(), encoded_report))
            return AccountingSaveResult(True)
        except (ExecutionJournalError, OSError):
            return AccountingSaveResult(False, AccountingPersistenceCode.PERSISTENCE_FAILED)

    def get_report(self, attempt_id: str) -> dict[str, Any] | None:
        """Return stored JSON; financial values are exact decimal strings, never floats."""
        with self.journal._connect() as db:
            row = db.execute("SELECT report_json FROM testnet_trade_reconciliations WHERE attempt_id=?",
                             (attempt_id,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def list_report_history(self, attempt_id: str) -> tuple[dict[str, Any], ...]:
        """Latest observation may fail; earlier results remain separately inspectable.

        Returning history never selects an old success as current qualification.
        Decimal values and NULL remain in their original report JSON form.
        """
        with self.journal._connect() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "testnet_trade_reconciliation_history" not in tables:
                if "testnet_trade_reconciliations" not in tables:
                    return ()
                row = db.execute("SELECT * FROM testnet_trade_reconciliations WHERE attempt_id=?",
                                 (attempt_id,)).fetchone()
                if row is None:
                    return ()
                return ({"attempt_id": row["attempt_id"], "schema_version": row["schema_version"],
                    "symbol": row["symbol"], "status": row["status"], "checked_at": row["checked_at"],
                    "archived_at": None, "origin": AccountingReportOrigin.LEGACY_UNMIGRATED.value,
                    "report_sha256": hashlib.sha256(row["report_json"].encode("utf-8")).hexdigest(),
                    "report": dict(json.loads(row["report_json"]))},)
            rows = db.execute("SELECT * FROM testnet_trade_reconciliation_history "
                "WHERE attempt_id=? ORDER BY checked_at,archived_at,report_sha256", (attempt_id,)).fetchall()
        return tuple({**{key: row[key] for key in row.keys() if key != "report_json"},
                      "report": dict(json.loads(row["report_json"]))} for row in rows)
