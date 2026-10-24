"""Durable native currency evidence for aborted TESTNET entries, never profit."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairExecutionResult,
    PairExecutionState,
    PairOrderPlan,
    SQLitePairExecutionJournal,
)
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.types import OrderState
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now

ABORT_SCOPE = "TESTNET_ABORT_CURRENCY_EVIDENCE_V1_NOT_PROFIT"
DIRECT_PARTIAL_CODES = frozenset({"SPOT_ENTRY_PARTIAL", "PERP_ENTRY_PARTIAL"})
ABORT_CODES = frozenset({"UNHEDGED_ENTRY_ABORTED", "UNHEDGED_ENTRY_NO_FILL", "ABORTED_ENTRY_RESTORED"}) | DIRECT_PARTIAL_CODES
REPAIR_TABLES = ("testnet_spot_exit_recoveries", "testnet_exit_recoveries")


def supported_abort_state(state: str, failure_code: str | None) -> bool:
    """A checkpoint claim is eligible only for the original terminal family."""
    return (state in (PairExecutionState.FLATTENED, PairExecutionState.FAILED)
            and failure_code in ABORT_CODES
            and (failure_code not in DIRECT_PARTIAL_CODES or state == PairExecutionState.FLATTENED))


def supported_abort(pair: PairExecutionResult | None) -> bool:
    if pair is None or not supported_abort_state(pair.state, pair.failure_code):
        return False
    if pair.failure_code not in DIRECT_PARTIAL_CODES:
        return True
    entry = pair.spot_entry if pair.failure_code == "SPOT_ENTRY_PARTIAL" else pair.perp_entry
    if (entry is None or entry.status not in (OrderState.CANCELED, OrderState.EXPIRED, OrderState.EXPIRED_IN_MATCH)
            or not 0 < entry.executed_quantity < entry.original_quantity):
        return False
    if pair.failure_code == "SPOT_ENTRY_PARTIAL":
        return pair.perp_entry is None
    return (pair.spot_entry is not None and pair.spot_entry.status is OrderState.FILLED
            and pair.spot_entry.executed_quantity == pair.spot_entry.original_quantity)


def journal_fingerprint(journal: SQLitePairExecutionJournal, plan: PairOrderPlan) -> str:
    """Bind the terminal parent, planned/actual intents and every repair receipt."""
    repairs: dict[str, object] = {}
    with journal._connect() as db:
        for table in REPAIR_TABLES:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                repairs[table] = [dict(row) for row in db.execute(
                    f"SELECT * FROM {table} WHERE attempt_id=? ORDER BY ordinal", (plan.attempt_id,))]
            else:
                repairs[table] = []
    value = {"scope": ABORT_SCOPE, "plan": plan, "pair": journal.get(plan.attempt_id),
             "intents": journal.intents(plan.attempt_id), "repairs": repairs}
    return hashlib.sha256(exact_json(value).encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(exact_json(value).encode("utf-8")).hexdigest()


def _frame_hash(plan: PairOrderPlan, before: object) -> str:
    return _digest({"attempt_id": plan.attempt_id, "symbol": plan.symbol,
                    "quantity": str(plan.spot_entry.quantity), "snapshots": before, "schema_version": 1})


def _account_states_valid(plan: PairOrderPlan, encoded: str, audit: dict[str, Any]) -> bool:
    """Source snapshot proof stays verifiable after restart; no current GET here."""
    try:
        states = json.loads(encoded)
        baseline = next(b["amount"] for s in audit["before"] if s["market"] == "SPOT"
                        for b in s["balances"] if b["asset"] == plan.symbol[:-4])
        if (len(states) != 1 + len(SUPPORTED_SYMBOLS) or states[0]["market"] != "SPOT"
                or states[0]["symbol"] != plan.symbol
                or type(states[0]["base_available"]) is not str or type(baseline) is not str
                or Decimal(states[0]["base_available"]) != Decimal(baseline)
                or states[0]["can_trade"] is not True or type(states[0]["open_order_count"]) is not int
                or states[0]["open_order_count"] != 0):
            return False
        for state, symbol in zip(states[1:], SUPPORTED_SYMBOLS, strict=True):
            if (state["market"] != "USDTM" or state["symbol"] != symbol
                    or state["can_trade"] is not True or state["one_way"] is not True
                    or state["single_asset"] is not True or type(state["leverage"]) is not str
                    or type(state["position_quantity"]) is not str
                    or Decimal(state["leverage"]) != 1 or Decimal(state["position_quantity"]) != 0
                    or type(state["open_order_count"]) is not int or state["open_order_count"] != 0):
                return False
        return True
    except (KeyError, TypeError, ValueError, ArithmeticError, StopIteration):
        return False


@dataclass(frozen=True)
class AbortProofSave:
    success: bool
    error_code: str | None = None


class AbortBalanceProofRepository:
    """Additive proof linking immutable currency archive to exact abort sources.

    It does not change a pair state, create a normal CLOSED reconciliation, or
    itself authorize a new order. The existing balance gate consumes it only
    after the ordinary bounded balance-check checkpoint has been archived.
    """

    def __init__(self, journal: SQLitePairExecutionJournal) -> None:
        self.journal = journal

    def initialize(self) -> None:
        BalanceAuditRepository(self.journal).initialize()
        with self.journal._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_abort_balance_proofs (
                attempt_id TEXT PRIMARY KEY REFERENCES testnet_pair_attempts(attempt_id),
                schema_version INTEGER NOT NULL CHECK(schema_version=1),
                audit_id TEXT NOT NULL UNIQUE REFERENCES testnet_balance_audits(audit_id),
                journal_sha256 TEXT NOT NULL, frame_sha256 TEXT NOT NULL,
                audit_sha256 TEXT NOT NULL, reconciliation_sha256 TEXT NOT NULL,
                account_states_json TEXT NOT NULL, created_at TEXT NOT NULL)""")

    def save(self, plan: PairOrderPlan, fingerprint: str, frame_sha256: str, audit_id: str,
             account_states: object) -> AbortProofSave:
        self.initialize()
        audit = BalanceAuditRepository(self.journal).get(audit_id)
        report = ReconciliationRepository(self.journal).get_report(plan.attempt_id)
        if (audit is None or report is None or report.get("accounting_scope") != ABORT_SCOPE
                or not supported_abort(self.journal.get(plan.attempt_id))
                or journal_fingerprint(self.journal, plan) != fingerprint
                or _frame_hash(plan, audit.get("before")) != frame_sha256
                or not _account_states_valid(plan, exact_json(account_states), audit)):
            return AbortProofSave(False, "ABORT_EVIDENCE_CONFLICT")
        identity = (audit_id, fingerprint, frame_sha256, _digest(audit), _digest(report), exact_json(account_states))
        with self.journal._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT attempt_id FROM testnet_balance_audits WHERE audit_id=?", (audit_id,)).fetchone()
            marker = db.execute("SELECT frame_sha256 FROM testnet_balance_frame_intents WHERE attempt_id=?",
                                (plan.attempt_id,)).fetchone()
            if (owner is None or owner[0] != plan.attempt_id or marker is None
                    or marker[0] != frame_sha256):
                return AbortProofSave(False, "ABORT_EVIDENCE_CONFLICT")
            old = db.execute("SELECT audit_id,journal_sha256,frame_sha256,audit_sha256,"
                             "reconciliation_sha256,account_states_json FROM testnet_abort_balance_proofs WHERE attempt_id=?",
                             (plan.attempt_id,)).fetchone()
            if old is not None:
                return AbortProofSave(tuple(old) == identity,
                                      None if tuple(old) == identity else "ABORT_EVIDENCE_CONFLICT")
            db.execute("INSERT INTO testnet_abort_balance_proofs VALUES(?,1,?,?,?,?,?,?,?)",
                       (plan.attempt_id, *identity, utc_now().isoformat()))
        return AbortProofSave(True)

    def get_verified(self, plan: PairOrderPlan, frame_sha256: str) -> dict[str, Any] | None:
        self.initialize()
        with self.journal._connect() as db:
            row = db.execute("SELECT * FROM testnet_abort_balance_proofs WHERE attempt_id=?", (plan.attempt_id,)).fetchone()
        if row is None or not supported_abort(self.journal.get(plan.attempt_id)):
            return None
        if row["journal_sha256"] != journal_fingerprint(self.journal, plan) or row["frame_sha256"] != frame_sha256:
            return None
        audit = BalanceAuditRepository(self.journal).get(row["audit_id"])
        report = ReconciliationRepository(self.journal).get_report(plan.attempt_id)
        if (audit is None or report is None or report.get("accounting_scope") != ABORT_SCOPE
                or row["audit_sha256"] != _digest(audit) or row["reconciliation_sha256"] != _digest(report)
                or _frame_hash(plan, audit.get("before")) != frame_sha256
                or not _account_states_valid(plan, row["account_states_json"], audit)):
            return None
        with self.journal._connect() as db:
            for trade in audit.get("trades", []):
                source = db.execute("SELECT attempt_id,evidence_json FROM testnet_trade_evidence "
                                    "WHERE market=? AND symbol=? AND trade_id=?",
                                    (trade["market"], trade["symbol"], trade["trade_id"])).fetchone()
                if source is None or tuple(source) != (plan.attempt_id, exact_json(trade)):
                    return None
        return dict(row)
