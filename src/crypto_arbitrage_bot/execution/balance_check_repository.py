"""Durable bounded testnet balance-check lifecycle, separate from positions."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from uuid import uuid4

from crypto_arbitrage_bot.execution.abort_balance_proof import supported_abort_state
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_frame_store import BeforeBalanceFrame
from crypto_arbitrage_bot.execution.pair_coordinator import SQLitePairExecutionJournal
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class BalanceCheckStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BalanceCheck:
    attempt_id: str
    status: BalanceCheckStatus
    attempts: int
    next_check_at: datetime
    lease_until: datetime | None
    token: str | None
    audit_id: str | None
    balance_status: str | None
    failure_code: str | None


@dataclass(frozen=True)
class BeforeBalanceIntent:
    attempt_id: str
    symbol: str
    quantity: Decimal
    frame_sha256: str


class BalanceCheckRepository:
    def __init__(self, journal: SQLitePairExecutionJournal):
        self.journal = journal

    def initialize(self) -> None:
        BalanceAuditRepository(self.journal).initialize()
        with self.journal._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            # No FK: commit this marker before a parent/order intent exists.
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_balance_frame_intents (
                attempt_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL CHECK(schema_version=1),
                symbol TEXT NOT NULL, quantity TEXT NOT NULL, frame_sha256 TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS testnet_balance_checks (
                attempt_id TEXT PRIMARY KEY REFERENCES testnet_pair_attempts(attempt_id),
                schema_version INTEGER NOT NULL CHECK(schema_version=1),
                status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','ARCHIVED','FAILED')),
                attempts INTEGER NOT NULL CHECK(attempts>=0), next_check_at TEXT NOT NULL,
                lease_until TEXT, token TEXT, audit_id TEXT REFERENCES testnet_balance_audits(audit_id),
                balance_status TEXT, failure_code TEXT, updated_at TEXT NOT NULL,
                CHECK(status!='ARCHIVED' OR audit_id IS NOT NULL))""")

    def register_before(self, frame: BeforeBalanceFrame) -> bool:
        self.initialize()
        digest = hashlib.sha256(exact_json(frame).encode('utf-8')).hexdigest()
        identity = (frame.symbol, str(frame.quantity), digest)
        with self.journal._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT symbol,quantity,frame_sha256 FROM testnet_balance_frame_intents WHERE attempt_id=?',
                               (frame.attempt_id,)).fetchone()
            if prior is not None:
                return tuple(prior) == identity
            if db.execute('SELECT 1 FROM testnet_pair_attempts WHERE attempt_id=?', (frame.attempt_id,)).fetchone():
                return False  # No post-entry marker creation.
            db.execute('INSERT INTO testnet_balance_frame_intents VALUES(?,1,?,?,?)', (frame.attempt_id, *identity))
            return True

    def before_intents(self) -> tuple[BeforeBalanceIntent, ...]:
        self.initialize()
        with self.journal._connect() as db:
            rows = db.execute('SELECT attempt_id,symbol,quantity,frame_sha256 FROM testnet_balance_frame_intents ORDER BY attempt_id').fetchall()
        return tuple(BeforeBalanceIntent(r[0], r[1], Decimal(r[2]), r[3]) for r in rows)

    def get(self, attempt_id: str) -> BalanceCheck | None:
        self.initialize()
        with self.journal._connect() as db:
            row = db.execute("SELECT * FROM testnet_balance_checks WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            return None
        return BalanceCheck(row['attempt_id'], BalanceCheckStatus(row['status']), row['attempts'],
            datetime.fromisoformat(row['next_check_at']),
            None if row['lease_until'] is None else datetime.fromisoformat(row['lease_until']),
            row['token'], row['audit_id'], row['balance_status'], row['failure_code'])

    def claim(self, attempt_id: str, now: datetime, *, max_attempts: int, lease_seconds: int) -> BalanceCheck | None:
        now = utc_timestamp(now)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError('bounded balance check attempts required')
        if type(lease_seconds) is not int or not 30 <= lease_seconds <= 3600:
            raise ValueError('bounded balance check lease required')
        self.initialize()
        with self.journal._connect() as db:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            parent = db.execute('SELECT state,failure_code FROM testnet_pair_attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
            if parent is None:
                return None
            if parent[0] != 'CLOSED' and not supported_abort_state(parent[0], parent[1]):
                return None
            row = db.execute('SELECT * FROM testnet_balance_checks WHERE attempt_id=?', (attempt_id,)).fetchone()
            if row is not None:
                if row['status'] in ('ARCHIVED', 'FAILED') or datetime.fromisoformat(row['next_check_at']) > now:
                    return None
                if row['status'] == 'RUNNING' and datetime.fromisoformat(row['lease_until']) > now:
                    return None
                if row['attempts'] >= max_attempts:
                    db.execute("UPDATE testnet_balance_checks SET status='FAILED',token=NULL,lease_until=NULL,"
                               "failure_code='CHECK_ATTEMPTS_EXHAUSTED',updated_at=? WHERE attempt_id=?",
                               (now.isoformat(), attempt_id))
                    return None
            attempts = 1 if row is None else row['attempts'] + 1
            token = uuid4().hex
            lease = now + timedelta(seconds=lease_seconds)
            db.execute("INSERT INTO testnet_balance_checks VALUES(?,1,'RUNNING',?,?,?,?,NULL,NULL,NULL,?) "
                       "ON CONFLICT(attempt_id) DO UPDATE SET status='RUNNING',attempts=excluded.attempts,"
                       "next_check_at=excluded.next_check_at,lease_until=excluded.lease_until,token=excluded.token,"
                       "audit_id=NULL,balance_status=NULL,failure_code=NULL,updated_at=excluded.updated_at",
                       (attempt_id, attempts, now.isoformat(), lease.isoformat(), token, now.isoformat()))
        return BalanceCheck(attempt_id, BalanceCheckStatus.RUNNING, attempts, now, lease, token, None, None, None)

    def finish(self, claim: BalanceCheck, now: datetime, *, max_attempts: int, retry_seconds: int,
               audit_id: str | None = None, balance_status: str | None = None,
               failure_code: str | None = None, terminal_failure: bool = False) -> bool:
        now = utc_timestamp(now)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError('bounded balance check attempts required')
        if type(retry_seconds) is not int or not 1 <= retry_seconds <= 3600:
            raise ValueError('positive bounded balance retry interval required')
        status = (BalanceCheckStatus.ARCHIVED if audit_id is not None else BalanceCheckStatus.FAILED
                  if terminal_failure or claim.attempts >= max_attempts else BalanceCheckStatus.PENDING)
        with self.journal._connect() as db:
            db.execute('PRAGMA foreign_keys=ON')
            if audit_id is not None:
                audit = db.execute('SELECT attempt_id,status FROM testnet_balance_audits WHERE audit_id=?', (audit_id,)).fetchone()
                if audit is None or tuple(audit) != (claim.attempt_id, balance_status):
                    return False
            changed = db.execute("UPDATE testnet_balance_checks SET status=?,next_check_at=?,lease_until=NULL,"
                "token=NULL,audit_id=?,balance_status=?,failure_code=?,updated_at=? "
                "WHERE attempt_id=? AND status='RUNNING' AND token=?",
                (status.value, (now + timedelta(seconds=retry_seconds)).isoformat(), audit_id, balance_status,
                 failure_code, now.isoformat(), claim.attempt_id, claim.token))
            return changed.rowcount == 1

    def recover_archive(self, attempt_id: str, audit_id: str, now: datetime) -> bool:
        """A prior committed audit survives a crash before the checkpoint commit."""
        self.initialize()
        now = utc_timestamp(now)
        with self.journal._connect() as db:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            audit = db.execute('SELECT attempt_id,status FROM testnet_balance_audits WHERE audit_id=?', (audit_id,)).fetchone()
            if audit is None or audit[0] != attempt_id:
                return False
            prior = db.execute('SELECT status,audit_id FROM testnet_balance_checks WHERE attempt_id=?', (attempt_id,)).fetchone()
            if prior is not None and prior[0] == 'ARCHIVED':
                return bool(prior[1] == audit_id)
            db.execute("INSERT INTO testnet_balance_checks VALUES(?,1,'ARCHIVED',0,?,NULL,NULL,?,?,NULL,?) "
                       "ON CONFLICT(attempt_id) DO UPDATE SET status='ARCHIVED',lease_until=NULL,token=NULL,"
                       "audit_id=excluded.audit_id,balance_status=excluded.balance_status,failure_code=NULL,"
                       "updated_at=excluded.updated_at",
                       (attempt_id, now.isoformat(), audit_id, audit[1], now.isoformat()))
            return True
