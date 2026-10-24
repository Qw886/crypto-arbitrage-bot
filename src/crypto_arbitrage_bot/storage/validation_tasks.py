"""Additive scheduler table; existing Opportunity/Validation writes stay unchanged."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from uuid import NAMESPACE_URL, uuid4, uuid5

from crypto_arbitrage_bot.models.validation_task import TaskFailure, TaskStatus, ValidationTask
from crypto_arbitrage_bot.storage.opportunity_records import utc_text
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SCHEMA_VERSION,
    RepositoryError,
    RepositoryOperation,
    SQLiteConfig,
)
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.failure_summary import limit_failure_reason


def _task(row: sqlite3.Row) -> ValidationTask:
    fields = dict(row)
    for name in ("due_at", "created_at", "updated_at", "lease_until"):
        if fields[name] is not None:
            fields[name] = utc_timestamp(datetime.fromisoformat(fields[name]))
    fields["status"] = TaskStatus(fields["status"])
    if fields["failure_code"] is not None:
        fields["failure_code"] = TaskFailure(fields["failure_code"])
    return ValidationTask(**fields)


class SQLiteValidationTaskRepository:
    def __init__(self, config: SQLiteConfig) -> None:
        self.config = config

    @contextmanager
    def _connection(self, operation: RepositoryOperation) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self.config.database_path.as_uri() + "?mode=rw", uri=True, timeout=0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA busy_timeout={self.config.busy_timeout_ms}")
            with connection:
                connection.execute("BEGIN IMMEDIATE" if operation in (
                    RepositoryOperation.INITIALIZE, RepositoryOperation.SAVE) else "BEGIN")
                if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                    raise sqlite3.DatabaseError("opportunity schema must already be initialized")
                yield connection
        except (sqlite3.Error, OSError, ValueError, TypeError, OverflowError) as exc:
            raise RepositoryError(operation, exc) from exc
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        with self._connection(RepositoryOperation.INITIALIZE) as db:
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS opportunity_validation_identity ON opportunities(id,symbol)")
            db.execute("""CREATE TABLE IF NOT EXISTS validation_tasks (
                id TEXT PRIMARY KEY NOT NULL,
                opportunity_id TEXT NOT NULL, symbol TEXT NOT NULL,
                horizon_seconds INTEGER NOT NULL CHECK(horizon_seconds > 0),
                due_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETED','FAILED')),
                validation_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                lease_until TEXT, claim_token TEXT, failure_code TEXT, failure_reason TEXT,
                UNIQUE(opportunity_id,horizon_seconds),
                FOREIGN KEY(opportunity_id,symbol) REFERENCES opportunities(id,symbol))""")
            db.execute("CREATE INDEX IF NOT EXISTS validation_tasks_due ON validation_tasks(status,due_at,lease_until)")

    def create_tasks(self, opportunity_id: str, horizons: tuple[int, ...], *, now: datetime) -> None:
        if not horizons or any(type(h) is not int or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be distinct positive integer seconds")
        at = utc_text(now)
        with self._connection(RepositoryOperation.SAVE) as db:
            parent = db.execute("SELECT symbol,created_at FROM opportunities WHERE id=?", (opportunity_id,)).fetchone()
            if parent is None:
                raise ValueError("opportunity does not exist")
            for horizon in horizons:
                identity = str(uuid5(NAMESPACE_URL, f"crypto-arbitrage-bot/validation/{opportunity_id}/{horizon}"))
                due = utc_timestamp(datetime.fromisoformat(parent["created_at"])) + timedelta(seconds=horizon)
                db.execute("""INSERT INTO validation_tasks
                    (id,opportunity_id,symbol,horizon_seconds,due_at,status,validation_id,created_at,updated_at)
                    VALUES(?,?,?,?,?,'PENDING',?,?,?)
                    ON CONFLICT(opportunity_id,horizon_seconds) DO NOTHING""",
                           (identity, opportunity_id, parent["symbol"], horizon, utc_text(due), identity, at, at))

    def reconcile(self, horizons: tuple[int, ...], *, now: datetime, limit: int) -> None:
        """Bounded repair of save/create crash gaps, including pre-scheduler records."""
        for horizon in horizons:
            with self._connection(RepositoryOperation.LIST) as db:
                ids = db.execute("""SELECT o.id FROM opportunities o WHERE NOT EXISTS (
                    SELECT 1 FROM validation_tasks t WHERE t.opportunity_id=o.id AND t.horizon_seconds=?)
                    ORDER BY o.created_at,o.id LIMIT ?""", (horizon, limit)).fetchall()
            for row in ids:
                self.create_tasks(row["id"], horizons, now=now)

    def get(self, task_id: str) -> ValidationTask | None:
        with self._connection(RepositoryOperation.GET) as db:
            row = db.execute("SELECT * FROM validation_tasks WHERE id=?", (task_id,)).fetchone()
            return None if row is None else _task(row)

    def list_for_opportunity(self, opportunity_id: str) -> tuple[ValidationTask, ...]:
        with self._connection(RepositoryOperation.LIST) as db:
            return tuple(_task(row) for row in db.execute(
                "SELECT * FROM validation_tasks WHERE opportunity_id=? ORDER BY horizon_seconds", (opportunity_id,)))

    def list_due_tasks(self, now: datetime, *, limit: int,
                       max_lateness_seconds: int | None = None) -> tuple[ValidationTask, ...]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        at = utc_text(now)
        if max_lateness_seconds is not None and (
            type(max_lateness_seconds) is not int or max_lateness_seconds < 0
        ):
            raise ValueError("max_lateness_seconds must be a nonnegative integer or None")
        cutoff = None if max_lateness_seconds is None else utc_text(
            utc_timestamp(now) - timedelta(seconds=max_lateness_seconds))
        with self._connection(RepositoryOperation.LIST) as db:
            return tuple(_task(row) for row in db.execute("""SELECT * FROM validation_tasks WHERE
                due_at<=? AND (status='PENDING' OR (status='RUNNING' AND lease_until<=?))
                ORDER BY CASE WHEN due_at < ? THEN 1 ELSE 0 END,due_at,id LIMIT ?""",
                (at, at, cutoff, limit)))

    def mark_running(self, task_id: str, *, now: datetime, lease_seconds: int) -> ValidationTask | None:
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        at = utc_text(now)
        until = utc_text(utc_timestamp(now) + timedelta(seconds=lease_seconds))
        with self._connection(RepositoryOperation.SAVE) as db:
            cursor = db.execute("""UPDATE validation_tasks SET status='RUNNING',attempts=attempts+1,
                updated_at=?,lease_until=?,claim_token=? WHERE id=? AND due_at<=? AND
                (status='PENDING' OR (status='RUNNING' AND lease_until<=?))""",
                                (at, until, str(uuid4()), task_id, at, at))
            if cursor.rowcount != 1:
                return None
            return _task(db.execute("SELECT * FROM validation_tasks WHERE id=?", (task_id,)).fetchone())

    def finish(self, task: ValidationTask, *, now: datetime, failure: TaskFailure | None = None,
               reason: str | None = None) -> bool:
        """Only the current claim owner may change task state."""
        status = TaskStatus.COMPLETED if failure is None else TaskStatus.FAILED
        with self._connection(RepositoryOperation.SAVE) as db:
            cursor = db.execute("""UPDATE validation_tasks SET status=?,updated_at=?,lease_until=NULL,
                claim_token=NULL,failure_code=?,failure_reason=?
                WHERE id=? AND status='RUNNING' AND claim_token=?""",
                                (status.value, utc_text(now), failure, limit_failure_reason(reason), task.id, task.claim_token))
            return cursor.rowcount == 1
