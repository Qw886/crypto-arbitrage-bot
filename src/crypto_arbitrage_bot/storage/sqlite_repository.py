"""File-backed SQLite repository. No imports from the calculation engines."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal, DecimalException
from enum import StrEnum
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.models.opportunity import (
    OpportunityRecord,
    ScenarioRecord,
    SourceTimingRecord,
)
from crypto_arbitrage_bot.storage.opportunity_records import utc_text
from crypto_arbitrage_bot.storage.scenario_codec import compact_record, expand_record
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

SCHEMA_VERSION = 1
CHILD_FIELDS = frozenset({"source_timings", "scenarios"})


@dataclass(frozen=True)
class SQLiteConfig:
    database_path: Path
    busy_timeout_ms: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.database_path, Path):
            raise TypeError("database_path must be Path")
        if type(self.busy_timeout_ms) is not int or not 0 <= self.busy_timeout_ms <= 60000:
            raise ValueError("busy_timeout_ms must be an integer between 0 and 60000")
        object.__setattr__(self, "database_path", self.database_path.resolve())


class RepositoryOperation(StrEnum):
    INITIALIZE = "INITIALIZE"
    SAVE = "SAVE"
    GET = "GET"
    LIST = "LIST"


class RepositoryError(RuntimeError):
    """An explicit persistence failure, never a change to the research decision."""

    def __init__(self, operation: RepositoryOperation, cause: Exception) -> None:
        self.operation = operation
        self.sqlite_errorcode: int | None = getattr(cause, "sqlite_errorcode", None)
        self.sqlite_errorname: str | None = getattr(cause, "sqlite_errorname", None)
        super().__init__(f"SQLite {operation.value} failed: {cause}")


def _columns(model: type[Any]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(model) if field.name not in CHILD_FIELDS)


def _encode(record: Any) -> tuple[str | None, ...]:
    result = []
    for name in _columns(type(record)):
        value = getattr(record, name)
        if value is not None and name in getattr(record, "DECIMAL_FIELDS", ()):
            if not isinstance(value, Decimal):
                raise TypeError(f"{name} must be Decimal, never float")
            value = str(value)
        elif value is not None and name in record.DATETIME_FIELDS:
            value = utc_text(value)
        elif name in getattr(record, "TUPLE_FIELDS", ()):
            if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
                raise TypeError(f"{name} must be a tuple of strings")
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a stored text value or None")
        result.append(value)
    return tuple(result)


def _decode(model: type[Any], row: sqlite3.Row) -> dict[str, Any]:
    values = {name: row[name] for name in _columns(model)}
    for name in getattr(model, "DECIMAL_FIELDS", ()):
        if values[name] is not None:
            values[name] = Decimal(values[name])
    for name in model.DATETIME_FIELDS:
        if values[name] is not None:
            values[name] = utc_timestamp(datetime.fromisoformat(values[name]))
    for name in getattr(model, "TUPLE_FIELDS", ()):
        codes = json.loads(values[name])
        if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
            raise ValueError(f"corrupt {name}: expected string list")
        values[name] = tuple(codes)
    return values


class SQLiteOpportunityRepository:
    """Explicit initialization and transactional inserts; duplicates never overwrite.

    Each operation closes its connection. A finite busy timeout is the only
    lock wait; there is no application retry loop or in-memory fallback.
    Reads use one transaction for consistent parent/child records.
    """

    def __init__(self, config: SQLiteConfig) -> None:
        if not isinstance(config, SQLiteConfig):
            raise TypeError("config must be SQLiteConfig")
        self.config = config

    @contextmanager
    def _connection(
        self, operation: RepositoryOperation, *, create: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            path = self.config.database_path
            if create:
                path.parent.mkdir(parents=True, exist_ok=True)
            # as_uri escapes user-supplied filename characters; mode is fixed.
            # Read/save calls cannot silently create an uninitialized database.
            uri = path.as_uri() + ("?mode=rwc" if create else "?mode=rw")
            connection = sqlite3.connect(uri, uri=True, timeout=0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.config.busy_timeout_ms}")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in ((0, SCHEMA_VERSION) if create else (SCHEMA_VERSION,)):
                raise sqlite3.DatabaseError("unsupported or uninitialized opportunity schema version")
            with connection:
                connection.execute("BEGIN IMMEDIATE" if operation in (
                    RepositoryOperation.INITIALIZE, RepositoryOperation.SAVE,
                ) else "BEGIN")
                yield connection
        except (sqlite3.Error, OSError, ValueError, TypeError, DecimalException, OverflowError) as exc:
            raise RepositoryError(operation, exc) from exc
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        with self._connection(RepositoryOperation.INITIALIZE, create=True) as connection:
            columns = ",".join(
                f"{name} TEXT" + (
                    " PRIMARY KEY NOT NULL" if name == "id" else
                    " NOT NULL" if name in ("created_at", "symbol", "decision") else ""
                ) for name in _columns(OpportunityRecord)
            )
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS opportunities ({columns}, "
                "CHECK (decision IN ('PASS','OBSERVE','REJECT')))"
            )
            for table, model in (("opportunity_sources", SourceTimingRecord),
                                 ("opportunity_scenarios", ScenarioRecord)):
                columns = ",".join(f"{name} TEXT" for name in _columns(model))
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ("
                    "opportunity_id TEXT NOT NULL REFERENCES opportunities(id), "
                    f"position INTEGER NOT NULL, {columns}, "
                    "PRIMARY KEY (opportunity_id, position))"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS opportunities_recent "
                "ON opportunities(created_at DESC, id DESC)"
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def save_opportunity(self, record: OpportunityRecord) -> str:
        if not isinstance(record, OpportunityRecord):
            raise TypeError("record must be OpportunityRecord")
        with self._connection(RepositoryOperation.SAVE) as connection:
            record = compact_record(record)
            names = _columns(OpportunityRecord)
            connection.execute(
                f"INSERT INTO opportunities ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                _encode(record),
            )
            for table, children in (("opportunity_sources", record.source_timings),
                                    ("opportunity_scenarios", record.scenarios)):
                for position, child in enumerate(children):
                    names = ("opportunity_id", "position", *_columns(type(child)))
                    connection.execute(
                        f"INSERT INTO {table} ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                        (record.id, position, *_encode(child)),
                    )
        return record.id

    @staticmethod
    def _load(connection: sqlite3.Connection, row: sqlite3.Row) -> OpportunityRecord:
        sources = connection.execute(
            "SELECT * FROM opportunity_sources WHERE opportunity_id = ? ORDER BY position", (row["id"],),
        ).fetchall()
        scenarios = connection.execute(
            "SELECT * FROM opportunity_scenarios WHERE opportunity_id = ? ORDER BY position", (row["id"],),
        ).fetchall()
        return expand_record(OpportunityRecord(
            **_decode(OpportunityRecord, row),
            source_timings=tuple(SourceTimingRecord(**_decode(SourceTimingRecord, item)) for item in sources),
            scenarios=tuple(ScenarioRecord(**_decode(ScenarioRecord, item)) for item in scenarios),
        ))

    def get_opportunity_by_id(self, opportunity_id: str) -> OpportunityRecord | None:
        with self._connection(RepositoryOperation.GET) as connection:
            row = connection.execute("SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)).fetchone()
            return None if row is None else self._load(connection, row)

    def list_recent_opportunities(self, limit: int = 100) -> tuple[OpportunityRecord, ...]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        with self._connection(RepositoryOperation.LIST) as connection:
            rows = connection.execute(
                "SELECT * FROM opportunities ORDER BY created_at DESC, id DESC LIMIT ?", (limit,),
            ).fetchall()
            return tuple(self._load(connection, row) for row in rows)
