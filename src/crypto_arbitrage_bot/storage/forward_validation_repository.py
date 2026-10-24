"""Independent forward validation storage, linked to existing opportunities."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from enum import StrEnum
from typing import Any

from crypto_arbitrage_bot.models.forward_validation import ForwardValidationResult
from crypto_arbitrage_bot.models.validation_record import ForwardValidationRecord
from crypto_arbitrage_bot.storage.forward_validation_records import build_forward_validation_record
from crypto_arbitrage_bot.storage.opportunity_records import utc_text
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SCHEMA_VERSION,
    RepositoryError,
    RepositoryOperation,
    SQLiteConfig,
)
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp

COLUMNS = tuple(field.name for field in fields(ForwardValidationRecord))


class ValidationPersistenceStatus(StrEnum):
    SAVED = "SAVED"
    BUILD_FAILED = "BUILD_FAILED"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"


@dataclass(frozen=True)
class ValidationPersistenceFailure:
    message: str
    operation: RepositoryOperation
    sqlite_errorcode: int | None = None
    sqlite_errorname: str | None = None


@dataclass(frozen=True)
class ValidationSaveResult:
    status: ValidationPersistenceStatus
    record: ForwardValidationRecord | None = None
    saved_id: str | None = None
    failure: ValidationPersistenceFailure | None = None


def _encode(record: ForwardValidationRecord) -> tuple[str | None, ...]:
    encoded = []
    for name in COLUMNS:
        value = getattr(record, name)
        if value is not None:
            if name in record.DECIMAL_FIELDS:
                value = str(require_decimal(value, name))
            elif name in record.DATETIME_FIELDS:
                value = utc_text(value)
            elif name == "holding_duration":
                value = str((value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds)
            elif name == "reason_codes":
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if not isinstance(value, str):
                raise TypeError(f"{name} must serialize to TEXT")
        encoded.append(value)
    return tuple(encoded)


def _decode(row: sqlite3.Row) -> ForwardValidationRecord:
    values: dict[str, Any] = dict(row)
    for name in ForwardValidationRecord.DECIMAL_FIELDS:
        if values[name] is not None:
            if not isinstance(values[name], str):
                raise ValueError("stored Decimal must be TEXT")
            values[name] = require_decimal(Decimal(values[name]), name)
    for name in ForwardValidationRecord.DATETIME_FIELDS:
        if values[name] is not None:
            values[name] = utc_timestamp(datetime.fromisoformat(values[name]))
    if values["holding_duration"] is not None:
        values["holding_duration"] = timedelta(microseconds=int(values["holding_duration"]))
    codes = json.loads(values["reason_codes"])
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ValueError("corrupt validation reason codes")
    values["reason_codes"] = tuple(codes)
    return ForwardValidationRecord(**values)


class SQLiteForwardValidationRepository:
    """Additive table on an initialized opportunity database; no parent writes.

    Initialization is explicit and idempotent. Distinct UUIDs allow repeated
    attempts even at the same validation time; existing IDs never overwrite.
    """

    def __init__(self, config: SQLiteConfig) -> None:
        if not isinstance(config, SQLiteConfig):
            raise TypeError("config must be SQLiteConfig")
        self.config = config

    @contextmanager
    def _connection(self, operation: RepositoryOperation) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            connection = sqlite3.connect(self.config.database_path.as_uri() + "?mode=rw", uri=True, timeout=0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.config.busy_timeout_ms}")
            with connection:
                connection.execute("BEGIN IMMEDIATE" if operation in (
                    RepositoryOperation.INITIALIZE, RepositoryOperation.SAVE,
                ) else "BEGIN")
                if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                    raise sqlite3.DatabaseError("opportunity schema must already be initialized")
                yield connection
        except (sqlite3.Error, OSError, ValueError, TypeError, DecimalException, OverflowError) as exc:
            raise RepositoryError(operation, exc) from exc
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        with self._connection(RepositoryOperation.INITIALIZE) as connection:
            # Composite FK enforces both identity and symbol, including parent updates.
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS opportunity_validation_identity "
                               "ON opportunities(id, symbol)")
            columns = ",".join(f"{name} TEXT" + (
                " PRIMARY KEY NOT NULL" if name == "id" else
                " NOT NULL" if name in ("opportunity_id", "symbol", "status", "reason_codes", "created_at", "details_json") else ""
            ) for name in COLUMNS)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS forward_validations ({columns}, "
                "FOREIGN KEY(opportunity_id,symbol) REFERENCES opportunities(id,symbol), "
                "CHECK(status IN ('VALIDATED','INCOMPLETE','INVALID')), "
                "CHECK(entry_time IS NULL OR validation_time IS NULL OR validation_time > entry_time), "
                "CHECK(status != 'VALIDATED' OR (entry_time IS NOT NULL AND validation_time IS NOT NULL "
                "AND validated_simulated_net_profit IS NOT NULL AND validated_simulated_net_return IS NOT NULL)), "
                "CHECK(status = 'VALIDATED' OR (validated_simulated_net_profit IS NULL "
                "AND validated_simulated_net_return IS NULL)))"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS forward_validations_opportunity "
                               "ON forward_validations(opportunity_id,validation_time,id)")
            connection.execute("CREATE INDEX IF NOT EXISTS forward_validations_recent "
                               "ON forward_validations(created_at DESC,id DESC)")

    def save_forward_validation(
        self, result: ForwardValidationResult, *, validation_id: str | None = None,
        created_at: datetime | None = None,
    ) -> ValidationSaveResult:
        try:
            record = build_forward_validation_record(result, validation_id=validation_id, created_at=created_at)
        except (ValueError, TypeError, AttributeError, DecimalException, OverflowError) as exc:
            return ValidationSaveResult(ValidationPersistenceStatus.BUILD_FAILED,
                                        failure=ValidationPersistenceFailure(str(exc), RepositoryOperation.SAVE))
        try:
            with self._connection(RepositoryOperation.SAVE) as connection:
                connection.execute(
                    f"INSERT INTO forward_validations ({','.join(COLUMNS)}) "
                    f"VALUES ({','.join('?' for _ in COLUMNS)})", _encode(record),
                )
        except RepositoryError as exc:
            return ValidationSaveResult(
                ValidationPersistenceStatus.PERSISTENCE_FAILED, record,
                failure=ValidationPersistenceFailure(str(exc), exc.operation,
                                                     exc.sqlite_errorcode, exc.sqlite_errorname),
            )
        return ValidationSaveResult(ValidationPersistenceStatus.SAVED, record, record.id)

    def get_forward_validation_by_id(self, validation_id: str) -> ForwardValidationRecord | None:
        with self._connection(RepositoryOperation.GET) as connection:
            row = connection.execute("SELECT * FROM forward_validations WHERE id=?", (validation_id,)).fetchone()
            return None if row is None else _decode(row)

    def list_validations_for_opportunity(self, opportunity_id: str) -> tuple[ForwardValidationRecord, ...]:
        with self._connection(RepositoryOperation.LIST) as connection:
            rows = connection.execute("SELECT * FROM forward_validations WHERE opportunity_id=? "
                                      "ORDER BY validation_time,id", (opportunity_id,)).fetchall()
            return tuple(_decode(row) for row in rows)

    def list_recent_validations(self, limit: int = 100) -> tuple[ForwardValidationRecord, ...]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        with self._connection(RepositoryOperation.LIST) as connection:
            rows = connection.execute("SELECT * FROM forward_validations "
                                      "ORDER BY created_at DESC,id DESC LIMIT ?", (limit,)).fetchall()
            return tuple(_decode(row) for row in rows)
