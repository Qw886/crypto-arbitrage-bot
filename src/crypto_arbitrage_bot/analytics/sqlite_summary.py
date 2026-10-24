"""Read all stored research observations in one read-only SQLite transaction."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from decimal import Decimal, DecimalException
from typing import Any

from crypto_arbitrage_bot.analytics.summary import (
    RecordedObservation,
    RecordedScenario,
    ResearchSummary,
    summarize_observations,
)
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.scenario import ScenarioType
from crypto_arbitrage_bot.storage.sqlite_repository import SCHEMA_VERSION, SQLiteConfig
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class AnalyticsReadError(RuntimeError):
    """An unreadable/corrupt database is not reported as an empty history."""

    def __init__(self, cause: Exception) -> None:
        self.sqlite_errorcode: int | None = getattr(cause, "sqlite_errorcode", None)
        self.sqlite_errorname: str | None = getattr(cause, "sqlite_errorname", None)
        super().__init__(f"Research analytics read failed: {cause}")


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected Decimal stored as TEXT")
    return require_decimal(Decimal(value), "stored Decimal")


def _codes(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    codes = json.loads(value)
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ValueError("reason codes must be a JSON list of strings")
    return tuple(codes)


def read_observations(config: SQLiteConfig) -> tuple[RecordedObservation, ...]:
    """Project needed columns only; never use recent-record limits or SQL casts.

    mode=ro prevents creation and writes; query_only additionally protects this
    connection. BEGIN holds one consistent view of parents and scenarios.
    """
    if not isinstance(config, SQLiteConfig):
        raise TypeError("config must be SQLiteConfig")
    try:
        with closing(sqlite3.connect(config.database_path.as_uri() + "?mode=ro",
                                     uri=True, timeout=0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute(f"PRAGMA busy_timeout = {config.busy_timeout_ms}")
            connection.execute("BEGIN")
            if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise ValueError("unsupported or uninitialized opportunity schema version")
            parents = connection.execute(
                "SELECT id, created_at, symbol, decision, net_profit, net_return, funding_rate, "
                "data_quality_status, decision_reason_codes, data_quality_reason_codes "
                "FROM opportunities ORDER BY created_at, id"
            ).fetchall()
            children: dict[str, list[RecordedScenario]] = {row["id"]: [] for row in parents}
            for row in connection.execute(
                "SELECT opportunity_id, scenario_type, net_profit, net_return "
                "FROM opportunity_scenarios ORDER BY opportunity_id, position"
            ):
                if row["opportunity_id"] not in children:
                    raise ValueError("orphan scenario record")
                children[row["opportunity_id"]].append(RecordedScenario(
                    ScenarioType(row["scenario_type"]), _decimal(row["net_profit"]),
                    _decimal(row["net_return"]),
                ))
            results = []
            for row in parents:
                if row["data_quality_status"] not in (None, "VALID", "INVALID"):
                    raise ValueError("unknown data quality status")
                results.append(RecordedObservation(
                    row["id"], utc_timestamp(datetime.fromisoformat(row["created_at"])),
                    row["symbol"], Decision(row["decision"]), _decimal(row["net_profit"]),
                    _decimal(row["net_return"]), _decimal(row["funding_rate"]),
                    row["data_quality_status"], _codes(row["decision_reason_codes"]),
                    _codes(row["data_quality_reason_codes"]), tuple(children[row["id"]]),
                ))
            return tuple(results)
    except (sqlite3.Error, OSError, ValueError, TypeError, DecimalException) as exc:
        raise AnalyticsReadError(exc) from exc


def summarize_database(config: SQLiteConfig) -> ResearchSummary:
    """Describe historical estimates as recorded; no trading performance claims."""
    return summarize_observations(read_observations(config))
