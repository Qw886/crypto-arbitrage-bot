"""Read original decisions and forward results in one SQLite read-only snapshot."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from typing import Any

from crypto_arbitrage_bot.analytics.forward_summary import (
    ForwardAnalyticsConfig,
    ForwardAnalyticsSummary,
    ValidationObservation,
    summarize_forward_observations,
)
from crypto_arbitrage_bot.analytics.sqlite_summary import AnalyticsReadError
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SCHEMA_VERSION,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("financial field must be Decimal text")
    return require_decimal(Decimal(value), "stored financial field")


def _check_research_identity(connection: sqlite3.Connection, opportunity_id: str,
                             identity: ResearchInstrument) -> None:
    """Use the existing lossless storage decoder; incomplete math may still be counted.

    The caller's identity and both recorded entry books must agree. Do not infer
    identity from a symbol string or require positive PnL for descriptive analytics.
    """
    row = connection.execute("SELECT * FROM opportunities WHERE id=?", (opportunity_id,)).fetchone()
    if row is None:
        raise ValueError("missing research opportunity")
    record = SQLiteOpportunityRepository._load(connection, row)
    snapshot = json.loads(record.details_json)["snapshot"]
    expected = {"symbol": identity.symbol, "base_asset": identity.base_asset}
    if (not isinstance(snapshot, dict) or record.symbol != identity.symbol
            or snapshot.get("symbol") != identity.symbol or snapshot.get("research_instrument") != expected):
        raise ValueError("recorded research snapshot identity mismatch")
    for name in ("spot_book", "perp_book"):
        book = snapshot.get(name)
        if (not isinstance(book, dict) or book.get("symbol") != identity.symbol
                or book.get("research_instrument") != expected):
            raise ValueError("recorded research book identity mismatch")


def summarize_forward_database(
    database: SQLiteConfig, config: ForwardAnalyticsConfig,
) -> ForwardAnalyticsSummary:
    """Full history, including parents without validation; no writes or SQL AVG.

    Both schemas must be initialized. Missing tables/files are read failures,
    not empty results. Orphans or symbol mismatches are never silently joined away.
    """
    if not isinstance(database, SQLiteConfig) or not isinstance(config, ForwardAnalyticsConfig):
        raise TypeError("SQLiteConfig and ForwardAnalyticsConfig are required")
    try:
        with closing(sqlite3.connect(database.database_path.as_uri() + "?mode=ro", uri=True, timeout=0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute(f"PRAGMA busy_timeout={database.busy_timeout_ms}")
            connection.execute("BEGIN")
            if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise ValueError("unsupported opportunity schema")
            parents = {}
            identities = {item.symbol: item for item in config.research_instruments}
            for row in connection.execute("SELECT id,symbol,decision FROM opportunities"):
                identity = identities.get(row["symbol"])
                if row["symbol"] not in SUPPORTED_SYMBOLS and identity is None:
                    raise ValueError("unsupported opportunity symbol: explicit research identity required")
                if identity is not None:
                    _check_research_identity(connection, row["id"], identity)
                parents[row["id"]] = (row["symbol"], Decision(row["decision"]))
            observations = []
            for row in connection.execute(
                "SELECT id,opportunity_id,symbol,status,validation_time,holding_duration,"
                "validated_simulated_net_profit,validated_simulated_net_return,profit_error,return_error,"
                "basis_change,funding_total FROM forward_validations ORDER BY id"
            ):
                parent = parents.get(row["opportunity_id"])
                if parent is None or parent[0] != row["symbol"]:
                    raise ValueError("validation opportunity/symbol mismatch")
                status = ValidationStatus(row["status"])
                at = None if row["validation_time"] is None else utc_timestamp(datetime.fromisoformat(row["validation_time"]))
                duration = None if row["holding_duration"] is None else timedelta(microseconds=int(row["holding_duration"]))
                if duration is not None and duration < timedelta(0):
                    raise ValueError("negative holding duration")
                # Failed attempts contribute status/counts only, even if corrupt
                # legacy data happens to retain non-NULL numeric fields.
                numbers = tuple(_decimal(row[name]) if status is ValidationStatus.VALIDATED else None
                                for name in ("validated_simulated_net_profit", "validated_simulated_net_return",
                                             "profit_error", "return_error", "basis_change", "funding_total"))
                observations.append(ValidationObservation(row["id"], row["opportunity_id"], row["symbol"],
                                                          parent[1], status, at, duration, *numbers))
        return summarize_forward_observations(frozenset(parents), observations, config)
    except (sqlite3.Error, ValueError, TypeError, KeyError, OSError, DecimalException, OverflowError) as exc:
        raise AnalyticsReadError(exc) from exc
