"""Read the original fixed pilot in one SQLite snapshot; never collect or trade."""

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.pilot_outcome import (
    PILOT_HORIZONS,
    PILOT_SYMBOLS,
    PilotEntry,
    assess_pilot_v1,
)
from crypto_arbitrage_bot.legacy_research_control import load_legacy_job
from crypto_arbitrage_bot.storage.forward_validation_repository import _decode
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SCHEMA_VERSION,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.storage.validation_tasks import _task
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import ControlError


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate evidence fields")
    return result


def read_pilot_outcome(job_directory: Path, entry_evidence: Path, *,
                       evaluated_at: datetime) -> dict[str, Any]:
    """Review this exploratory cohort without giving it retrospective registration.

    The archived entry capture fixes the four IDs. Config, original deadline and
    database come from the existing validated recovery binding. Missing tables,
    extra parents or corrupt inputs raise an error, never produce empty success.
    """
    at = utc_timestamp(evaluated_at)
    job = load_legacy_job(job_directory)
    research = job.settings.research
    assert research is not None and research.validation_scheduler is not None
    if (research.config_version != "candidate-multisettlement-observation-v1"
            or research.validation_scheduler.validation_horizons_seconds != PILOT_HORIZONS
            or research.validation_scheduler.max_lateness_seconds != 300):
        raise ValueError("original pilot protocol configuration required")
    with entry_evidence.open("rb") as stream:
        raw = stream.read(16_000_001)
    if len(raw) > 16_000_000:
        raise ValueError("entry evidence exceeds its bounded size")
    manifest = json.loads(raw, object_pairs_hook=_unique)
    opened = utc_timestamp(datetime.fromisoformat(manifest["started_at"]))
    closed = utc_timestamp(datetime.fromisoformat(manifest["finished_at"]))
    if (manifest["selection_committed_before_entries"] is not True
            or Path(manifest["config"]).resolve() != Path(job.binding["config_path"])
            or not opened <= closed <= job.started <= at):
        raise ValueError("original captured entry scope and timing required")
    references = {}
    captures = manifest["captures"]
    if not isinstance(captures, list) or len(captures) != 4:
        raise ValueError("all four original captures required")
    for item in captures:
        symbol = item["symbol"]
        groups = item["cost_result"]["captures"]
        if symbol not in PILOT_SYMBOLS or symbol in references or len(groups) != 1:
            raise ValueError("distinct original pilot symbols required")
        group = groups[0]
        windows = group["windows"]
        if group["symbol"] != symbol or group["status"] != "SAVED" or len(windows) != 1:
            raise ValueError("original saved entry required")
        reference = windows[0]
        if reference["persistence_status"] != "SAVED" or reference["profile"] != "next_settlement":
            raise ValueError("original persisted reference required")
        references[symbol] = reference
    expected_ids = {r["opportunity_id"] for r in references.values()}
    if len(expected_ids) != 4:
        raise ValueError("four distinct captured IDs required")
    with closing(sqlite3.connect(job.database.as_uri() + "?mode=ro", uri=True, timeout=0)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA busy_timeout=1000")
        db.execute("BEGIN")
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("unsupported pilot schema")
        records = tuple(SQLiteOpportunityRepository._load(db, row) for row in
                        db.execute("SELECT * FROM opportunities ORDER BY created_at,id"))
        if {r.id for r in records} != expected_ids:
            raise ValueError("database parent scope differs from original captures")
        entries = []
        for record in records:
            reference = references[record.symbol]
            evidence = decode_entry(record)
            if (reference["opportunity_id"] != record.id
                    or record.config_version != research.config_version + ":next_settlement"
                    or _sha(record.config_json.encode("utf-8")) != record.config_fingerprint
                    or not opened <= evidence.opened_at <= record.created_at <= closed
                    or evidence.opened_at != utc_timestamp(datetime.fromisoformat(reference["entry_time"]))
                    or record.decision != reference["decision"]
                    or any(not isinstance(reference[name], str) or Decimal(reference[name]) != stored
                           for name, stored in (("net_profit", record.net_profit),
                                                ("net_return", record.net_return)))
                    or evidence.spot.filled_quantity != record.matched_quantity
                    or evidence.perp.filled_quantity != record.matched_quantity):
                raise ValueError("entry reference differs from original saved record")
            assert record.matched_quantity is not None
            assert record.capital_occupied is not None and record.net_profit is not None and record.net_return is not None
            entries.append(PilotEntry(record.symbol, record.id, record.created_at,
                evidence.opened_at, record.matched_quantity, (
                    ("capital_occupied", record.capital_occupied),
                    ("original_entry_basis", evidence.entry_basis),
                    ("spot_entry_fee", evidence.spot_fee), ("perp_entry_fee", evidence.perp_fee),
                    ("other_costs", evidence.other_costs),
                    ("original_estimated_net_profit", record.net_profit),
                    ("original_estimated_net_return", record.net_return))))
        tasks = tuple(_task(row) for row in db.execute("SELECT * FROM validation_tasks ORDER BY id"))
        validations = tuple(_decode(row) for row in db.execute("SELECT * FROM forward_validations ORDER BY id"))
    report = assess_pilot_v1(tuple(entries), tasks, validations, evaluated_at=at, deadline=job.deadline)
    report.update({
        "schema_version": "pilot-outcome-report-v1", "credentials_read": False,
        "database": str(job.database), "entry_evidence": str(entry_evidence.resolve()),
        "entry_evidence_sha256": _sha(raw),
        "binding_sha256": _sha(exact_json(job.binding).encode("utf-8")),
        "config_sha256": job.binding["config_sha256"],
        "original_record_sha256": {r.id: _sha(exact_json(r).encode("utf-8")) for r in records},
        "task_snapshot_sha256": _sha(exact_json(tasks).encode("utf-8")),
        "validation_snapshot_sha256": _sha(exact_json(validations).encode("utf-8")),
        "report_source_sha256": {p.name: _sha(p.read_bytes()) for p in
                                 (Path(__file__), Path(__file__).parent / "analytics/pilot_outcome.py",
                                  Path(__file__).parent / "analytics/validation_evidence.py")},
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--entry-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("report already exists")
        report = read_pilot_outcome(args.job_dir, args.entry_evidence, evaluated_at=utc_now())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(exact_json(report) + "\n")
        print(exact_json({name: report[name] for name in
                         ("phase", "conclusion", "provisional_conclusion", "usable_validations",
                          "task_status_counts", "strategy_qualified", "orders_submitted", "credentials_read")}
                        | {"evidence": str(args.output.resolve())}))
        return 0
    except (ControlError, OSError, sqlite3.Error, ValueError, TypeError, KeyError,
            DecimalException, OverflowError, RecursionError) as error:
        print(exact_json({"status": "UNAVAILABLE", "error_type": type(error).__name__,
                          "orders_submitted": False, "credentials_read": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
