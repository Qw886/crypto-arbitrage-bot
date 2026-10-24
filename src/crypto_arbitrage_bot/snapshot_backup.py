"""Bounded online SQLite snapshots and isolated recovery rehearsal.

Existing databases are opened read-only. No credentials, orders, original-file
replacement, worker start, study replay or budget reset. Separate databases have
separate transaction snapshots; this is not a globally atomic project image.
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any

from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

MIB = 1024**2
RESEARCH = frozenset({"opportunities", "opportunity_sources", "opportunity_scenarios",
                      "forward_validations", "validation_tasks"})
EXECUTION = frozenset({"testnet_pair_attempts", "testnet_order_intents", "testnet_abort_balance_proofs",
    "testnet_balance_audits", "testnet_balance_checks", "testnet_balance_frame_intents",
    "testnet_exit_recoveries", "testnet_funding_checks", "testnet_funding_payments",
    "testnet_spot_exit_recoveries", "testnet_trade_evidence", "testnet_trade_reconciliations"})


def _check_time(deadline: float) -> None:
    if monotonic() >= deadline:
        raise TimeoutError("SNAPSHOT_TIME_LIMIT")


def _limits(max_bytes: int, min_free_bytes: int, timeout_seconds: int) -> float:
    if (type(max_bytes) is not int or not 1 <= max_bytes <= 4096*MIB
            or type(min_free_bytes) is not int or not 0 <= min_free_bytes <= 4096*MIB
            or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300):
        raise ValueError("FINITE_SNAPSHOT_LIMITS_REQUIRED")
    return monotonic()+timeout_seconds


def _capacity(path: Path, required: int, reserve: int) -> None:
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required+reserve:
        raise OSError("INSUFFICIENT_SNAPSHOT_CAPACITY")


def _digest(path: Path, deadline: float) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            _check_time(deadline)
            value.update(chunk)
    return value.hexdigest()


def _relative(value: str) -> Path:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path) != value or ".." in path.parts
            or "\\" in value or ":" in value or path.suffix != ".sqlite"
            or any("credential" in part.casefold() or "secret" in part.casefold() for part in path.parts)
            or len(path.parts) < 3 or path.parts[:2] not in (("data", "research"), ("data", "testnet"))):
        raise ValueError("SCOPED_DATABASE_PATH_REQUIRED")
    return Path(*path.parts)


def _new_target(root: Path, target: Path) -> Path:
    root, target = root.resolve(), target.resolve()
    if (not root.is_dir() or not target.is_relative_to(root) or target == root
            or target.relative_to(root).parts[0] not in ("work", "backups") or target.exists()):
        raise ValueError("NEW_ISOLATED_TARGET_REQUIRED")
    return target


def _schema(db: sqlite3.Connection) -> dict[str, Any]:
    rows = db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
    tables = {r[1] for r in rows if r[0] == "table"}
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if ({"opportunities", "opportunity_sources", "opportunity_scenarios"} <= tables <= RESEARCH
            and version == 1):
        kind = "RESEARCH"
    elif {"testnet_pair_attempts", "testnet_order_intents"} <= tables <= EXECUTION and version == 0:
        kind = "TESTNET_EXECUTION_ARCHIVE"
    else:
        raise ValueError("KNOWN_PROJECT_DATABASE_REQUIRED")
    return {"kind": kind, "user_version": version,
            "application_id": db.execute("PRAGMA application_id").fetchone()[0],
            "schema_sha256": hashlib.sha256(exact_json(rows).encode()).hexdigest(),
            "table_names": sorted(tables)}


def _audit(path: Path, deadline: float) -> dict[str, Any]:
    # Only a completed independent snapshot uses immutable=1. Never apply it
    # to a live source, where doing so would ignore committed WAL contents.
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True, timeout=0)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        db.set_progress_handler(lambda: int(monotonic() >= deadline), 1000)
        result = _schema(db)
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("SNAPSHOT_INTEGRITY_FAILED")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SNAPSHOT_FOREIGN_KEYS_FAILED")
        result["row_counts"] = {name: db.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                                for name in result.pop("table_names")}
    _check_time(deadline)
    return result


def _write(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(exact_json(value).encode())
        stream.flush()
        os.fsync(stream.fileno())


def create(root: Path, sources: tuple[Path, ...], output: Path, *, max_bytes: int = 2048*MIB,
           min_free_bytes: int = 512*MIB, timeout_seconds: int = 120) -> dict[str, Any]:
    deadline = _limits(max_bytes, min_free_bytes, timeout_seconds)
    root = root.resolve()
    output = _new_target(root, output)
    if not isinstance(sources, tuple) or not 1 <= len(sources) <= 16:
        raise ValueError("EXPLICIT_DATABASE_LIST_REQUIRED")
    resolved = tuple(p.resolve() for p in sources)
    if len(set(resolved)) != len(resolved):
        raise ValueError("DUPLICATE_DATABASE")
    for path in resolved:
        _relative(path.relative_to(root).as_posix())
        if not path.is_file():
            raise ValueError("EXISTING_DATABASE_REQUIRED")
    estimate = sum(path.stat().st_size for path in resolved)
    if estimate > max_bytes:
        raise ValueError("SNAPSHOT_BYTE_LIMIT")
    _capacity(output, estimate, min_free_bytes)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    total = 0
    try:
        for index, source in enumerate(resolved):
            _check_time(deadline)
            started = utc_now()
            snapshot = output / f"database-{index:02d}.sqlite"
            with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True, timeout=0)) as db:
                db.execute("PRAGMA query_only=ON")
                db.execute("PRAGMA trusted_schema=OFF")
                db.set_progress_handler(lambda: int(monotonic() >= deadline), 1000)
                db.execute("BEGIN")
                schema = _schema(db)  # Pin this source's read transaction, including its WAL.
                page_size = db.execute("PRAGMA page_size").fetchone()[0]
                size = page_size*db.execute("PRAGMA page_count").fetchone()[0]
                if total+size > max_bytes:
                    raise ValueError("SNAPSHOT_BYTE_LIMIT")
                _capacity(output, size, min_free_bytes)
                with snapshot.open("xb"):
                    pass
                with closing(sqlite3.connect(snapshot.as_uri()+"?mode=rw", uri=True, timeout=0)) as dest:
                    def progress(status: int, remaining: int, pages: int, *,
                                 completed_bytes: int = total, page_bytes: int = page_size) -> None:
                        _check_time(deadline)
                        if completed_bytes+pages*page_bytes > max_bytes:
                            raise ValueError("SNAPSHOT_BYTE_LIMIT")
                        _capacity(output, remaining*page_bytes, min_free_bytes)
                    db.backup(dest, pages=128, progress=progress, sleep=.01)
                    # Standalone snapshot, no WAL/SHM dependency on recovery.
                    if dest.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                        raise ValueError("STANDALONE_SNAPSHOT_REQUIRED")
            finished = utc_now()
            if finished < started:
                raise ValueError("SNAPSHOT_CLOCK_ROLLBACK")
            # Windows _commit/fsync requires a writable descriptor. This is
            # exclusively the newly created destination, never the source.
            with snapshot.open("r+b") as stream:
                os.fsync(stream.fileno())
            audit = _audit(snapshot, deadline)
            if {k: v for k, v in audit.items() if k != "row_counts"} != {
                    k: v for k, v in schema.items() if k != "table_names"}:
                raise ValueError("SOURCE_SNAPSHOT_SCHEMA_DIFFERS")
            total += snapshot.stat().st_size
            if total > max_bytes:
                raise ValueError("SNAPSHOT_BYTE_LIMIT")
            entries.append({"source_relative_path": source.relative_to(root).as_posix(),
                            "snapshot_file": snapshot.name, "bytes": snapshot.stat().st_size,
                            "sha256": _digest(snapshot, deadline), "snapshot_started_at": started,
                            "snapshot_finished_at": finished, "audit": audit})
        manifest = {"schema_version": 1, "scope": "ISOLATED_DATABASE_SNAPSHOTS", "created_at": utc_now(),
                    "global_atomic_snapshot": False, "source_project_directory": str(root),
                    "databases": entries, "total_bytes": total, "max_bytes": max_bytes,
                    "min_free_bytes": min_free_bytes, "credentials_included": False,
                    "orders_submitted": False, "workers_restarted": False}
        _check_time(deadline)
        _write(output / "manifest.json", manifest)
        return {"status": "SNAPSHOT_COMPLETE", "output_directory": str(output),
                "manifest_sha256": hashlib.sha256(exact_json(manifest).encode()).hexdigest(),
                "databases": len(entries), "total_bytes": total, "orders_submitted": False}
    except BaseException as error:
        if not (output / "manifest.json").exists():
            _write(output / "failure.json", {"status": "INCOMPLETE", "error_type": type(error).__name__,
                                           "orders_submitted": False})
        raise


def verify(directory: Path, sha: str, *, timeout_seconds: int = 120) -> dict[str, Any]:
    deadline = _limits(4096*MIB, 0, timeout_seconds)
    directory = directory.resolve()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("PINNED_MANIFEST_HASH_REQUIRED")
    path = directory / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 256000 or _digest(path, deadline) != sha:
        raise ValueError("MANIFEST_HASH_DIFFERS")
    body = path.read_bytes()
    manifest = json.loads(body)
    if (not isinstance(manifest, dict) or exact_json(manifest).encode() != body
            or set(manifest) != {"schema_version", "scope", "created_at", "global_atomic_snapshot",
                "source_project_directory", "databases", "total_bytes", "max_bytes", "min_free_bytes",
                "credentials_included", "orders_submitted", "workers_restarted"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["scope"] != "ISOLATED_DATABASE_SNAPSHOTS"
            or any(manifest[k] is not False for k in ("global_atomic_snapshot", "credentials_included",
                                                     "orders_submitted", "workers_restarted"))
            or not isinstance(manifest["databases"], list) or not 1 <= len(manifest["databases"]) <= 16):
        raise ValueError("INVALID_SNAPSHOT_MANIFEST")
    expected = {"manifest.json"} | {f"database-{i:02d}.sqlite" for i in range(len(manifest["databases"]))}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError("UNEXPECTED_OR_MISSING_SNAPSHOT_FILE")
    created = utc_timestamp(datetime.fromisoformat(manifest["created_at"]))
    if created > utc_now():
        raise ValueError("SNAPSHOT_TIME_IN_FUTURE")
    _limits(manifest["max_bytes"], manifest["min_free_bytes"], timeout_seconds)
    names: set[str] = set()
    total = 0
    for index, entry in enumerate(manifest["databases"]):
        if (set(entry) != {"source_relative_path", "snapshot_file", "bytes", "sha256", "snapshot_started_at",
                          "snapshot_finished_at", "audit"} or entry["snapshot_file"] != f"database-{index:02d}.sqlite"
                or type(entry["bytes"]) is not int or not 1 <= entry["bytes"] <= 4096*MIB):
            raise ValueError("INVALID_SNAPSHOT_ENTRY")
        _relative(entry["source_relative_path"])
        started = utc_timestamp(datetime.fromisoformat(entry["snapshot_started_at"]))
        finished = utc_timestamp(datetime.fromisoformat(entry["snapshot_finished_at"]))
        if not started <= finished <= created:
            raise ValueError("SNAPSHOT_TIME_ORDER_DIFFERS")
        if entry["source_relative_path"] in names:
            raise ValueError("DUPLICATE_SNAPSHOT_SOURCE")
        names.add(entry["source_relative_path"])
        snapshot = directory / entry["snapshot_file"]
        if (snapshot.is_symlink() or snapshot.resolve().parent != directory
                or snapshot.stat().st_size != entry["bytes"] or _digest(snapshot, deadline) != entry["sha256"]
                or _audit(snapshot, deadline) != entry["audit"]):
            raise ValueError("SNAPSHOT_CONTENT_DIFFERS")
        total += entry["bytes"]
    if (type(manifest["total_bytes"]) is not int or manifest["total_bytes"] != total
            or type(manifest["max_bytes"]) is not int or not total <= manifest["max_bytes"] <= 4096*MIB):
        raise ValueError("SNAPSHOT_TOTAL_DIFFERS")
    return manifest


def rehearse(root: Path, directory: Path, sha: str, output: Path, *,
             min_free_bytes: int = 512*MIB, timeout_seconds: int = 120) -> dict[str, Any]:
    deadline = _limits(4096*MIB, min_free_bytes, timeout_seconds)
    output = _new_target(root, output)
    directory = directory.resolve()
    if output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("SEPARATE_REHEARSAL_REQUIRED")
    manifest = verify(directory, sha, timeout_seconds=timeout_seconds)
    _capacity(output, manifest["total_bytes"], min_free_bytes)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for entry in manifest["databases"]:
            source = directory / entry["snapshot_file"]
            target = output / _relative(entry["source_relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(output):
                raise ValueError("REHEARSAL_PATH_ESCAPES")
            with source.open("rb") as original, target.open("xb") as restored:
                for chunk in iter(lambda: original.read(MIB), b""):
                    _check_time(deadline)
                    restored.write(chunk)
                restored.flush()
                os.fsync(restored.fileno())
            if (_digest(target, deadline) != entry["sha256"] or _audit(target, deadline) != entry["audit"]):
                raise ValueError("REHEARSAL_CONTENT_DIFFERS")
        result = {"status": "ISOLATED_REHEARSAL_VERIFIED", "manifest_sha256": sha,
                  "output_directory": str(output), "databases": len(manifest["databases"]),
                  "completed_at": utc_now(), "original_database_replaced": False,
                  "workers_started": False, "orders_submitted": False,
                  "original_budget_and_record_values_preserved": True,
                  "live_recovery_authorized": False}
        _write(output / "rehearsal.json", result)
        return result
    except BaseException as error:
        _write(output / "failure.json", {"status": "INCOMPLETE_REHEARSAL", "error_type": type(error).__name__})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consistent database backup and isolated recovery rehearsal")
    parser.add_argument("command", choices=("create", "verify", "rehearse"))
    parser.add_argument("--database", action="append", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-mib", type=int, default=2048)
    parser.add_argument("--min-free-mib", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            if not args.database or args.output_dir is None or args.backup_dir or args.manifest_sha256:
                parser.error("create requires explicit databases and a new output directory")
            result = create(Path.cwd(), tuple(args.database), args.output_dir, max_bytes=args.max_mib*MIB,
                min_free_bytes=args.min_free_mib*MIB, timeout_seconds=args.timeout_seconds)
        else:
            if args.database or args.backup_dir is None or args.manifest_sha256 is None:
                parser.error("verify/rehearse requires a backup directory and original manifest hash")
            if args.command == "verify":
                if args.output_dir is not None:
                    parser.error("verify is read only")
                manifest = verify(args.backup_dir, args.manifest_sha256, timeout_seconds=args.timeout_seconds)
                result = {"status": "SNAPSHOT_VERIFIED", "databases": len(manifest["databases"]),
                          "manifest_sha256": args.manifest_sha256, "orders_submitted": False}
            else:
                if args.output_dir is None:
                    parser.error("isolated rehearsal requires a new output directory")
                result = rehearse(Path.cwd(), args.backup_dir, args.manifest_sha256, args.output_dir,
                    min_free_bytes=args.min_free_mib*MIB, timeout_seconds=args.timeout_seconds)
        print(exact_json(result))
        return 0
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        print(exact_json({"status": "UNAVAILABLE", "error_type": type(error).__name__,
                          "orders_submitted": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
