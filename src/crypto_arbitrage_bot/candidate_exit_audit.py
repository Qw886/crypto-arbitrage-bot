"""Bounded raw-public-exit audit, using the original validator and no writes."""

import gzip
import hashlib
import json
import re
import sqlite3
import zlib
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.analytics.public_exit_evidence import (
    check_catalogs,
    decode_funding,
    decode_market,
    time,
)
from crypto_arbitrage_bot.analytics.validation_evidence import read_validation_attribution
from crypto_arbitrage_bot.candidate_diagnostics import _summarize, _unique
from crypto_arbitrage_bot.candidate_study import MAX_RAW_ARCHIVE_BYTES, load_index, planned_horizon
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.storage.forward_validation_records import build_forward_validation_record
from crypto_arbitrage_bot.storage.forward_validation_repository import _decode
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteOpportunityRepository
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

_FILENAME = re.compile(r"[0-9a-f]{32}\.json(?:\.gz)?\Z")


@dataclass(frozen=True)
class ExitAuditLimits:
    max_files: int = 5000
    max_records: int = 10000
    max_total_bytes: int = 512 * 1024 * 1024
    max_seconds: int = 60

    def __post_init__(self) -> None:
        for number, maximum in (
            (self.max_files, 10000),
            (self.max_records, 30000),
            (self.max_total_bytes, 4 * 1024 * 1024 * 1024),
            (self.max_seconds, 120),
        ):
            if type(number) is not int or not 1 <= number <= maximum:
                raise ValueError("BOUNDED_EXIT_AUDIT_LIMITS_REQUIRED")


def audit_candidate_exits(
    settings: Settings, *, evaluated_at: datetime, limits: ExitAuditLimits | None = None
) -> dict[str, Any]:
    at = utc_timestamp(evaluated_at)
    limits = ExitAuditLimits() if limits is None else limits
    index, studies = load_index(settings)
    research = settings.research
    assert research is not None and research.study_registration is not None
    assert research.candidate_catalog is not None and research.validation_scheduler is not None
    end = monotonic() + limits.max_seconds

    def check() -> None:
        if monotonic() > end:
            raise ValueError("EXIT_AUDIT_TIME_LIMIT")

    with closing(
        sqlite3.connect(research.sqlite_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    ) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.set_progress_handler(lambda: int(monotonic() > end), 1000)
        db.execute("BEGIN")
        rows = db.execute(
            "SELECT * FROM validation_tasks LIMIT ?", (limits.max_records + 1,)
        ).fetchall()
        values = db.execute(
            "SELECT * FROM forward_validations LIMIT ?", (limits.max_records + 1,)
        ).fetchall()
        if max(len(rows), len(values)) > limits.max_records:
            raise ValueError("EXIT_AUDIT_RECORD_LIMIT")
        tasks = {row["id"]: dict(row) for row in rows}
        validations = {row["id"]: _decode(row) for row in values}
        records = {}
        for task in tasks.values():
            check()
            parent = db.execute(
                "SELECT * FROM opportunities WHERE id=?", (task["opportunity_id"],)
            ).fetchone()
            if (
                parent is None
                or parent["symbol"] != task["symbol"]
                or task["symbol"] not in studies
            ):
                raise ValueError("EXIT_AUDIT_TASK_SCOPE_DIFFERS")
            records[task["opportunity_id"]] = SQLiteOpportunityRepository._load(db, parent)
    by_validation: dict[str, list[dict[str, Any]]] = {}
    for task in tasks.values():
        by_validation.setdefault(task["validation_id"], []).append(task)
    refs: dict[str, dict[str, list[dict[str, str]]]] = {}
    issues = []
    total = 0
    reference_ok = False
    reference = research.study_registration.path.parent / "reference-catalog.json"
    try:
        if (
            reference.is_symlink()
            or reference.resolve().parent != research.study_registration.path.parent.resolve()
        ):
            raise ValueError("UNSAFE_REFERENCE_CATALOG")
        digest = hashlib.sha256()
        reference_size = 0
        with reference.open("rb") as stream:
            while True:
                check()
                body = stream.read(min(1024 * 1024, max(0, limits.max_total_bytes - total)) + 1)
                if not body:
                    break
                total += len(body)
                reference_size += len(body)
                if total > limits.max_total_bytes or reference_size > MAX_RAW_ARCHIVE_BYTES:
                    raise ValueError("EXIT_AUDIT_REFERENCE_BYTES_LIMIT")
                digest.update(body)
        if digest.hexdigest() != index["catalog_sha256"]:
            raise ValueError("ORIGINAL_REFERENCE_CATALOG_HASH_DIFFERS")
        reference_ok = True
    except (OSError, ValueError) as error:
        issues.append(
            {"code": "ORIGINAL_REFERENCE_CATALOG_UNAVAILABLE", "error_type": type(error).__name__}
        )

    def read_archive(path: Path) -> tuple[dict[str, Any], str]:
        nonlocal total
        check()
        if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
            raise ValueError("UNSAFE_EXIT_ARCHIVE")
        remaining = limits.max_total_bytes - total
        if remaining <= 0:
            raise ValueError("EXIT_AUDIT_TOTAL_BYTES_LIMIT")
        stream = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
        with stream:
            body = stream.read(min(remaining, MAX_RAW_ARCHIVE_BYTES) + 1)
        total += len(body)
        if len(body) > MAX_RAW_ARCHIVE_BYTES or total > limits.max_total_bytes:
            raise ValueError("EXIT_AUDIT_ARCHIVE_BYTES_LIMIT")
        check()
        value = json.loads(body, object_pairs_hook=_unique)
        if not isinstance(value, dict) or exact_json(value).encode() != body:
            raise ValueError("CANONICAL_EXIT_ARCHIVE_REQUIRED")
        return value, hashlib.sha256(body).hexdigest()

    count = 0
    directory = research.study_registration.path.parent / "exit-evidence"
    if directory.exists():
        for path in directory.iterdir():
            check()
            count += 1
            if count > limits.max_files:
                issues.append({"code": "EXIT_AUDIT_FILE_LIMIT"})
                break
            if (
                not _FILENAME.fullmatch(path.name)
                or path.is_symlink()
                or not path.is_file()
                or path.resolve().parent != directory.resolve()
            ):
                issues.append({"code": "UNKNOWN_OR_UNSAFE_ARCHIVE"})
                continue
            try:
                value, sha = read_archive(path)
                archive_task = tasks.get(value["task_id"])
                if archive_task is None:
                    raise ValueError("UNKNOWN_EXIT_TASK")
                _summarize(value, archive_task, studies[archive_task["symbol"]].sha256, at)
                refs.setdefault(archive_task["id"], {}).setdefault(value["phase"], []).append(
                    {"filename": path.name, "sha256": sha}
                )
            except (
                OSError,
                EOFError,
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                zlib.error,
            ) as error:
                issues.append(
                    {
                        "code": "CORRUPT_OR_UNBOUND_ARCHIVE",
                        "filename": path.name,
                        "error_type": type(error).__name__,
                    }
                )
                if total > limits.max_total_bytes:
                    break
    identifiers = sorted(
        set(validations)
        | {t["validation_id"] for t in tasks.values() if t["status"] == "COMPLETED"}
    )
    items = []
    for identifier in identifiers:
        check()
        item: dict[str, Any] = {
            "validation_id": identifier,
            "audit_status": "UNAVAILABLE",
            "recomputed_net_profit": None,
            "recomputed_net_return": None,
            "issue_codes": [],
        }
        try:
            saved = validations[identifier]
            item.update(
                symbol=saved.symbol, opportunity_id=saved.opportunity_id, stored_status=saved.status
            )
            if saved.status != "VALIDATED":
                item["audit_status"] = "NOT_VALIDATED"
                items.append(item)
                continue
            if not reference_ok:
                raise ValueError("ORIGINAL_REFERENCE_CATALOG_UNAVAILABLE")
            candidates = by_validation.get(identifier, [])
            if len(candidates) != 1:
                raise ValueError("UNIQUE_VALIDATION_TASK_REQUIRED")
            task = candidates[0]
            if (
                task["status"] != "COMPLETED"
                or task["failure_code"] is not None
                or task["symbol"] != saved.symbol
                or task["opportunity_id"] != saved.opportunity_id
                or task["id"] != saved.id
            ):
                raise ValueError("COMPLETED_VALIDATION_TASK_REQUIRED")
            record = records[saved.opportunity_id]
            study = studies[record.symbol]
            protocol = study.protocol
            if (
                record.config_version != protocol.config_version
                or record.formula_version != protocol.formula_version
                or record.config_fingerprint != dict(protocol.config_fingerprints)[record.symbol]
                or hashlib.sha256(record.config_json.encode()).hexdigest()
                != record.config_fingerprint
                or not protocol.study_start <= record.created_at < protocol.study_end
            ):
                raise ValueError("ORIGINAL_REGISTERED_ENTRY_REQUIRED")
            entry = decode_entry(record)
            # Read only this parent's original receipt, with finite IO; never
            # create a missing receipt after learning the exit outcome.
            receipt_path = study.receipt_directory / (
                hashlib.sha256(record.id.encode()).hexdigest() + ".json"
            )
            if (
                receipt_path.is_symlink()
                or receipt_path.resolve().parent != study.receipt_directory.resolve()
            ):
                raise ValueError("UNSAFE_ORIGINAL_CAPTURE_RECEIPT")
            with receipt_path.open("rb") as stream:
                receipt_body = stream.read(256001)
            total += len(receipt_body)
            if len(receipt_body) > 256000 or total > limits.max_total_bytes:
                raise ValueError("EXIT_AUDIT_RECEIPT_BYTES_LIMIT")
            receipt = json.loads(receipt_body, object_pairs_hook=_unique)
            captured = time(receipt["captured_at"])
            if (
                exact_json(receipt).encode() != receipt_body
                or receipt["schema_version"] != 1
                or receipt["registration_sha256"] != study.sha256
                or receipt["opportunity_id"] != record.id
                or receipt["record_sha256"]
                != hashlib.sha256(exact_json(record).encode()).hexdigest()
                or receipt["policy_sha256"]
                != hashlib.sha256(study.policy_json.encode()).hexdigest()
                or not study.registered_at
                <= record.created_at
                <= captured
                <= min(at, record.created_at + study.max_capture_delay)
            ):
                raise ValueError("ORIGINAL_CAPTURE_RECEIPT_DIFFERS")
            if entry.research_instrument is None or saved.validation_time is None:
                raise ValueError("REGISTERED_EXIT_IDENTITY_AND_TIME_REQUIRED")
            planned = time(json.loads(record.details_json)["estimate"]["scenario"]["closed_at"])
            if (
                task["horizon_seconds"] != planned_horizon(record)
                or time(task["due_at"])
                != record.created_at + timedelta(seconds=task["horizon_seconds"])
                or not planned <= saved.validation_time <= planned + protocol.max_exit_lateness
                or not saved.validation_time <= saved.created_at <= at
            ):
                raise ValueError("ORIGINAL_EXIT_TIME_OR_HORIZON_DIFFERS")
            phases = refs.get(task["id"], {})
            if set(phases) != {"EXIT_MARKET", "ACTUAL_FUNDING"} or any(
                len(v) != 1 for v in phases.values()
            ):
                raise ValueError("UNIQUE_COMPLETE_EXIT_ARCHIVES_REQUIRED")
            packs = {}
            for phase, files in phases.items():
                value, sha = read_archive(directory / files[0]["filename"])
                if sha != files[0]["sha256"]:
                    raise ValueError("EXIT_ARCHIVE_CHANGED_DURING_AUDIT")
                _summarize(value, task, study.sha256, at)
                if len(value["captures"]) != 1:
                    raise ValueError("ONE_PHASE_CAPTURE_REQUIRED")
                packs[phase] = value
            context = json.loads(study.candidate_context_json or "null")
            market = packs["EXIT_MARKET"]
            funding = packs["ACTUAL_FUNDING"]
            if (
                not time(market["recorded_at"])
                <= saved.validation_time
                <= time(funding["recorded_at"])
                <= saved.created_at
            ):
                raise ValueError("EXIT_ARCHIVE_PUBLICATION_TIMING_DIFFERS")
            if not market["catalogs"]:
                raise ValueError("EXIT_METADATA_REQUIRED")
            for pack in (market, funding):
                check_catalogs(pack["catalogs"], research.candidate_catalog, context["candidate"])
            observation = market["captures"][0]["observation"]
            if exact_json(observation["candidate"]) != exact_json(context["candidate"]):
                raise ValueError("FROZEN_EXIT_CANDIDATE_DIFFERS")
            assembly = decode_market(observation, entry.research_instrument)
            if exact_json(
                validate_market_data(
                    assembly,
                    entry.quality_config,
                    evaluated_at=time(observation["data_quality"]["evaluated_at"]),
                )
            ) != exact_json(observation["data_quality"]):
                raise ValueError("RAW_CAPTURE_QUALITY_DIFFERS")
            events = decode_funding(
                funding["captures"][0]["window"],
                symbol=record.symbol,
                opened_at=entry.opened_at,
                closed_at=saved.validation_time,
                criteria=research.validation_scheduler,
            )
            expected = record.next_funding_time
            if expected is None:
                raise ValueError("KNOWN_FUNDING_TIME_REQUIRED")
            if entry.opened_at < expected <= saved.validation_time:
                tolerance = timedelta(
                    milliseconds=research.validation_scheduler.funding_time_tolerance_ms
                )
                if (
                    len(
                        [event for event in events if abs(event.settled_at - expected) <= tolerance]
                    )
                    != 1
                ):
                    raise ValueError("KNOWN_SETTLEMENT_ABSENT_OR_AMBIGUOUS")
            recomputed = validate_forward_outcome(
                record, assembly, validation_time=saved.validation_time, funding_events=events
            )
            rebuilt = build_forward_validation_record(
                recomputed, validation_id=saved.id, created_at=saved.created_at
            )
            read_validation_attribution(saved)
            if recomputed.status != "VALIDATED" or exact_json(rebuilt) != exact_json(saved):
                raise ValueError("ORIGINAL_VALIDATOR_PROJECTION_DIFFERS")
            item.update(
                audit_status="VERIFIED",
                recomputed_net_profit=rebuilt.validated_simulated_net_profit,
                recomputed_net_return=rebuilt.validated_simulated_net_return,
                original_decision=record.decision,
                parent_sha256=hashlib.sha256(exact_json(record).encode()).hexdigest(),
                funding_event_count=len(events),
                archives=phases,
            )
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RecursionError,
            zlib.error,
        ) as error:
            code = (
                str(error)
                if isinstance(error, ValueError)
                and re.fullmatch(r"[A-Z][A-Z0-9_]{1,79}", str(error))
                else "EXIT_EVIDENCE_UNAVAILABLE"
            )
            item["issue_codes"] = [code]
            item["error_type"] = type(error).__name__
        items.append(item)
    check()
    verified = [item for item in items if item["audit_status"] == "VERIFIED"]
    complete = (
        bool(verified)
        and not issues
        and not any(item["audit_status"] == "UNAVAILABLE" for item in items)
    )
    return {
        "schema_version": "registered-candidate-exit-audit-v1",
        "evaluated_at": at,
        "status": "AUDITED_VALIDATED_EXITS"
        if complete
        else "INCOMPLETE"
        if items or issues
        else "NO_VALIDATED_EXITS",
        "meaning": "Original validator recomputed from archived public bytes; local simulated exits, no external authentication or trading permission",
        "index_sha256": research.study_registration.sha256,
        "original_catalog_sha256_verified": reference_ok,
        "symbols": index["symbols"],
        "task_states": dict(Counter(task["status"] for task in tasks.values())),
        "saved_validation_states": dict(Counter(row.status for row in validations.values())),
        "verified_exits": len(verified),
        "all_audited_profits_negative": all(item["recomputed_net_profit"] < 0 for item in verified)
        if complete
        else None,
        "items": items,
        "archive_issues": issues,
        "scan": {
            "files_observed": count,
            "decompressed_bytes_including_reread": total,
            "limits": limits,
        },
        "credentials_read": False,
        "orders_submitted": False,
        "trade_allowed": False,
        "qualification_issued": False,
        "realized_trade_profit": False,
        "external_authentication_proved": False,
        "original_financial_engine": True,
    }
