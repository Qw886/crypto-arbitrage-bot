"""Complete registered SQLite cohort in one read-only transaction, no PnL math."""

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import DecimalException
from enum import StrEnum
from pathlib import Path

from crypto_arbitrage_bot.analytics.strategy_study import assess_strategy_study
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.strategy_study import (
    StrategyStudyAssessment,
    StudyIssue,
    StudyReason,
    StudyStatus,
)
from crypto_arbitrage_bot.storage.forward_validation_repository import _decode
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SCHEMA_VERSION,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.storage.study_attempts import AttemptCoverage, read_attempt_coverage
from crypto_arbitrage_bot.storage.study_registry import (
    REGISTRY_VERSION,
    RegisteredStudy,
    StudyArchiveError,
    StudyArchiveFailure,
    load_registered_study,
)
from crypto_arbitrage_bot.storage.validation_tasks import _task
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class StudyReadStatus(StrEnum):
    READ = "READ"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StudyTaskEvidence:
    id: str
    opportunity_id: str
    horizon_seconds: int
    status: str
    validation_id: str
    failure_code: str | None
    due_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class RegisteredStudyEvidence:
    status: StudyReadStatus
    registration_sha256: str
    assessment: StrategyStudyAssessment | None = None
    opportunity_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    tasks: tuple[StudyTaskEvidence, ...] = ()
    failure: StudyArchiveFailure | None = None
    sqlite_errorcode: int | None = None
    attempt_coverage: AttemptCoverage | None = None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _capture_issues(study: RegisteredStudy, records: tuple[OpportunityRecord, ...],
                    evaluated_at: datetime) -> tuple[StudyIssue, ...]:
    issues = []
    receipts = {}
    directory = study.receipt_directory
    if directory.exists():
        for path in sorted(directory.iterdir()):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                identifier = value["opportunity_id"]
                if (not isinstance(identifier, str) or path.name != _digest(identifier) + ".json"
                        or identifier in receipts or value["schema_version"] != REGISTRY_VERSION):
                    raise ValueError("invalid capture identity")
                receipts[identifier] = value
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_CORRUPT) from exc
    for record in records:
        receipt = receipts.pop(record.id, None)
        if receipt is None:
            issues.append(StudyIssue(StudyReason.MISSING_CAPTURE_RECEIPT, record.symbol, record.id))
            continue
        try:
            at = utc_timestamp(datetime.fromisoformat(receipt["captured_at"]))
            if (receipt["registration_sha256"] != study.sha256
                    or receipt["record_sha256"] != _digest(exact_json(record))
                    or receipt["policy_sha256"] != _digest(study.policy_json)
                    or not study.registered_at <= record.created_at <= at <= min(
                        record.created_at+study.max_capture_delay, evaluated_at)):
                raise ValueError("record/policy/timing differs from capture")
        except (ValueError, TypeError, KeyError, OverflowError, DecimalException):
            issues.append(StudyIssue(StudyReason.CAPTURE_RECEIPT_MISMATCH, record.symbol, record.id))
    # A deleted/reassigned/retimed original must not disappear from the cohort.
    for identifier in sorted(receipts):
        issues.append(StudyIssue(StudyReason.ORPHAN_CAPTURE_RECEIPT, opportunity_id=identifier))
    return tuple(issues)


def read_registered_study(path: Path, expected_sha256: str, *, evaluated_at: datetime,
                          busy_timeout_ms: int = 1000) -> RegisteredStudyEvidence:
    """Never filter by decision, sign of PnL, config match, or validation success.

    Every saved parent in the preregistered symbol/time range must have an
    immediate receipt. Scheduled failure states remain in the audit, and no
    manual/different-horizon validation can replace the saved planned exit.
    V2 also requires started/finished attempt evidence and predeclared sampling
    coverage. Legacy V1 remains readable but cannot qualify without this evidence.
    It is evidence review only, never authorization for TESTNET orders.
    """
    try:
        if type(busy_timeout_ms) is not int or not 0 <= busy_timeout_ms <= 60000:
            raise ValueError("bounded SQLite timeout required")
        at = utc_timestamp(evaluated_at)
        study = load_registered_study(path, expected_sha256)
        p = study.protocol
        with closing(sqlite3.connect(study.database_path.as_uri()+"?mode=ro", uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            db.execute("BEGIN")
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise ValueError("unsupported opportunity schema")
            parents = {}
            selected = []
            # Python UTC decoding avoids assuming all old TEXT times share a
            # representation suitable for lexicographic SQL range comparisons.
            for row in db.execute("SELECT * FROM opportunities ORDER BY created_at,id"):
                parents[row["id"]] = row["symbol"]
                created = utc_timestamp(datetime.fromisoformat(row["created_at"]))
                if row["symbol"] in p.symbols and p.study_start <= created < p.study_end:
                    selected.append(SQLiteOpportunityRepository._load(db, row))
            selected.sort(key=lambda record: (record.created_at, record.id))
            records = tuple(selected)
            selected_ids = {r.id for r in records}
            validations = []
            all_validations = {}
            for row in db.execute("SELECT * FROM forward_validations ORDER BY id"):
                if parents.get(row["opportunity_id"]) != row["symbol"]:
                    raise ValueError("orphan/mismatched validation")
                all_validations[row["id"]] = (row["opportunity_id"], row["symbol"])
                if row["opportunity_id"] in selected_ids:
                    validations.append(_decode(row))
            tasks = []
            conflicts = []
            for row in db.execute("SELECT * FROM validation_tasks ORDER BY due_at,id"):
                if parents.get(row["opportunity_id"]) != row["symbol"]:
                    raise ValueError("orphan/mismatched task")
                if row["opportunity_id"] not in selected_ids:
                    continue
                task = _task(row)
                if (type(row["horizon_seconds"]) is not int or row["horizon_seconds"] <= 0
                        or row["status"] not in ("PENDING", "RUNNING", "COMPLETED", "FAILED")):
                    raise ValueError("invalid task state")
                identity = all_validations.get(row["validation_id"])
                if ((row["status"] == "COMPLETED" and identity != (row["opportunity_id"], row["symbol"]))
                        or (identity is not None and identity != (row["opportunity_id"], row["symbol"]))):
                    conflicts.append(StudyIssue(StudyReason.TASK_VALIDATION_CONFLICT,
                                                row["symbol"], row["opportunity_id"]))
                tasks.append(StudyTaskEvidence(row["id"], row["opportunity_id"], row["horizon_seconds"],
                                              row["status"], row["validation_id"], row["failure_code"],
                                              task.due_at, task.updated_at))
            receipt_issues = _capture_issues(study, records, at)
            coverage = read_attempt_coverage(study, records, evaluated_at=at)
        # Replace the declared date with the registry's actual local receipt date;
        # declaring an earlier date never legitimizes an already captured book.
        assessment = assess_strategy_study(replace(p, registered_at=study.registered_at), records,
                                           tuple(validations), evaluated_at=at)
        planned_tasks = {task.validation_id: task for task in tasks if task.status == "COMPLETED"}
        by_record = {record.id: record for record in records}
        by_validation = {validation.id: validation for validation in validations}
        for symbol_evidence in assessment.symbols:
            for identifier in symbol_evidence.selected_validation_ids:
                validation = by_validation[identifier]
                record = by_record[validation.opportunity_id]
                closed = utc_timestamp(datetime.fromisoformat(json.loads(record.details_json)["estimate"]["scenario"]["closed_at"]))
                completed = planned_tasks.get(identifier)
                if study.candidate_context_json is not None and completed is not None:
                    from crypto_arbitrage_bot.candidate_study import planned_horizon
                    if completed.horizon_seconds != planned_horizon(record):
                        conflicts.append(StudyIssue(StudyReason.TASK_VALIDATION_CONFLICT, record.symbol, record.id))
                if (completed is None or completed.opportunity_id != record.id
                        or completed.due_at != record.created_at + timedelta(seconds=completed.horizon_seconds)
                        or not closed <= completed.due_at <= closed+p.max_exit_lateness
                        or validation.validation_time is None or validation.validation_time < completed.due_at
                        or not validation.validation_time <= validation.created_at <= completed.updated_at <= at):
                    conflicts.append(StudyIssue(StudyReason.PLANNED_TASK_NOT_COMPLETED, record.symbol, record.id))
        issues = tuple(dict.fromkeys((*assessment.issues, *receipt_issues, *conflicts, *coverage.issues)))
        if receipt_issues or conflicts or not coverage.complete:
            issues = tuple(issue for issue in issues if issue.code is not StudyReason.STUDY_RULES_MET)
            assessment = replace(assessment, status=StudyStatus.INSUFFICIENT_EVIDENCE, issues=issues)
        return RegisteredStudyEvidence(StudyReadStatus.READ, study.sha256, assessment,
                                       tuple(r.id for r in records), tuple(v.id for v in validations), tuple(tasks),
                                       attempt_coverage=coverage)
    except StudyArchiveError as exc:
        return RegisteredStudyEvidence(StudyReadStatus.FAILED, expected_sha256, failure=exc.code)
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, AttributeError,
            OverflowError, DecimalException) as exc:
        return RegisteredStudyEvidence(StudyReadStatus.FAILED, expected_sha256,
                                       failure=StudyArchiveFailure.DATABASE_READ_FAILED,
                                       sqlite_errorcode=getattr(exc, "sqlite_errorcode", None))
