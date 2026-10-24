"""Started/finished research attempts, including failures before Opportunity save.

No PnL, network or SQL writes. Local chained files detect accidental deletion;
they do not authenticate a malicious operator or guarantee missing-data fairness.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.strategy_study import StudyIssue, StudyReason
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_registry import (
    RegisteredStudy,
    StudyArchiveError,
    StudyArchiveFailure,
)
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import file_lock


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _head_path(study: RegisteredStudy) -> Path:
    return study.path.with_suffix(".attempt-head.json")


def _directory(study: RegisteredStudy) -> Path:
    return study.path.with_suffix(".attempts")


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        body = stream.read(64001).decode("utf-8")
    value = json.loads(body)
    if len(body) > 64000 or not isinstance(value, dict) or exact_json(value) != body:
        raise ValueError("bounded canonical attempt evidence required")
    return dict(value)


def _write(path: Path, value: object, *, exclusive: bool = True) -> None:
    with path.open("xb" if exclusive else "wb") as output:
        output.write(exact_json(value).encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def create_attempt_head(study: RegisteredStudy) -> None:
    _write(_head_path(study), {"registration_sha256": study.sha256, "count": 0, "last_sha256": study.sha256})


def _at(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


def _head(study: RegisteredStudy) -> dict[str, Any]:
    head = _read(_head_path(study))
    if (head["registration_sha256"] != study.sha256 or type(head["count"]) is not int
            or not 0 <= head["count"] <= 1_000_000):
        raise ValueError("attempt head identity/count differs")
    return head


def _append(study: RegisteredStudy, value: dict[str, Any]) -> None:
    if study.max_sampling_gap is None:
        raise StudyArchiveError(StudyArchiveFailure.INVALID_REGISTRATION)
    try:
        with file_lock(study.path.with_suffix(".attempt.lock")):
            head = _head(study)
            count = head["count"]
            if count >= 1_000_000:
                raise ValueError("bounded attempt archive exhausted")
            directory = _directory(study)
            directory.mkdir(exist_ok=True)
            previous = study.sha256 if count == 0 else _sha(exact_json(_read(directory / f"{count-1:06d}.json")))
            if previous != head["last_sha256"]:
                raise ValueError("lost/changed prior attempt event")
            event = {"registration_sha256": study.sha256, "previous_sha256": previous, "sequence": count, **value}
            _write(directory / f"{count:06d}.json", event)
            temporary = _head_path(study).with_suffix(".tmp")
            _write(temporary, {"registration_sha256": study.sha256, "count": count+1,
                               "last_sha256": _sha(exact_json(event))}, exclusive=False)
            temporary.replace(_head_path(study))
    except StudyArchiveError:
        raise
    except Exception as error:
        # Preserve partial evidence; never reset or print exception bodies.
        raise StudyArchiveError(StudyArchiveFailure.ATTEMPT_LOG_CORRUPT) from error


def start_attempt(study: RegisteredStudy, cycle_id: str, symbol: str, started_at: datetime,
                  *, recorded_at: datetime) -> None:
    started, at = utc_timestamp(started_at), utc_timestamp(recorded_at)
    if (not isinstance(cycle_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", cycle_id)
            or symbol not in study.protocol.symbols or not study.protocol.study_start <= started < study.protocol.study_end
            or not started <= at <= started+study.max_capture_delay):
        raise StudyArchiveError(StudyArchiveFailure.RECORD_OUTSIDE_STUDY)
    _append(study, {"kind": "START", "cycle_id": cycle_id, "symbol": symbol,
                    "started_at": started, "recorded_at": at, "policy_sha256": _sha(study.policy_json)})


def finish_attempt(study: RegisteredStudy, cycle_id: str, symbol: str, *, ended_at: datetime,
                   status: str, failure_stage: str | None, reason_codes: tuple[str, ...],
                   record: OpportunityRecord | None, recorded_at: datetime) -> None:
    ended, at = utc_timestamp(ended_at), utc_timestamp(recorded_at)
    if (status not in ("SAVED", "FAILED", "INTERRUPTED") or not ended <= at <= ended+study.max_capture_delay
            or (status == "SAVED" and record is None) or symbol not in study.protocol.symbols
            or (record is not None and record.symbol != symbol)
            or len(reason_codes) > 32 or any(not re.fullmatch(r"[A-Z0-9_]{1,80}", code) for code in reason_codes)
            or (failure_stage is not None and not re.fullmatch(r"[A-Z_]{1,64}", failure_stage))):
        raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_CORRUPT)
    _append(study, {"kind": "FINISH", "cycle_id": cycle_id, "symbol": symbol, "ended_at": ended, "recorded_at": at,
                    "status": status, "failure_stage": failure_stage, "reason_codes": reason_codes,
                    "opportunity_id": None if record is None else record.id,
                    "record_sha256": None if record is None else _sha(exact_json(record))})


@dataclass(frozen=True)
class AttemptCoverage:
    complete: bool
    started: int
    finished: int
    failed: int
    interrupted: int
    maximum_gaps: tuple[tuple[str, timedelta], ...]
    issues: tuple[StudyIssue, ...]


def read_attempt_coverage(study: RegisteredStudy, records: tuple[OpportunityRecord, ...], *,
                          evaluated_at: datetime) -> AttemptCoverage:
    evaluated_at = utc_timestamp(evaluated_at)
    if study.max_sampling_gap is None:
        return AttemptCoverage(False, 0, 0, 0, 0, (), (StudyIssue(StudyReason.MISSING_ATTEMPT_COVERAGE),))
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    try:
        with file_lock(study.path.with_suffix(".attempt.lock")):
            head = _head(study)
            paths = sorted(_directory(study).iterdir()) if _directory(study).exists() else []
            if head["count"] != len(paths):
                raise ValueError("lost/extra event, including deleted final event")
            previous = study.sha256
            for index, path in enumerate(paths):
                event = _read(path)
                if (path.name != f"{index:06d}.json" or event["sequence"] != index
                        or event["registration_sha256"] != study.sha256 or event["previous_sha256"] != previous
                        or _at(event["recorded_at"]) > evaluated_at):
                    raise ValueError("attempt chain/timing changed")
                previous = _sha(exact_json(event))
                identifier = event["cycle_id"]
                if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                    raise ValueError("invalid attempt identity")
                if event["kind"] == "START":
                    if (identifier in starts or event["symbol"] not in study.protocol.symbols
                            or event["policy_sha256"] != _sha(study.policy_json)
                            or not study.protocol.study_start <= _at(event["started_at"]) < study.protocol.study_end
                            or not _at(event["started_at"]) <= _at(event["recorded_at"])
                            <= _at(event["started_at"])+study.max_capture_delay):
                        raise ValueError("invalid/duplicate start")
                    starts[identifier] = event
                elif event["kind"] == "FINISH":
                    start = starts.get(identifier)
                    codes = event["reason_codes"]
                    stage = event["failure_stage"]
                    if (start is None or identifier in ends or event["symbol"] != start["symbol"]
                            or not _at(start["recorded_at"]) <= _at(event["ended_at"]) <= _at(event["recorded_at"])
                            <= _at(event["ended_at"])+study.max_capture_delay
                            or event["status"] not in ("SAVED", "FAILED", "INTERRUPTED")
                            or not isinstance(codes, list) or len(codes) > 32
                            or any(not isinstance(code, str) or not re.fullmatch(r"[A-Z0-9_]{1,80}", code) for code in codes)
                            or (stage is not None and (not isinstance(stage, str) or not re.fullmatch(r"[A-Z_]{1,64}", stage)))
                            or (event["opportunity_id"] is not None and not isinstance(event["opportunity_id"], str))
                            or (event["record_sha256"] is not None and (not isinstance(event["record_sha256"], str)
                                or not re.fullmatch(r"[a-f0-9]{64}", event["record_sha256"])))):
                        raise ValueError("invalid/duplicate finish")
                    ends[identifier] = event
                else:
                    raise ValueError("unknown attempt kind")
            if previous != head["last_sha256"]:
                raise ValueError("attempt head changed")
    except Exception:
        return AttemptCoverage(False, len(starts), len(ends), 0, 0, (), (StudyIssue(StudyReason.CORRUPT_ATTEMPT_LOG),))
    issues = []
    linked = {}
    by_record = {r.id: r for r in records}
    for identifier, start in starts.items():
        end = ends.get(identifier)
        if end is None:
            issues.append(StudyIssue(StudyReason.INCOMPLETE_ATTEMPT, start["symbol"]))
            continue
        oid = end["opportunity_id"]
        if oid is None:
            if end["status"] == "SAVED" or end["record_sha256"] is not None:
                issues.append(StudyIssue(StudyReason.ATTEMPT_RECORD_MISMATCH, start["symbol"]))
        else:
            record = by_record.get(oid)
            if (oid in linked or record is None or record.symbol != start["symbol"]
                    or _sha(exact_json(record)) != end["record_sha256"]
                    or not _at(start["started_at"]) <= record.created_at <= _at(end["ended_at"])):
                issues.append(StudyIssue(StudyReason.ATTEMPT_RECORD_MISMATCH, start["symbol"], oid))
            linked[oid] = identifier
    for record in records:
        if record.id not in linked:
            issues.append(StudyIssue(StudyReason.MISSING_ATTEMPT_COVERAGE, record.symbol, record.id))
    gaps = []
    limit = min(study.protocol.study_end, utc_timestamp(evaluated_at))
    for symbol in study.protocol.symbols:
        observations = sorted(_at(e["started_at"]) for e in starts.values() if e["symbol"] == symbol)
        anchors = [study.protocol.study_start, *observations, max(study.protocol.study_start, limit)]
        gap = max((b-a for a, b in zip(anchors, anchors[1:], strict=False)), default=timedelta(0))
        gaps.append((symbol, gap))
        if gap > study.max_sampling_gap or not observations:
            issues.append(StudyIssue(StudyReason.SAMPLING_GAP_TOO_LARGE, symbol))
    return AttemptCoverage(not issues, len(starts), len(ends),
        sum(e["status"] == "FAILED" for e in ends.values()), sum(e["status"] == "INTERRUPTED" for e in ends.values()),
        tuple(gaps), tuple(dict.fromkeys(issues)))
