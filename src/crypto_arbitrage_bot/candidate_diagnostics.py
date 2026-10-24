"""Bounded read-only failure diagnosis from registered public exit archives.

Archive consistency and recorded source failures are separate from financial
validation, exchange authentication and strategy qualification. No IO refresh,
credential access, task mutation or trading permission is provided here.
"""

import gzip
import hashlib
import json
import re
import sqlite3
import zlib
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any

from crypto_arbitrage_bot.candidate_study import MAX_RAW_ARCHIVE_BYTES, load_index
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.exchange_metadata import MetadataErrorKind
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure
from crypto_arbitrage_bot.market_data.candidate_catalog import CatalogReason
from crypto_arbitrage_bot.models.data_quality import DataQualityReason
from crypto_arbitrage_bot.models.snapshot import SnapshotIssueKind
from crypto_arbitrage_bot.models.spot import SpotErrorKind
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind
from crypto_arbitrage_bot.models.validation_task import TaskFailure
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

_FILENAME = re.compile(r"[0-9a-f]{32}\.json(?:\.gz)?\Z")
_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,79}\Z")
_PHASES = frozenset({"EXIT_MARKET", "ACTUAL_FUNDING"})
_STATES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED"})
_KNOWN_CODES = frozenset(code.value for enum in (MetadataErrorKind, FundingFailure, CatalogReason,
    DataQualityReason, SnapshotIssueKind, SpotErrorKind, UsdtmErrorKind, TaskFailure) for code in enum)


@dataclass(frozen=True)
class DiagnosticLimits:
    max_files: int = 5000
    max_tasks: int = 10000
    max_total_bytes: int = 512*1024*1024
    max_seconds: int = 60

    def __post_init__(self) -> None:
        for value, maximum in ((self.max_files, 10000), (self.max_tasks, 100000),
                               (self.max_total_bytes, 1024*1024*1024), (self.max_seconds, 120)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("BOUNDED_DIAGNOSTIC_LIMITS_REQUIRED")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("DUPLICATE_ARCHIVE_FIELD")
    return value


def _codes(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("INVALID_DIAGNOSTIC_CODES")
    result = []
    for item in value:
        code = item.get("code", item.get("kind")) if isinstance(item, dict) else item
        if not isinstance(code, str) or not _CODE.fullmatch(code) or code not in _KNOWN_CODES:
            raise ValueError("INVALID_DIAGNOSTIC_CODE")
        result.append(code)
    return result


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("ARCHIVE_UTC_TIME_REQUIRED")
    return utc_timestamp(datetime.fromisoformat(value))


def _source(source: dict[str, Any], *, metadata: bool, endpoint: PublicEndpoint,
            symbol: str) -> dict[str, Any]:
    # Never echo raw responses, headers, arbitrary error messages or request
    # parameters. Only known public endpoints, stable codes and timings leave.
    observed_endpoint = PublicEndpoint(source["endpoint"] if metadata else source["request"]["endpoint"])
    if observed_endpoint is not endpoint:
        raise ValueError("SOURCE_ENDPOINT_DIFFERS")
    if metadata:
        identity = source.get("research_instrument")
        if identity is not None and identity["symbol"] != symbol:
            raise ValueError("SOURCE_SYMBOL_DIFFERS")
    else:
        params = source["request"]["params"]
        # PublicGetRequest normally retains immutable pairs; some original
        # offline sources retain a mapping. Both require unambiguous identity.
        if isinstance(params, list):
            if len(params) > 10 or any(not isinstance(pair, list) or len(pair) != 2
                                      or not isinstance(pair[0], str) for pair in params):
                raise ValueError("INVALID_SOURCE_PARAMETERS")
            params = _unique([(pair[0], pair[1]) for pair in params])
        if not isinstance(params, dict) or params.get("symbol") != symbol:
            raise ValueError("SOURCE_SYMBOL_DIFFERS")
    started = _time(source["requested_at"] if metadata else source["started_at"])
    finished = _time(source["finished_at"])
    if finished < started:
        raise ValueError("INVALID_SOURCE_TIME_ORDER")
    error = source.get("error")
    kinds = [] if error is None else _codes([error["kind"]])
    status = source.get("status_code")
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        raise ValueError("INVALID_SOURCE_HTTP_STATUS")
    return {"endpoint": endpoint.name, "started_at": started, "finished_at": finished,
            "http_status": status, "error_codes": kinds, "body_truncated": source["body_truncated"] is True}


def _summarize(value: dict[str, Any], task: dict[str, Any], sha: str,
               at: datetime) -> dict[str, Any]:
    required = {"task_id", "opportunity_id", "validation_id", "symbol", "registration_sha256",
                "phase", "recorded_at", "catalogs", "captures"}
    if (set(value) != required or value["phase"] not in _PHASES
            or value["registration_sha256"] != sha
            or any(value[key] != task[key] for key in ("opportunity_id", "validation_id", "symbol"))
            or value["task_id"] != task["id"] or _time(value["recorded_at"]) > at):
        raise ValueError("ARCHIVE_TASK_OR_REGISTRATION_DIFFERS")
    if (not isinstance(value["catalogs"], list) or len(value["catalogs"]) > 2
            or not isinstance(value["captures"], list) or len(value["captures"]) > 1):
        raise ValueError("BOUNDED_TASK_CAPTURE_REQUIRED")
    causes: list[str] = []
    sources: list[dict[str, Any]] = []
    for catalog in value["catalogs"]:
        causes.extend("CATALOG:"+code for code in _codes(catalog["reason_codes"]))
        if len(catalog["sources"]) != 2:
            raise ValueError("TWO_METADATA_SOURCES_REQUIRED")
        sources.extend(_source(source, metadata=True, endpoint=endpoint, symbol=task["symbol"])
            for source, endpoint in zip(catalog["sources"],
                (PublicEndpoint.SPOT_EXCHANGE_INFO, PublicEndpoint.PERP_EXCHANGE_INFO), strict=True))
    for capture in value["captures"]:
        if capture["phase"] != value["phase"] or capture["symbol"] != task["symbol"]:
            raise ValueError("CAPTURE_TASK_PHASE_DIFFERS")
        if value["phase"] == "EXIT_MARKET":
            observation = capture["observation"]
            assembly = observation["assembly"]
            if assembly["symbol"] != task["symbol"] or observation["candidate"]["symbol"] != task["symbol"]:
                raise ValueError("CAPTURE_SYMBOL_DIFFERS")
            causes.extend("ASSEMBLY:"+code for code in _codes(assembly["issues"]))
            causes.extend("QUALITY:"+code for code in _codes(observation["data_quality"]["issues"]))
            expected = {"spot_depth": PublicEndpoint.SPOT_DEPTH, "perp_depth": PublicEndpoint.PERP_DEPTH,
                "premium_index": PublicEndpoint.PERP_PREMIUM_INDEX, "open_interest": PublicEndpoint.PERP_OPEN_INTEREST}
            if not isinstance(assembly["sources"], dict) or set(assembly["sources"]) != set(expected):
                raise ValueError("SNAPSHOT_SOURCES_DIFFER")
            for key, source in assembly["sources"].items():
                if source is not None:
                    sources.append(_source(source, metadata=False, endpoint=expected[key], symbol=task["symbol"]))
        else:
            window = capture["window"]
            if window["symbol"] != task["symbol"] or _time(window["opened_at"]) >= _time(window["closed_at"]):
                raise ValueError("INVALID_FUNDING_WINDOW")
            if window["failure"] is not None:
                causes.extend("FUNDING:"+code for code in _codes([window["failure"]]))
    for source in sources:
        causes.extend(source["endpoint"]+":"+code for code in source["error_codes"])
        if source["body_truncated"]:
            causes.append(source["endpoint"]+":BODY_TRUNCATED")
    return {"phase": value["phase"], "recorded_at": _time(value["recorded_at"]),
            "recorded_cause_codes": sorted(set(causes)), "source_observations": sources}


def diagnose_candidate_exits(settings: Settings, *, evaluated_at: datetime,
                             limits: DiagnosticLimits | None = None) -> dict[str, Any]:
    """One bounded DB read transaction plus an independently observed archive scan.

    Concurrent archival publication can precede task FINISH; nonterminal tasks
    are counted separately. Missing/ambiguous/corrupt files never prove failure
    causality or qualify a result. No cross-filesystem atomic snapshot is claimed.
    """
    at = utc_timestamp(evaluated_at)
    limits = DiagnosticLimits() if limits is None else limits
    index, studies = load_index(settings)
    research = settings.research
    assert research is not None and research.study_registration is not None
    deadline = monotonic()+limits.max_seconds
    def check_time() -> None:
        if monotonic() > deadline:
            raise ValueError("DIAGNOSTIC_TIME_LIMIT")
    with closing(sqlite3.connect(research.sqlite_path.resolve().as_uri()+"?mode=ro", uri=True,
                                timeout=min(limits.max_seconds, 5))) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.set_progress_handler(lambda: int(monotonic() > deadline), 1000)
        db.execute("BEGIN")
        rows = db.execute("SELECT t.*,o.symbol AS parent_symbol FROM validation_tasks t "
                          "LEFT JOIN opportunities o ON o.id=t.opportunity_id ORDER BY t.id LIMIT ?",
                          (limits.max_tasks+1,)).fetchall()
        if len(rows) > limits.max_tasks:
            raise ValueError("DIAGNOSTIC_TASK_LIMIT")
        tasks = {row["id"]: dict(row) for row in rows}
    for task in tasks.values():
        if (task["symbol"] not in studies or task["symbol"] != task["parent_symbol"]
                or task["status"] not in _STATES):
            raise ValueError("DIAGNOSTIC_TASK_SCOPE_DIFFERS")
        if task["status"] == "FAILED":
            _codes([task["failure_code"]])
    failed = {key: value for key, value in tasks.items() if value["status"] == "FAILED"}
    archives: dict[str, list[dict[str, Any]]] = {key: [] for key in failed}
    issues: list[dict[str, str]] = []
    count, total = 0, 0
    directory = research.study_registration.path.parent / "exit-evidence"
    if directory.exists():
        for path in directory.iterdir():
            check_time()
            count += 1
            if count > limits.max_files:
                issues.append({"code": "DIAGNOSTIC_FILE_LIMIT"})
                break
            if (not _FILENAME.fullmatch(path.name) or path.is_symlink() or not path.is_file()
                    or path.resolve().parent != directory.resolve()):
                issues.append({"code": "UNKNOWN_OR_UNSAFE_ARCHIVE"})
                continue
            try:
                remaining = limits.max_total_bytes-total
                if remaining <= 0:
                    issues.append({"code": "DIAGNOSTIC_TOTAL_BYTES_LIMIT"})
                    break
                stream = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
                with stream:
                    body = stream.read(min(MAX_RAW_ARCHIVE_BYTES, remaining)+1)
                total += len(body)
                if len(body) > MAX_RAW_ARCHIVE_BYTES or total > limits.max_total_bytes:
                    issues.append({"code": "DIAGNOSTIC_ARCHIVE_BYTES_LIMIT"})
                    break
                check_time()
                value = json.loads(body, object_pairs_hook=_unique)
                if not isinstance(value, dict) or exact_json(value).encode() != body:
                    raise ValueError("CANONICAL_PUBLIC_ARCHIVE_REQUIRED")
                archive_task = tasks.get(value["task_id"])
                if archive_task is None:
                    raise ValueError("UNKNOWN_ARCHIVE_TASK")
                summary = _summarize(value, archive_task, studies[archive_task["symbol"]].sha256, at)
                if archive_task["id"] in failed:
                    archives[archive_task["id"]].append({"filename": path.name,
                        "canonical_sha256": hashlib.sha256(body).hexdigest(), **summary})
            except (OSError, EOFError, ValueError, TypeError, KeyError, RecursionError, zlib.error) as error:
                # The exception message can contain arbitrary source text; do
                # not expose it. A file with bad identity cannot be attributed.
                issues.append({"code": "CORRUPT_OR_UNBOUND_ARCHIVE", "filename": path.name,
                               "error_type": type(error).__name__})
    check_time()
    items = []
    for identifier, task in failed.items():
        files = archives[identifier]
        phases = Counter(file["phase"] for file in files)
        ambiguous = any(number > 1 for number in phases.values())
        items.append({"task_id": identifier, "symbol": task["symbol"], "due_at": _time(task["due_at"]),
            "failure_code": task["failure_code"], "attempts": task["attempts"],
            "evidence_status": "AMBIGUOUS" if ambiguous else "READ" if files else "MISSING",
            "recorded_cause_codes": [] if ambiguous else sorted({code for file in files
                for code in file["recorded_cause_codes"]}), "archives": files})
    return {"schema_version": "registered-candidate-exit-diagnostics-v1", "evaluated_at": at,
        "status": "INCOMPLETE" if issues or any(v["evidence_status"] != "READ" for v in items) else "READ",
        "meaning": "Recorded public source failures and local archive consistency; no financial audit or causality guarantee",
        "index_sha256": research.study_registration.sha256, "symbols": index["symbols"],
        "task_states": dict(Counter(task["status"] for task in tasks.values())),
        "failure_codes": dict(Counter(task["failure_code"] for task in failed.values())),
        "scan": {"files_observed": count, "decompressed_bytes": total, "limits": limits},
        "archive_issues": issues, "failed_tasks": items, "credentials_read": False,
        "orders_submitted": False, "trade_allowed": False, "qualification_issued": False,
        "financial_values_recomputed": False, "external_authentication_proved": False}
