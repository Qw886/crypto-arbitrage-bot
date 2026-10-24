"""Preregistered public candidates, original costs and planned-exit validation.

Candidate scope never changes Settings.symbols or the execution whitelist.
Each fixed symbol has its own assessment; controls need not show positive PnL
for another symbol's evidence to be readable. No trading permission is issued.
"""

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot.arbitrage.candidate_estimate import (
    candidate_estimate_config,
    compare_candidate_windows,
)
from crypto_arbitrage_bot.arbitrage.sizing import QuantityRules
from crypto_arbitrage_bot.candidate_forward_research import CandidateValidationSources
from crypto_arbitrage_bot.candidate_research import collect_catalog
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CandidateCatalog,
    CatalogStatus,
    ResearchCandidate,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.market_data.candidate_market import CandidateObservation
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.models.strategy_study import StrategyStudyProtocol
from crypto_arbitrage_bot.models.validation_task import TaskFailure, ValidationTask
from crypto_arbitrage_bot.research_task_batch import process_task_batch
from crypto_arbitrage_bot.runner import (
    CycleResult,
    CycleStage,
    CycleStatus,
    ResearchRunner,
    StopSignal,
)
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.persistence import PersistenceStatus, build_and_save_opportunity
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository
from crypto_arbitrage_bot.storage.study_attempts import create_attempt_head, start_attempt
from crypto_arbitrage_bot.storage.study_registry import (
    RegisteredStudy,
    StrategyStudyRegistry,
    StudyArchiveError,
    StudyArchiveFailure,
    full_research_policy,
    load_registered_study,
)
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.validation_scheduler import ValidationScheduler

INDEX_VERSION = "registered-candidate-index-v1"
EXIT_RULE = "ceil_seconds_to_saved_exit_original_lateness"
MAX_RAW_ARCHIVE_BYTES = 32*1024*1024
MIN_FREE_ARCHIVE_BYTES = 512*1024*1024
_ARCHIVE_WRITE_LOCK = Lock()
DRAIN_MAX_TASKS = 160
DRAIN_MAX_BATCHES = 16
DRAIN_MAX_SECONDS = 240


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_pinned(path: Path, sha: str) -> dict[str, Any]:
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("PINNED_CANDIDATE_HASH_REQUIRED")
    with path.open("rb") as stream:
        body = stream.read(256001)
    if len(body) > 256000 or hashlib.sha256(body).hexdigest() != sha:
        raise ValueError("CANDIDATE_REFERENCE_HASH_DIFFERS")
    value = json.loads(body)
    if not isinstance(value, dict) or exact_json(value).encode() != body:
        raise ValueError("CANONICAL_CANDIDATE_REFERENCE_REQUIRED")
    return value


def is_candidate_index(path: Path, sha: str) -> bool:
    return read_pinned(path, sha).get("schema_version") == INDEX_VERSION


def write_new(path: Path, value: object) -> str:
    body = exact_json(value)
    with path.open("xb") as stream:
        stream.write(body.encode())
        stream.flush()
        os.fsync(stream.fileno())
    return digest(body)


def write_public_archive(path: Path, value: object) -> str:
    """Lossless canonical public data, bounded per file and with disk reserve.

    Registration/index/head/receipt bytes remain unchanged. Partial archives
    remain for diagnosis; no validation qualifies after archive persistence fails.
    """
    body = exact_json(value).encode()
    if len(body) > MAX_RAW_ARCHIVE_BYTES:
        raise ValueError("PUBLIC_ARCHIVE_BODY_LIMIT_EXCEEDED")
    packed = gzip.compress(body, mtime=0)
    with _ARCHIVE_WRITE_LOCK:
        if shutil.disk_usage(path.parent).free < MIN_FREE_ARCHIVE_BYTES+len(packed):
            raise OSError("PUBLIC_ARCHIVE_DISK_RESERVE_REQUIRED")
        with path.with_suffix(path.suffix+".gz").open("xb") as stream:
            stream.write(packed)
            stream.flush()
            os.fsync(stream.fileno())
    return hashlib.sha256(body).hexdigest()


def _candidate(value: dict[str, Any]) -> ResearchCandidate:
    data = dict(value)
    for name in ("spot_rules", "perp_rules"):
        rules = data[name]
        if set(rules) != {"step_size", "min_quantity", "min_notional", "max_quantity"} or any(
                type(val) is not str for key, val in rules.items() if val is not None or key != "max_quantity"):
            raise ValueError("EXACT_RULE_DECIMALS_REQUIRED")
        data[name] = QuantityRules(Decimal(rules["step_size"]), Decimal(rules["min_quantity"]),
            Decimal(rules["min_notional"]), None if rules["max_quantity"] is None else Decimal(rules["max_quantity"]))
    for name in ("spot_tick_size", "perp_tick_size"):
        data[name] = Decimal(data[name])
        if not data[name].is_finite() or data[name] <= 0:
            raise ValueError("INVALID_CANDIDATE_TICK")
    for name in ("perp_onboard_at", "perp_delivery_at"):
        data[name] = utc_timestamp(datetime.fromisoformat(data[name]))
    result = ResearchCandidate(**data)
    ResearchInstrument(result.symbol, result.base_asset)
    if type(result.benchmark) is not bool or result.perp_onboard_at >= result.perp_delivery_at:
        raise ValueError("INVALID_CANDIDATE_IDENTITY")
    return result


def fingerprint(settings: Settings, candidate: ResearchCandidate) -> str:
    research = settings.research
    assert research is not None and research.candidate_economics is not None
    config = candidate_estimate_config(candidate, research.candidate_economics,
                                       settings.fees.as_rates(), research.data_quality)
    return digest(exact_json({"estimate": config, "decision": research.decision}))


def verify_candidate_policy(study: RegisteredStudy, settings: Settings) -> RegisteredStudy:
    try:
        context = json.loads(study.candidate_context_json or "null")
        research = settings.research
        if (not isinstance(context, dict) or set(context) != {"profile", "candidate", "catalog_sha256", "exit_rule"}
                or context["exit_rule"] != EXIT_RULE or research is None or settings.testnet is not None
                or research.candidate_catalog is None or research.candidate_economics is None
                or research.validation_scheduler is None or study.max_sampling_gap is None
                or full_research_policy(settings) != study.policy_json
                or research.sqlite_path.resolve() != study.database_path):
            raise ValueError("candidate policy scope differs")
        profile = context["profile"]
        if tuple(research.candidate_economics.windows) != (profile,):
            raise ValueError("single predeclared candidate profile required")
        candidate = _candidate(context["candidate"])
        p = study.protocol
        assumptions = research.candidate_economics.windows[profile]
        bound = assumptions.holding_seconds if assumptions.funding_projection is None else (
            assumptions.funding_projection.max_wait_seconds+assumptions.funding_projection.exit_delay_seconds)
        if (p.symbols != (candidate.symbol,) or p.config_fingerprints != ((candidate.symbol, fingerprint(settings, candidate)),)
                or p.config_version != f"{research.config_version}:{profile}" or p.formula_version != research.formula_version
                or p.max_holding_period != timedelta(seconds=bound)
                or research.validation_scheduler.max_lateness_seconds is None
                or p.max_exit_lateness != timedelta(seconds=research.validation_scheduler.max_lateness_seconds or 0)
                or len(context["catalog_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in context["catalog_sha256"])):
            raise ValueError("candidate protocol differs")
        return study
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise StudyArchiveError(StudyArchiveFailure.POLICY_MISMATCH) from error


def register_candidate(registry: StrategyStudyRegistry, protocol: StrategyStudyProtocol,
                       settings: Settings, candidate: ResearchCandidate, *, profile: str,
                       catalog_sha256: str, max_capture_delay: timedelta,
                       max_sampling_gap: timedelta) -> RegisteredStudy:
    """Reference metadata precedes registration; entry books must follow it."""
    research = settings.research
    assert research is not None
    at = utc_timestamp(registry.clock())
    if not protocol.registered_at <= at <= protocol.study_start or max_capture_delay < timedelta(0):
        raise StudyArchiveError(StudyArchiveFailure.REGISTRATION_NOT_PRIOR)
    registry.directory.mkdir(parents=True, exist_ok=True)
    path = registry.directory / (digest(protocol.id)+".json")
    context = {"profile": profile, "candidate": candidate, "catalog_sha256": catalog_sha256, "exit_rule": EXIT_RULE}
    envelope = {"schema_version": 3, "protocol": protocol, "registered_at": at,
                "database_path": str(research.sqlite_path.resolve()), "policy": json.loads(full_research_policy(settings)),
                "max_capture_delay": max_capture_delay, "max_sampling_gap": max_sampling_gap,
                "candidate_context": context}
    # Validate before writing an irreversible registration. The field layout is
    # the same as the loaded value; no SQL initialization or market IO here.
    provisional = RegisteredStudy(path, "0"*64, protocol, at, research.sqlite_path.resolve(),
        full_research_policy(settings), max_capture_delay, max_sampling_gap, exact_json(context))
    verify_candidate_policy(provisional, settings)
    if not timedelta(0) < max_sampling_gap <= protocol.study_end-protocol.study_start:
        raise ValueError("POSITIVE_PREDECLARED_SAMPLING_GAP_REQUIRED")
    sha = write_new(path, envelope)
    saved = load_registered_study(path, sha)
    create_attempt_head(saved)
    return saved


def load_index(settings: Settings) -> tuple[dict[str, Any], dict[str, RegisteredStudy]]:
    research = settings.research
    if research is None or research.study_registration is None:
        raise ValueError("PINNED_CANDIDATE_INDEX_REQUIRED")
    ref = research.study_registration
    index = read_pinned(ref.path, ref.sha256)
    if (set(index) != {"schema_version", "symbols", "policy_sha256", "database_path", "catalog_sha256",
                      "entry_interval_seconds", "studies"} or index["schema_version"] != INDEX_VERSION
            or index["policy_sha256"] != digest(full_research_policy(settings))
            or index["database_path"] != str(research.sqlite_path.resolve())
            or type(index["entry_interval_seconds"]) is not int or not 1 <= index["entry_interval_seconds"] <= 86400
            or not isinstance(index["symbols"], list) or not isinstance(index["studies"], dict)
            or any(type(symbol) is not str for symbol in index["symbols"])
            or not 1 <= len(index["symbols"]) <= 10 or len(set(index["symbols"])) != len(index["symbols"])
            or set(index["studies"]) != set(index["symbols"])):
        raise ValueError("CANDIDATE_INDEX_SCOPE_DIFFERS")
    studies = {}
    windows = set()
    for symbol in index["symbols"]:
        value = index["studies"][symbol]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError("EXACT_CANDIDATE_STUDY_REFERENCE_REQUIRED")
        study = load_registered_study(Path(value["path"]), value["sha256"])
        verify_candidate_policy(study, settings)
        if (study.protocol.symbols != (symbol,)
                or json.loads(study.candidate_context_json or "null")["catalog_sha256"] != index["catalog_sha256"]
                or study.max_sampling_gap is None
                or timedelta(seconds=index["entry_interval_seconds"]) > study.max_sampling_gap):
            raise ValueError("CANDIDATE_INDEX_REGISTRATION_DIFFERS")
        windows.add((study.protocol.study_start, study.protocol.study_end, study.protocol.max_holding_period,
                     study.protocol.max_exit_lateness))
        studies[symbol] = study
    if len(windows) != 1:
        raise ValueError("COMMON_FIXED_CANDIDATE_WINDOWS_REQUIRED")
    return index, studies


def planned_horizon(record: OpportunityRecord) -> int:
    closed = utc_timestamp(datetime.fromisoformat(json.loads(record.details_json)["estimate"]["scenario"]["closed_at"]))
    delta = closed-record.created_at
    micros = (delta.days*86400+delta.seconds)*1_000_000+delta.microseconds
    if micros <= 0:
        raise ValueError("PLANNED_EXIT_ALREADY_PASSED_AT_SAVE")
    return (micros+999999)//1_000_000


class ArchivedCandidateSources(CandidateValidationSources):
    """Raw public exit responses reach disk before any validation can be saved."""

    def __init__(self, settings: Settings, studies: dict[str, RegisteredStudy],
                 clock: Callable[[], datetime]) -> None:
        super().__init__(settings)
        self.studies, self.clock = studies, clock
        self.task: ValidationTask | None = None
        self.archive_failed = False

    def _archive(self, phase: str) -> None:
        if not self.catalogs and not self.captures:
            return
        if self.task is None or self.task.symbol not in self.studies:
            raise ValueError("REGISTERED_EXIT_TASK_REQUIRED")
        study = self.studies[self.task.symbol]
        directory = study.path.parent / "exit-evidence"
        try:
            directory.mkdir(exist_ok=True)
            write_public_archive(directory / f"{uuid4().hex}.json", {
                "task_id": self.task.id, "opportunity_id": self.task.opportunity_id,
                "validation_id": self.task.validation_id, "symbol": self.task.symbol,
                "registration_sha256": study.sha256, "phase": phase,
                "recorded_at": self.clock(), "catalogs": self.catalogs, "captures": self.captures})
        except Exception:
            self.archive_failed = True
            raise
        self.catalogs.clear()
        self.captures.clear()

    def collect(self, symbol: str) -> SnapshotAssembly:
        if self.task is None or self.task.symbol != symbol:
            raise ValueError("REGISTERED_EXIT_TASK_REQUIRED")
        try:
            assembly = super().collect(symbol)
            observation = self.captures[-1]["observation"]
            context = json.loads(self.studies[symbol].candidate_context_json or "null")
            if (not isinstance(observation, CandidateObservation)
                    or exact_json(observation.candidate) != exact_json(context["candidate"])):
                raise ValueError("FROZEN_EXIT_CANDIDATE_RULES_OR_IDENTITY_CHANGED")
            return assembly
        finally:
            self._archive("EXIT_MARKET")

    def fetch_window(self, symbol: str, opened_at: datetime, closed_at: datetime) -> FundingWindow:
        if self.task is None or self.task.symbol != symbol:
            raise ValueError("REGISTERED_EXIT_TASK_REQUIRED")
        try:
            return super().fetch_window(symbol, opened_at, closed_at)
        finally:
            self._archive("ACTUAL_FUNDING")


class PlannedCandidateScheduler(ValidationScheduler):
    def __init__(self, settings: Settings, studies: dict[str, RegisteredStudy], sources: CandidateValidationSources,
                 *, clock: Callable[[], datetime] = utc_now) -> None:
        research = settings.research
        assert research is not None and research.validation_scheduler is not None
        config = SQLiteConfig(research.sqlite_path)
        super().__init__(research.validation_scheduler, SQLiteOpportunityRepository(config),
            SQLiteForwardValidationRepository(config), SQLiteValidationTaskRepository(config), sources, sources, clock=clock)
        self.studies = studies
        self.sources = sources
        self.public_settings = settings
        self.progress: Callable[[], None] = lambda: None
        self.steady: Callable[[], float] = monotonic
        self.source_factory: Callable[[], CandidateValidationSources] = lambda: ArchivedCandidateSources(settings, studies, clock)

    def _parallel_execute(self, task: ValidationTask) -> None:
        if self.validations.get_forward_validation_by_id(task.validation_id) is not None:
            # A saved on-time observation survives a crash before task FINISH.
            # Original identity/status checks recover it without new sources.
            ValidationScheduler._execute(self, task)
            return
        if self._window_missed(task, utc_timestamp(self.clock())):
            return
        # Transport/catalog/capture state is owned by exactly one task. SQL
        # connections and immutable policies remain the existing repositories.
        child = PlannedCandidateScheduler(self.public_settings, self.studies, self.source_factory(), clock=self.clock)
        child.stopped, child.logger = self.stopped, self.logger
        child._execute(task)

    def _execute(self, task: ValidationTask) -> None:
        if isinstance(self.sources, ArchivedCandidateSources):
            if self.sources.archive_failed:
                self._finish(task, TaskFailure.PERSISTENCE_FAILED, "RAW_EXIT_ARCHIVE_UNAVAILABLE")
                return
            self.sources.task = task
        try:
            super()._execute(task)
        finally:
            if isinstance(self.sources, ArchivedCandidateSources):
                self.sources.task = None

    def schedule(self, opportunity_id: str) -> None:
        self.initialize()
        record = self.opportunities.get_opportunity_by_id(opportunity_id)
        if record is None or record.symbol not in self.studies:
            raise ValueError("REGISTERED_CANDIDATE_RECORD_REQUIRED")
        study = self.studies[record.symbol]
        p = study.protocol
        closed = utc_timestamp(datetime.fromisoformat(json.loads(record.details_json)["estimate"]["scenario"]["closed_at"]))
        if (record.config_version != p.config_version or record.formula_version != p.formula_version
                or record.config_fingerprint != dict(p.config_fingerprints)[record.symbol]
                or not p.study_start <= record.created_at < p.study_end
                or not timedelta(0) < closed-record.created_at <= p.max_holding_period):
            raise ValueError("PLANNED_CANDIDATE_SCOPE_DIFFERS")
        self.tasks.create_tasks(opportunity_id, (planned_horizon(record),), now=self.clock())

    def _window_missed(self, task: ValidationTask, at: datetime) -> bool:
        record = self.opportunities.get_opportunity_by_id(task.opportunity_id)
        if record is None:
            raise ValueError("CANDIDATE_ENTRY_MISSING")
        closed = utc_timestamp(datetime.fromisoformat(json.loads(record.details_json)["estimate"]["scenario"]["closed_at"]))
        if at > closed+self.studies[task.symbol].protocol.max_exit_lateness:
            self._finish(task, TaskFailure.VALIDATION_WINDOW_MISSED, "original saved exit lateness exceeded")
            return True
        return super()._window_missed(task, at)

    def process_due(self) -> None:
        self._process_batch(self.settings.batch_size)

    def drain_due(self) -> None:
        """Original batch size; bounded batches without an artificial poll wait."""
        deadline = self.steady()+DRAIN_MAX_SECONDS
        consumed = 0
        for _ in range(DRAIN_MAX_BATCHES):
            if self.stopped() or self.steady() >= deadline or consumed >= DRAIN_MAX_TASKS:
                break
            used = self._process_batch(min(self.settings.batch_size, DRAIN_MAX_TASKS-consumed),
                                       extra_stopped=lambda: self.steady() >= deadline)
            consumed += used
            if used == 0:
                break

    def _process_batch(self, limit: int, *, extra_stopped: Callable[[], bool] = lambda: False) -> int:
        if self.stopped():
            return 0
        self.initialize()
        # Recover save-before-schedule interruption, bounded each batch. Do not
        # create the legacy fixed horizons or retry terminal FAILED tasks.
        with closing(sqlite3.connect(self.opportunities.config.database_path.as_uri()+"?mode=ro", uri=True)) as db:
            ids = [row[0] for row in db.execute("SELECT o.id FROM opportunities o WHERE NOT EXISTS "
                "(SELECT 1 FROM validation_tasks t WHERE t.opportunity_id=o.id) ORDER BY o.created_at,o.id LIMIT ?",
                (self.settings.batch_size,))]
        for identifier in ids:
            try:
                self.schedule(identifier)
            except Exception as error:
                self.logger.error(exact_json({"failure_stage": "PLANNED_TASK_RECONCILIATION",
                    "opportunity_id": identifier, "error_type": type(error).__name__}))
        def claim(pending: ValidationTask) -> ValidationTask | None:
            try:
                return self.tasks.mark_running(pending.id, now=self.clock(), lease_seconds=self.settings.lease_seconds)
            except Exception as error:
                self.logger.error(exact_json({"failure_stage": "PLANNED_TASK_EXECUTION",
                    "task_id": pending.id, "error_type": type(error).__name__}))
                return None
        def failure(task: ValidationTask, error: Exception) -> None:
            self._finish(task, TaskFailure.PERSISTENCE_FAILED, type(error).__name__)
        pending = self.tasks.list_due_tasks(self.clock(), limit=limit,
                                            max_lateness_seconds=self.settings.max_lateness_seconds)
        return process_task_batch(pending, claim=claim, execute=self._parallel_execute,
            failure=failure, stopped=lambda: self.stopped() or extra_stopped(), progress=self.progress)


class CandidateStudyRunner(ResearchRunner):
    def __init__(self, settings: Settings, *, clock: Callable[[], datetime] = utc_now,
                 stop_signal: StopSignal | None = None, logger: logging.Logger | None = None,
                 observation: Callable[[str], tuple[CandidateCatalog, CandidateObservation]] | None = None) -> None:
        index, studies = load_index(settings)
        research = settings.research
        assert research is not None
        sources = ArchivedCandidateSources(settings, studies, clock)
        scheduler = PlannedCandidateScheduler(settings, studies, sources, clock=clock)
        unbound = settings.model_copy(update={"research": research.model_copy(update={"study_registration": None})})
        super().__init__(unbound, sources, scheduler.opportunities, clock=clock, scheduler=scheduler,
                         stop_signal=stop_signal, logger=logger)
        self.settings, self.index, self.studies = settings, index, studies
        self.study_registry = StrategyStudyRegistry(research.study_registration.path.parent, clock=clock)  # type: ignore[union-attr]
        self.observation = observation or self._observe

    def set_progress(self, progress: Callable[[], None]) -> None:
        assert isinstance(self.scheduler, PlannedCandidateScheduler)
        self.scheduler.progress = progress

    def _observe(self, symbol: str) -> tuple[CandidateCatalog, CandidateObservation]:
        research = self.settings.research
        assert research is not None and research.candidate_catalog is not None and research.validation_scheduler is not None
        from crypto_arbitrage_bot.market_data.candidate_market import CandidateMarketCollector
        catalog = collect_catalog(self.settings, spot_instrument=ResearchInstrument(symbol, symbol.removesuffix("USDT")))
        directory = self.studies[symbol].path.parent / "entry-metadata"
        directory.mkdir(exist_ok=True)
        write_public_archive(directory / f"{uuid4().hex}.json", {"symbol": symbol, "recorded_at": self.clock(),
            "registration_sha256": self.studies[symbol].sha256, "catalog": catalog})
        collector = CandidateMarketCollector(catalog, research.candidate_catalog, research.data_quality,
            timeout_seconds=research.timeout_seconds, funding_page_limit=research.validation_scheduler.funding_page_limit,
            funding_max_pages=research.validation_scheduler.funding_max_pages)
        return catalog, collector.collect(symbol)

    def _candidate_cycle(self, result: CycleResult) -> CycleResult:
        symbol = result.symbol
        research = self.settings.research
        assert (research is not None and research.candidate_economics is not None
                and research.candidate_catalog is not None and self.study_registry is not None)
        self.study = study = self.studies[symbol]
        stage = CycleStage.STUDY_REGISTRATION
        try:
            self.study_registry.verify_policy(study, self.settings)
            stage = CycleStage.COLLECTION
            catalog, observed = self.observation(symbol)
            directory = study.path.parent / "entry-evidence"
            directory.mkdir(exist_ok=True)
            write_public_archive(directory / f"{result.cycle_id}.json", {
                "cycle_id": result.cycle_id, "symbol": symbol, "registration_sha256": study.sha256,
                "recorded_at": self.clock(), "catalog": catalog, "observation": observed})
            context = json.loads(study.candidate_context_json or "null")
            if (observed.assembly.symbol != symbol or exact_json(observed.candidate) != exact_json(context["candidate"])):
                raise ValueError("FROZEN_CANDIDATE_RULES_OR_IDENTITY_CHANGED")
            stage = CycleStage.DATA_QUALITY
            opened = utc_timestamp(self.clock())
            quality = validate_market_data(observed.assembly, research.data_quality, evaluated_at=opened)
            result = replace(result, data_quality=quality)
            if not quality.valid:
                return self._finish(replace(result, failure_stage=stage, failure_reason="INVALID_CANDIDATE_DATA_QUALITY"))
            stage = CycleStage.SCENARIO_SETUP
            profile = research.candidate_economics.windows[context["profile"]]
            projection = profile.funding_projection
            if projection is not None:
                assert observed.assembly.snapshot is not None
                next_at = observed.assembly.snapshot.next_funding_time
                lead = None if next_at is None else next_at-opened
                if lead is None or not timedelta(seconds=projection.min_entry_lead_seconds) <= lead <= timedelta(seconds=projection.entry_window_seconds):
                    raise ValueError("OUTSIDE_PREDECLARED_NEXT_SETTLEMENT_ENTRY_WINDOW")
            stage = CycleStage.ESTIMATE
            comparison, = compare_candidate_windows(catalog, research.candidate_catalog, observed,
                research.candidate_economics, self.settings.fees.as_rates(), research.data_quality,
                research.decision, evaluated_at=opened)
            stage = CycleStage.PERSISTENCE
            if not study.protocol.study_start <= opened <= utc_timestamp(self.clock()) < study.protocol.study_end:
                raise ValueError("STUDY_ENTRY_WINDOW_ENDED_DURING_COLLECTION")
            self.repository.initialize()
            saved = build_and_save_opportunity(self.repository, comparison.scenarios[0].estimate, comparison.decision,
                created_at=self.clock(), formula_version=research.formula_version,
                config_version=study.protocol.config_version)
            if saved.status is not PersistenceStatus.SAVED or saved.record is None or saved.saved_id is None:
                raise ValueError("CANDIDATE_PERSISTENCE_FAILED")
            result = replace(result, status=CycleStatus.SAVED, opportunity_id=saved.saved_id,
                             decision=comparison.decision.status.value)
            stage = CycleStage.STUDY_REGISTRATION
            try:
                self.study_registry.capture(study, saved.record, self.settings)
            except StudyArchiveError:
                result = replace(result, status=CycleStatus.FAILED, failure_stage=stage, failure_reason="CAPTURE_RECEIPT_FAILED")
            stage = CycleStage.SCHEDULING
            assert self.scheduler is not None
            self.scheduler.schedule(saved.saved_id)
        except KeyboardInterrupt:
            self.stop()
            result = replace(result, status=CycleStatus.INTERRUPTED, failure_stage=stage, failure_reason="INTERRUPTED")
        except Exception as error:
            result = replace(result, status=CycleStatus.FAILED, failure_stage=stage, failure_reason=type(error).__name__)
        return self._finish(result)

    def run_once(self) -> tuple[CycleResult, ...]:
        assert self.scheduler is not None
        index, studies = load_index(self.settings)
        if index != self.index or studies != self.studies:
            raise ValueError("PINNED_CANDIDATE_SCOPE_CHANGED")
        at = utc_timestamp(self.clock())
        protocol = next(iter(studies.values())).protocol
        if at >= protocol.study_end+protocol.max_holding_period+protocol.max_exit_lateness:
            self.stop()
            return ()
        assert isinstance(self.scheduler, PlannedCandidateScheduler)
        self.scheduler.drain_due()
        at = utc_timestamp(self.clock())
        if self.stop_signal.is_set():
            return ()
        results = []
        if protocol.study_start <= at < protocol.study_end:
            elapsed = at-protocol.study_start
            micros = (elapsed.days*86400+elapsed.seconds)*1_000_000+elapsed.microseconds
            slot = micros//(index["entry_interval_seconds"]*1_000_000)
            # Slot markers persist before IO. Restart cannot repeat a consumed
            # entry round, including a partial/crashed four-symbol round.
            ref = self.settings.research.study_registration  # type: ignore[union-attr]
            assert ref is not None
            directory = ref.path.with_suffix(".rounds")
            directory.mkdir(exist_ok=True)
            marker = directory / f"{slot:06d}.json"
            try:
                write_new(marker, {"slot": slot, "started_at": at, "symbols": index["symbols"]})
            except FileExistsError:
                with marker.open("rb") as stream:
                    body = stream.read(256001)
                value = json.loads(body)
                stamped = utc_timestamp(datetime.fromisoformat(value["started_at"]))
                if (len(body) > 256000 or exact_json(value).encode() != body
                        or set(value) != {"slot", "started_at", "symbols"} or type(value["slot"]) is not int
                        or value["slot"] != slot or value["symbols"] != index["symbols"]
                        or not protocol.study_start+timedelta(seconds=slot*index["entry_interval_seconds"])
                            <= stamped < min(protocol.study_end, protocol.study_start+timedelta(
                                seconds=(slot+1)*index["entry_interval_seconds"]))):
                    raise ValueError("CORRUPT_CONSUMED_CANDIDATE_ROUND") from None
                return ()
            planned = [CycleResult(uuid4().hex, symbol, utc_timestamp(self.clock())) for symbol in index["symbols"]]
            for result in planned:
                start_attempt(studies[result.symbol], result.cycle_id, result.symbol, result.started_at,
                              recorded_at=self.clock())
                self._active_study_attempts.add(result.cycle_id)
            for result in planned:
                symbol = result.symbol
                if self.stop_signal.is_set() or not protocol.study_start <= utc_timestamp(self.clock()) < protocol.study_end:
                    self.study = studies[symbol]
                    results.append(self._finish(replace(result, status=CycleStatus.INTERRUPTED,
                        failure_stage=CycleStage.COLLECTION, failure_reason="STOP_OR_WINDOW_END_BEFORE_IO")))
                else:
                    results.append(self._candidate_cycle(result))
        self.scheduler.drain_due()
        return tuple(results)


def prepare_candidate_index(settings: Settings, directory: Path, *, symbols: tuple[str, ...],
                            id_prefix: str, study_start: datetime, study_end: datetime,
                            min_entry_separation: timedelta, min_entries: int, min_days: int,
                            min_mean_return: Decimal, min_median_return: Decimal, max_loss_return: Decimal,
                            max_capture_delay: timedelta, max_sampling_gap: timedelta,
                            entry_interval_seconds: int, clock: Callable[[], datetime] = utc_now,
                            catalog: CandidateCatalog | None = None) -> tuple[Path, str]:
    """Fix intent before metadata IO, then register all subjects before entry IO.

    Reference metadata resolves exact lot rules; no price/funding ranking,
    Opportunity or retrospective receipt is used for selecting this cohort.
    """
    research = settings.research
    started = utc_timestamp(clock())
    study_start, study_end = utc_timestamp(study_start), utc_timestamp(study_end)
    if (research is None or settings.testnet is not None or research.study_registration is not None
            or research.candidate_catalog is None or research.candidate_economics is None
            or research.validation_scheduler is None or research.validation_scheduler.max_lateness_seconds is None
            or len(research.candidate_economics.windows) != 1 or research.sqlite_path.exists()
            or research.sqlite_path.with_suffix(research.sqlite_path.suffix+".research-job.json").exists()
            or not isinstance(symbols, tuple) or not 1 <= len(symbols) <= 10 or len(set(symbols)) != len(symbols)
            or directory.exists() or not started <= study_start < study_end
            or type(entry_interval_seconds) is not int or not 1 <= entry_interval_seconds <= 86400
            or timedelta(seconds=entry_interval_seconds) > max_sampling_gap):
        raise ValueError("NEW_FIXED_PUBLIC_CANDIDATE_STUDY_REQUIRED")
    for symbol in symbols:
        ResearchInstrument(symbol, symbol.removesuffix("USDT"))
    profile, = research.candidate_economics.windows
    assumptions = research.candidate_economics.windows[profile]
    holding = assumptions.holding_seconds if assumptions.funding_projection is None else (
        assumptions.funding_projection.max_wait_seconds+assumptions.funding_projection.exit_delay_seconds)
    lateness = research.validation_scheduler.max_lateness_seconds
    # Validate all criterion types/window feasibility before the first request.
    declarations = [StrategyStudyProtocol(f"{id_prefix}:{symbol}:{profile}", started, study_start, study_end,
        (symbol,), research.formula_version, f"{research.config_version}:{profile}", ((symbol, "0"*64),),
        timedelta(seconds=holding), min_entry_separation, timedelta(seconds=lateness),
        min_entries, min_days, min_mean_return, min_median_return, max_loss_return) for symbol in symbols]
    if (study_end-study_start+timedelta(seconds=holding+lateness) > timedelta(days=8)
            or study_start+min_entry_separation*(min_entries-1) >= study_end
            or ((study_end-timedelta(microseconds=1)).date()-study_start.date()).days+1 < min_days):
        raise ValueError("BOUNDED_FEASIBLE_CANDIDATE_WINDOW_REQUIRED")
    if not timedelta(0) < max_sampling_gap <= study_end-study_start or max_capture_delay < timedelta(0):
        raise ValueError("EXPLICIT_CAPTURE_AND_SAMPLING_LIMITS_REQUIRED")
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "intent.json", {"declared_at": started, "symbols": symbols,
        "protocols": declarations, "policy_sha256": digest(full_research_policy(settings)),
        "database_path": str(research.sqlite_path.resolve()), "profile": profile,
        "entry_interval_seconds": entry_interval_seconds, "max_sampling_gap": max_sampling_gap,
        "max_capture_delay": max_capture_delay, "orders_submitted": False})
    catalog = collect_catalog(settings) if catalog is None else catalog
    catalog_sha = write_new(directory / "reference-catalog.json", catalog)
    derived = build_candidate_catalog(*catalog.sources, research.candidate_catalog, now=utc_timestamp(clock()))
    if (catalog.status is not CatalogStatus.COMPLETE or derived.status is not CatalogStatus.COMPLETE
            or derived.candidates != catalog.candidates or any(s not in {c.symbol for c in derived.candidates} for s in symbols)):
        raise ValueError("ALL_PREDECLARED_CANDIDATES_MUST_QUALIFY")
    registry = StrategyStudyRegistry(directory, clock=clock)
    studies = {}
    for declaration in declarations:
        symbol, = declaration.symbols
        candidate = next(c for c in derived.candidates if c.symbol == symbol)
        protocol = replace(declaration, config_fingerprints=((symbol, fingerprint(settings, candidate)),))
        saved = register_candidate(registry, protocol, settings, candidate, profile=profile,
            catalog_sha256=catalog_sha, max_capture_delay=max_capture_delay, max_sampling_gap=max_sampling_gap)
        studies[symbol] = {"path": str(saved.path), "sha256": saved.sha256}
    index = {"schema_version": INDEX_VERSION, "symbols": symbols,
             "policy_sha256": digest(full_research_policy(settings)), "database_path": str(research.sqlite_path.resolve()),
             "catalog_sha256": catalog_sha, "entry_interval_seconds": entry_interval_seconds, "studies": studies}
    path = directory / "index.json"
    return path.resolve(), write_new(path, index)
