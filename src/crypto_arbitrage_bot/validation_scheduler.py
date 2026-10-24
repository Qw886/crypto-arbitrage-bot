"""Automatic forward research validation; orchestration only, never trading."""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.config.research import ValidationSchedulerSettings
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.models.validation_task import TaskFailure, ValidationTask
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
    ValidationPersistenceStatus,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteOpportunityRepository
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.failure_summary import (
    collection_failure_summary,
    funding_failure_summary,
    limit_failure_reason,
)


class Collector(Protocol):
    def collect(self, symbol: str) -> SnapshotAssembly: ...


class FundingProvider(Protocol):
    def fetch_window(self, symbol: str, opened_at: datetime, closed_at: datetime) -> FundingWindow: ...


class ValidationScheduler:
    def __init__(self, settings: ValidationSchedulerSettings, opportunities: SQLiteOpportunityRepository,
                 validations: SQLiteForwardValidationRepository, tasks: SQLiteValidationTaskRepository,
                 collector: Collector, funding: FundingProvider, *, clock: Callable[[], datetime] = utc_now,
                 stopped: Callable[[], bool] = lambda: False, logger: logging.Logger | None = None) -> None:
        if len({repo.config.database_path for repo in (opportunities, validations, tasks)}) != 1:
            raise ValueError("scheduler repositories must share one database")
        self.settings = settings
        self.opportunities = opportunities
        self.validations = validations
        self.tasks = tasks
        self.collector = collector
        self.funding = funding
        self.clock = clock
        self.stopped = stopped
        self.logger = logging.getLogger("crypto_arbitrage_bot.validation_scheduler") if logger is None else logger

    def initialize(self) -> None:
        self.opportunities.initialize()
        self.validations.initialize()
        self.tasks.initialize()

    def schedule(self, opportunity_id: str) -> None:
        self.initialize()
        self.tasks.create_tasks(opportunity_id, self.settings.validation_horizons_seconds, now=self.clock())

    def _finish(self, task: ValidationTask, failure: TaskFailure | None = None, reason: str | None = None) -> None:
        reason = limit_failure_reason(reason)
        try:
            updated = self.tasks.finish(task, now=self.clock(), failure=failure, reason=reason)
        except Exception as exc:
            self.logger.error(exact_json({"task_id": task.id, "failure": TaskFailure.PERSISTENCE_FAILED,
                                          "reason": str(exc), "pending_failure": failure,
                                          "pending_reason": reason, "recovery": "RUNNING lease expiry"}))
            return
        if not updated:
            self.logger.warning(exact_json({"task_id": task.id, "failure": "CLAIM_LOST"}))
        else:
            self.logger.info(exact_json(self.tasks.get(task.id)))

    def _execute(self, task: ValidationTask) -> None:
        stage = TaskFailure.PERSISTENCE_FAILED
        try:
            # Resolves a crash after validation INSERT but before task completion.
            saved = self.validations.get_forward_validation_by_id(task.validation_id)
            if saved is not None:
                if saved.opportunity_id != task.opportunity_id or saved.symbol != task.symbol:
                    raise ValueError("validation identity collision")
                self._finish(task, None if saved.status == ValidationStatus.VALIDATED else TaskFailure.VALIDATION_FAILED,
                             None if saved.status == ValidationStatus.VALIDATED else exact_json(saved.reason_codes))
                return
            if task.attempts > self.settings.max_attempts:
                self._finish(task, TaskFailure.RECOVERY_LIMIT_EXCEEDED, "bounded interrupted-attempt recovery exhausted")
                return
            if self._window_missed(task, utc_timestamp(self.clock())):
                return
            stage = TaskFailure.ENTRY_EVIDENCE_INVALID
            record = self.opportunities.get_opportunity_by_id(task.opportunity_id)
            if record is None or record.symbol != task.symbol:
                raise ValueError("missing or mismatched opportunity")
            entry = decode_entry(record)
            stage = TaskFailure.COLLECTION_FAILED
            snapshot = self.collector.collect(task.symbol)
            if snapshot.symbol != task.symbol or not snapshot.complete:
                self._finish(task, stage, collection_failure_summary(snapshot))
                return
            at = utc_timestamp(self.clock())
            # A slow collection (or OS suspend during it) can miss the deadline
            # even when the task was claimed on time. Never backdate the exit.
            if self._window_missed(task, at):
                return
            stage = TaskFailure.INVALID_EXIT_MARKET_DATA
            quality = validate_market_data(snapshot, entry.quality_config, evaluated_at=at)
            if not quality.valid:
                raise ValueError(exact_json(quality.reason_codes))
            stage = TaskFailure.FUNDING_UNAVAILABLE
            window = self.funding.fetch_window(task.symbol, entry.opened_at, at)
            if (window.failure is not None or window.symbol != task.symbol
                    or window.opened_at != entry.opened_at or window.closed_at != at):
                self._finish(task, stage, funding_failure_summary(window))
                return
            # A known scheduled settlement cannot disappear behind a successful
            # but empty/not-yet-published history response. Never substitute zero.
            expected = record.next_funding_time
            if expected is None:
                raise ValueError("original next funding time is missing")
            if any(e.symbol != task.symbol or not entry.opened_at < e.settled_at <= at for e in window.events):
                raise ValueError("funding event outside validation window or symbol")
            if entry.opened_at < expected <= at:
                tolerance = timedelta(milliseconds=self.settings.funding_time_tolerance_ms)
                matches = tuple(e for e in window.events if abs(e.settled_at - expected) <= tolerance)
                if len(matches) != 1:
                    # Diagnostic matching only: pass unchanged actual timestamps,
                    # marks and rates to the existing validator. Never round or
                    # manufacture a settlement or borrow an out-of-window event.
                    raise ValueError(exact_json({
                        "reason": "known due funding settlement absent or ambiguous",
                        "expected_at": expected, "opened_at": entry.opened_at, "closed_at": at,
                        "tolerance_ms": self.settings.funding_time_tolerance_ms,
                        "event_count": len(window.events),
                        "event_times": tuple(e.settled_at for e in window.events[:6]),
                        "page_count": len(window.pages),
                        "http_statuses": tuple(p.status_code for p in window.pages[:6]),
                    }))
            if self.stopped():
                self._finish(task, TaskFailure.INTERRUPTED, "shutdown before validation")
                return
            stage = TaskFailure.VALIDATION_FAILED
            result = validate_forward_outcome(record, snapshot, validation_time=at, funding_events=window.events)
            stage = TaskFailure.PERSISTENCE_FAILED
            current = self.tasks.get(task.id)
            if current is None or current.claim_token != task.claim_token:
                raise ValueError("task claim expired and was replaced before persistence")
            saved_result = self.validations.save_forward_validation(result, validation_id=task.validation_id, created_at=self.clock())
            if saved_result.status is not ValidationPersistenceStatus.SAVED:
                # A concurrent expired-lease worker may have inserted the same ID.
                existing = self.validations.get_forward_validation_by_id(task.validation_id)
                if existing is None or existing.opportunity_id != task.opportunity_id or existing.symbol != task.symbol:
                    raise ValueError(exact_json(saved_result.failure))
                status, reasons = existing.status, existing.reason_codes
            else:
                status, reasons = result.status, result.reason_codes
            self._finish(task, None if status == ValidationStatus.VALIDATED else TaskFailure.VALIDATION_FAILED,
                         None if status == ValidationStatus.VALIDATED else exact_json(reasons))
        except KeyboardInterrupt:
            # No final mark: lease recovery permits a restart to resume this task.
            raise
        except Exception as exc:
            try:
                self._finish(task, stage, f"{type(exc).__name__}: {exc}")
            except Exception as storage_exc:
                # A DB outage may prevent any durable status update. Leave RUNNING
                # for lease recovery and surface both errors; never claim success.
                self.logger.error(exact_json({"task_id": task.id, "failure": stage,
                                              "reason": str(exc), "status_write_failure": str(storage_exc)}))

    def _window_missed(self, task: ValidationTask, at: datetime) -> bool:
        limit = self.settings.max_lateness_seconds
        if limit is None or at - task.due_at <= timedelta(seconds=limit):
            return False
        self._finish(task, TaskFailure.VALIDATION_WINDOW_MISSED, exact_json({
            "due_at": task.due_at, "observed_at": at, "horizon_seconds": task.horizon_seconds,
            "lateness": at - task.due_at, "max_lateness_seconds": limit,
        }))
        return True

    def process_due(self) -> None:
        """One bounded batch, no retries/waits here; Runner owns polling cadence."""
        if self.stopped():
            return
        self.initialize()
        now = utc_timestamp(self.clock())
        self.tasks.reconcile(self.settings.validation_horizons_seconds, now=now, limit=self.settings.batch_size)
        for pending in self.tasks.list_due_tasks(
            now, limit=self.settings.batch_size, max_lateness_seconds=self.settings.max_lateness_seconds,
        ):
            if self.stopped():
                break
            try:
                task = self.tasks.mark_running(pending.id, now=self.clock(), lease_seconds=self.settings.lease_seconds)
                if task is not None:
                    self._execute(task)
            except Exception as exc:
                self.logger.error(exact_json({"task_id": pending.id, "failure": TaskFailure.PERSISTENCE_FAILED,
                                              "reason": str(exc)}))
