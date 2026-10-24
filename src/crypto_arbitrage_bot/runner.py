"""Read-only research orchestration; no order, position or account operations."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from threading import Event
from typing import Literal, Protocol
from uuid import uuid4

from crypto_arbitrage_bot.arbitrage.basis import mid_price
from crypto_arbitrage_bot.arbitrage.funding_projection import projected_scenario_definitions
from crypto_arbitrage_bot.arbitrage.scenario import (
    evaluate_scenario,
    evaluate_scenarios,
    reverse_execution_scenario,
    target_basis_scenario,
)
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.funding_history import BinanceFundingHistoryClient
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.market_data.snapshot import SnapshotCollector
from crypto_arbitrage_bot.models.data_quality import DataQualityResult
from crypto_arbitrage_bot.models.scenario import ExitBasisAssumption, ScenarioType
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.persistence import PersistenceStatus, build_and_save_opportunity
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository
from crypto_arbitrage_bot.storage.study_attempts import finish_attempt, start_attempt
from crypto_arbitrage_bot.storage.study_registry import (
    RegisteredStudy,
    StrategyStudyRegistry,
    load_registered_study,
)
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.logging import configure_logging
from crypto_arbitrage_bot.validation_scheduler import ValidationScheduler


class Collector(Protocol):
    def collect(self, symbol: str) -> SnapshotAssembly: ...


class StopSignal(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def wait(self, timeout: float | None = None) -> bool: ...


class CycleStage(StrEnum):
    COLLECTION = "COLLECTION"
    DATA_QUALITY = "DATA_QUALITY"
    SCENARIO_SETUP = "SCENARIO_SETUP"
    ESTIMATE = "ESTIMATE"
    SCENARIOS = "SCENARIOS"
    DECISION = "DECISION"
    PERSISTENCE = "PERSISTENCE"
    LOGGING = "LOGGING"
    SCHEDULING = "SCHEDULING"
    STUDY_REGISTRATION = "STUDY_REGISTRATION"


class CycleStatus(StrEnum):
    SAVED = "SAVED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    symbol: str
    started_at: datetime
    ended_at: datetime | None = None
    status: CycleStatus = CycleStatus.FAILED
    data_quality: DataQualityResult | None = None
    decision: str | None = None
    opportunity_id: str | None = None
    failure_stage: CycleStage | None = None
    failure_reason: str | None = None


class ResearchRunner:
    def __init__(
        self, settings: Settings, collector: Collector, repository: SQLiteOpportunityRepository,
        *, clock: Callable[[], datetime] = utc_now, stop_signal: StopSignal | None = None,
        logger: logging.Logger | None = None,
        scheduler: ValidationScheduler | None = None,
    ) -> None:
        if settings.research is None:
            raise ValueError("explicit [research] settings are required for the runner")
        research = settings.research
        for symbol in settings.symbols:
            if symbol not in research.sizing or symbol not in research.scenarios:
                raise ValueError(f"missing research sizing/scenario for {symbol}")
            research.sizing[symbol].as_estimate(settings.fees.as_rates(), research.data_quality)
            if any(event.symbol != symbol for event in research.scenarios[symbol].funding_events):
                raise ValueError("configured funding event symbol must match its scenario")
        if repository.config.database_path != research.sqlite_path.resolve():
            raise ValueError("repository path must match research.sqlite_path")
        self.settings = settings
        self.collector = collector
        self.repository = repository
        self.clock = clock
        self.stop_signal = Event() if stop_signal is None else stop_signal
        self.logger = logging.getLogger("crypto_arbitrage_bot.runner") if logger is None else logger
        self.scheduler = scheduler
        self.study: RegisteredStudy | None = None
        self.study_registry: StrategyStudyRegistry | None = None
        self._active_study_attempts: set[str] = set()
        reference = research.study_registration
        if reference is not None:
            self.study_registry = StrategyStudyRegistry(reference.path.parent, clock=clock)
            self.study = self.study_registry.verify_policy(
                load_registered_study(reference.path, reference.sha256), settings)
        if scheduler is not None:
            if scheduler.opportunities.config.database_path != repository.config.database_path:
                raise ValueError("scheduler must share the runner database")
            scheduler.stopped = self.stop_signal.is_set

    def stop(self) -> None:
        self.stop_signal.set()

    def set_progress(self, progress: Callable[[], None]) -> None:
        """Optional controller pulse; ordinary scanner keeps its current cadence."""

    def _finish(self, result: CycleResult) -> CycleResult:
        result = replace(result, ended_at=utc_timestamp(self.clock()))
        try:
            self.logger.info(exact_json(result))
        except Exception as exc:
            result = replace(result, status=CycleStatus.FAILED, failure_stage=CycleStage.LOGGING,
                             failure_reason=f"{type(exc).__name__}: {exc}; prior failure: {result.failure_reason}")
        if result.cycle_id in self._active_study_attempts:
            try:
                assert self.study is not None and result.ended_at is not None
                record = None if result.opportunity_id is None else self.repository.get_opportunity_by_id(result.opportunity_id)
                if result.opportunity_id is not None and record is None:
                    raise ValueError("SAVED_RECORD_UNAVAILABLE")
                codes = () if result.data_quality is None else tuple(code.value for code in result.data_quality.reason_codes)
                if result.failure_stage is not None:
                    codes = (*codes, result.failure_stage.value + "_FAILED")
                finish_attempt(self.study, result.cycle_id, result.symbol, ended_at=result.ended_at,
                               status=result.status.value,
                               failure_stage=None if result.failure_stage is None else result.failure_stage.value,
                               reason_codes=codes, record=record, recorded_at=utc_timestamp(self.clock()))
            except Exception as exc:
                # Preserve the saved research/Decision. A missing terminal
                # event blocks qualification, never becomes a trading REJECT.
                result = replace(result, status=CycleStatus.FAILED, failure_stage=CycleStage.STUDY_REGISTRATION,
                                 failure_reason="ATTEMPT_CAPTURE_FAILED:" + type(exc).__name__)
            finally:
                self._active_study_attempts.discard(result.cycle_id)
        return result

    def _run_symbol(self, symbol: Literal["BTCUSDT", "ETHUSDT"]) -> CycleResult:
        result = CycleResult(str(uuid4()), symbol, utc_timestamp(self.clock()))
        stage = CycleStage.COLLECTION
        research = self.settings.research
        assert research is not None
        try:
            if self.study is not None:
                stage = CycleStage.STUDY_REGISTRATION
                assert self.study_registry is not None
                self.study_registry.verify_policy(self.study, self.settings)
                if not self.study.protocol.study_start <= result.started_at < self.study.protocol.study_end:
                    raise ValueError("OUTSIDE_REGISTERED_STUDY_WINDOW")
                if self.study.max_sampling_gap is not None:
                    start_attempt(self.study, result.cycle_id, symbol, result.started_at,
                                  recorded_at=utc_timestamp(self.clock()))
                    self._active_study_attempts.add(result.cycle_id)
            stage = CycleStage.COLLECTION
            assembly = self.collector.collect(symbol)
            if assembly.symbol != symbol or not assembly.complete:
                return self._finish(replace(result, failure_stage=stage,
                                           failure_reason=exact_json({"issues": assembly.issues, "source_errors": {
                                               name: getattr(getattr(assembly.sources, name), "error", None)
                                               for name in ("spot_depth", "perp_depth", "premium_index", "open_interest")
                                           }}) if assembly.symbol == symbol
                                           else "COLLECTED_SYMBOL_MISMATCH"))
            stage = CycleStage.DATA_QUALITY
            opened = utc_timestamp(self.clock())
            quality = validate_market_data(assembly, research.data_quality, evaluated_at=opened)
            result = replace(result, data_quality=quality)
            if not quality.valid:
                return self._finish(replace(result, failure_stage=stage,
                                           failure_reason=exact_json(quality.reason_codes)))
            stage = CycleStage.SCENARIO_SETUP
            snapshot = assembly.snapshot
            assert snapshot is not None
            assumptions = research.scenarios[symbol]
            closed = opened + timedelta(seconds=assumptions.holding_seconds)
            projected_definitions = None
            if assumptions.funding_projection is not None:
                projected_definitions = projected_scenario_definitions(snapshot, opened, assumptions)
            baseline = projected_definitions[0] if projected_definitions is not None else reverse_execution_scenario(
                name="baseline", spot_entry_book=snapshot.spot_book, perp_entry_book=snapshot.perp_book,
                opened_at=opened, closed_at=closed, funding_events=assumptions.funding_events,
                description="Configured unchanged-depth research exit; no basis convergence assumed.",
            )
            estimate_config = research.sizing[symbol].as_estimate(self.settings.fees.as_rates(), research.data_quality)
            stage = CycleStage.ESTIMATE
            # evaluate_scenario delegates once to 5D; do not evaluate baseline a second time.
            base_result = evaluate_scenario(assembly, estimate_config, baseline, evaluated_at=opened)
            if not base_result.complete:
                return self._finish(replace(result, failure_stage=stage,
                                           failure_reason=exact_json((base_result.estimate.status, base_result.estimate.issues))))
            stage = CycleStage.SCENARIOS
            spot_mid = assumptions.spot_exit_mid
            if assumptions.spot_exit_mid_mode == "entry_mid":
                assert snapshot.spot_book.best_bid is not None and snapshot.spot_book.best_ask is not None
                spot_mid = mid_price(snapshot.spot_book.best_bid, snapshot.spot_book.best_ask)
            assert spot_mid is not None
            definitions = projected_definitions[1:] if projected_definitions is not None else tuple(target_basis_scenario(
                scenario_type=kind, name=kind.value,
                assumption=ExitBasisAssumption(spot_mid, basis, snapshot.spot_book, snapshot.perp_book),
                opened_at=opened, closed_at=closed, funding_events=assumptions.funding_events,
                description="Configured hypothetical target basis and exit liquidity; not a prediction.",
            ) for kind, basis in (
                (ScenarioType.CONVERGENCE, assumptions.convergence_exit_basis_pct),
                (ScenarioType.ADVERSE, assumptions.adverse_exit_basis_pct),
            ))
            results = (base_result, *evaluate_scenarios(assembly, estimate_config, definitions, evaluated_at=opened))
            if any(not item.complete for item in results):
                return self._finish(replace(result, failure_stage=stage, failure_reason=exact_json(
                    tuple((item.name, item.estimate.status, item.estimate.issues) for item in results if not item.complete))))
            stage = CycleStage.DECISION
            decision = decide_scenarios(results, research.decision)
            result = replace(result, decision=decision.status.value)
            stage = CycleStage.PERSISTENCE
            self.repository.initialize()
            saved = build_and_save_opportunity(
                self.repository, base_result.estimate, decision, created_at=utc_timestamp(self.clock()),
                formula_version=research.formula_version, config_version=research.config_version,
            )
            if saved.status is not PersistenceStatus.SAVED:
                return self._finish(replace(result, failure_stage=stage, failure_reason=exact_json(saved.failure)))
            result = replace(result, status=CycleStatus.SAVED, opportunity_id=saved.saved_id)
            if self.study is not None:
                stage = CycleStage.STUDY_REGISTRATION
                assert self.study_registry is not None and saved.record is not None
                try:
                    self.study_registry.capture(self.study, saved.record, self.settings)
                except Exception as exc:
                    # A saved opportunity remains saved. Missing receipts block
                    # evidence review; do not fabricate a trading REJECT or
                    # prevent its original forward tasks from being scheduled.
                    result = replace(result, status=CycleStatus.FAILED, failure_stage=stage,
                                     failure_reason=f"{type(exc).__name__}: {exc}")
            if self.scheduler is not None:
                stage = CycleStage.SCHEDULING
                assert saved.saved_id is not None
                self.scheduler.schedule(saved.saved_id)
        except KeyboardInterrupt:
            self.stop()
            return self._finish(replace(result, status=CycleStatus.INTERRUPTED,
                                       failure_stage=stage, failure_reason="KeyboardInterrupt"))
        except Exception as exc:
            # The orchestration boundary isolates unexpected library/storage failures
            # by symbol; typed adapter/model diagnostics are retained above.
            result = replace(result, status=CycleStatus.FAILED, failure_stage=stage, failure_reason=f"{type(exc).__name__}: {exc}")
        return self._finish(result)

    def run_once(self) -> tuple[CycleResult, ...]:
        if self.study is not None:
            at = utc_timestamp(self.clock())
            protocol = self.study.protocol
            if at >= protocol.study_end+protocol.max_holding_period+protocol.max_exit_lateness:
                self.stop()
                return ()
            if not protocol.study_start <= at < protocol.study_end:
                # No new samples outside the fixed window. Keep processing due
                # validations during the explicitly bounded closing allowance.
                if self.scheduler is not None and not self.stop_signal.is_set():
                    try:
                        self.scheduler.process_due()
                    except KeyboardInterrupt:
                        self.stop()
                    except Exception as exc:
                        self.logger.error(exact_json({"failure_stage": "VALIDATION_SCHEDULER", "reason": str(exc)}))
                return ()
        results = []
        for symbol in self.settings.symbols:
            if self.stop_signal.is_set():
                break
            results.append(self._run_symbol(symbol))
        if self.scheduler is not None and not self.stop_signal.is_set():
            try:
                self.scheduler.process_due()
            except KeyboardInterrupt:
                self.stop()
            except Exception as exc:
                self.logger.error(exact_json({"failure_stage": "VALIDATION_SCHEDULER", "reason": str(exc)}))
        return tuple(results)

    def run_forever(self) -> None:
        """Wait the configured interval after every round, even if every cycle failed.

        Event.wait permits prompt stop during idle time. An in-flight public GET
        uses the adapter's bounded socket timeout; stop takes effect after it returns.
        """
        try:
            while not self.stop_signal.is_set():
                self.run_once()
                if self.stop_signal.is_set() or self.stop_signal.wait(self.settings.poll_interval_seconds):
                    break
        except KeyboardInterrupt:
            self.stop()


def create_runner(settings: Settings) -> ResearchRunner:
    """Construct only whitelisted public clients; construction sends no requests."""
    if settings.research is None:
        raise ValueError("explicit [research] settings are required for the runner")
    reference = settings.research.study_registration
    if reference is not None:
        from crypto_arbitrage_bot.candidate_study import CandidateStudyRunner, is_candidate_index
        if is_candidate_index(reference.path, reference.sha256):
            return CandidateStudyRunner(settings)
    timeout = settings.research.timeout_seconds
    collector = SnapshotCollector(BinanceSpotClient(timeout_seconds=timeout), BinanceUsdtmClient(timeout_seconds=timeout))
    repository = SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path))
    scheduler = None
    validation_settings = settings.research.validation_scheduler
    if validation_settings is not None:
        scheduler = ValidationScheduler(
            validation_settings, repository, SQLiteForwardValidationRepository(repository.config),
            SQLiteValidationTaskRepository(repository.config), collector,
            BinanceFundingHistoryClient(timeout_seconds=timeout, page_limit=validation_settings.funding_page_limit,
                                        max_pages=validation_settings.funding_max_pages),
        )
    return ResearchRunner(settings, collector, repository,
                          scheduler=scheduler,
                          logger=configure_logging(settings.log_dir, settings.log_level))
