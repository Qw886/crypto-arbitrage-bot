"""One-shot persistence of already produced research, with separate failure states."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from crypto_arbitrage_bot.models.decision import DecisionResult
from crypto_arbitrage_bot.models.estimate import OfflineEstimate
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.scenario import ScenarioResult
from crypto_arbitrage_bot.models.scenario_decision import ScenarioDecisionResult
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity
from crypto_arbitrage_bot.storage.record_validation import RecordBuildError, RecordBuildReason
from crypto_arbitrage_bot.storage.sqlite_repository import RepositoryError, RepositoryOperation


class OpportunityWriter(Protocol):
    """An initialized repository; expected storage failures raise RepositoryError."""

    def save_opportunity(self, record: OpportunityRecord) -> str: ...


class PersistenceStatus(StrEnum):
    SAVED = "SAVED"
    BUILD_FAILED = "BUILD_FAILED"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"


@dataclass(frozen=True)
class PersistenceFailure:
    message: str
    reason_codes: tuple[RecordBuildReason, ...] = ()
    operation: RepositoryOperation | None = None
    sqlite_errorcode: int | None = None
    sqlite_errorname: str | None = None


@dataclass(frozen=True)
class PersistenceResult:
    """record on write failure is unsaved evidence, never a new Decision."""

    status: PersistenceStatus
    record: OpportunityRecord | None
    saved_id: str | None = None
    failure: PersistenceFailure | None = None


def build_and_save_opportunity(
    repository: OpportunityWriter,
    estimate: OfflineEstimate,
    decision: DecisionResult | ScenarioDecisionResult,
    *, scenarios: tuple[ScenarioResult, ...] = (), created_at: datetime | None = None,
    opportunity_id: str | None = None, formula_version: str | None = None,
    config_version: str | None = None,
) -> PersistenceResult:
    """Map, validate, then save once. No initialization, retries or research calls.

    Versions are caller-supplied metadata; absent versions remain None. A
    persistence failure retains the built record and original decision for
    inspection. Input errors prevent any repository call.
    """
    try:
        record = build_opportunity(
            estimate, decision, scenarios=scenarios, created_at=created_at,
            opportunity_id=opportunity_id, formula_version=formula_version,
            config_version=config_version,
        )
    except (TypeError, ValueError) as exc:
        codes = exc.reason_codes if isinstance(exc, RecordBuildError) else (
            RecordBuildReason.INVALID_INPUT,
        )
        return PersistenceResult(PersistenceStatus.BUILD_FAILED, None,
                                 failure=PersistenceFailure(str(exc), codes))
    try:
        saved_id = repository.save_opportunity(record)
    except RepositoryError as exc:
        return PersistenceResult(
            PersistenceStatus.PERSISTENCE_FAILED, record,
            failure=PersistenceFailure(
                str(exc), operation=exc.operation, sqlite_errorcode=exc.sqlite_errorcode,
                sqlite_errorname=exc.sqlite_errorname,
            ),
        )
    return PersistenceResult(PersistenceStatus.SAVED, record, saved_id=saved_id)
