"""Optional witness for the original controlled settlement acceptance.

No order submission, cost calculation, fault injection or budget override here.
Evidence capture failure blocks a new entry but cannot block existing exits.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.execution.risk_witness import (
    RiskWitness,
    WitnessError,
    WitnessFailure,
    load_witness,
)
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance


class AcceptanceCaptureStatus(StrEnum):
    REGISTERED = "REGISTERED"
    STOPPED = "STOPPED"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


def _save(
    acceptance: "FundingAcceptance",
    status: AcceptanceCaptureStatus,
    code: str | None = None,
    **fields: object,
) -> None:
    acceptance._save(
        str(acceptance.state["status"]), witness_status=status.value, witness_failure=code, **fields
    )


def _claimed(acceptance: "FundingAcceptance") -> bool:
    return acceptance.bot.journal.get(str(acceptance.state.get("attempt_id", ""))) is not None


def _setup(acceptance: "FundingAcceptance", directory: Path | None,
           case: ExecutionRiskCase = ExecutionRiskCase.FEES_FUNDING_BALANCE) -> RiskWitness:
    bot = acceptance.bot
    reference = acceptance.state.get("witness_registration")
    if reference is not None:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise WitnessError(WitnessFailure.REGISTRATION_CONFLICT)
        pinned = StudyRegistrationSettings(path=reference["path"], sha256=reference["sha256"])
        registration = load_witness(pinned)
        if (
            registration.attempt_id != acceptance.state["attempt_id"]
            or registration.symbol != acceptance.symbol
            or registration.journal_path != bot.config.journal_path.resolve()
            or registration.case is not case
            or (
                directory is not None
                and registration.reference.path.parent.parent != directory.resolve()
            )
        ):
            raise WitnessError(WitnessFailure.REGISTRATION_CONFLICT)
        return RiskWitness(registration, bot.settings, clock=bot.clock)
    if directory is None or _claimed(acceptance):
        # Never attach new evidence to an already submitted attempt.
        raise WitnessError(WitnessFailure.ATTEMPT_ALREADY_EXISTS)
    budget = (
        acceptance.max_wait_seconds
        + acceptance.entry_lead_seconds
        + acceptance.policy.exit_delay_seconds
        + 300
        + bot.config.funding_check_max_attempts * bot.config.funding_check_interval_seconds
    )
    witness = RiskWitness.register(
        directory,
        bot.settings,
        bot.spot,
        bot.perp,
        attempt_id=str(acceptance.state["attempt_id"]),
        symbol=acceptance.symbol,
        case=case,
        duration=timedelta(seconds=budget),
        clock=bot.clock,
    )
    _save(
        acceptance,
        AcceptanceCaptureStatus.REGISTERED,
        witness_registration={
            "path": str(witness.registration.reference.path),
            "sha256": witness.registration.reference.sha256,
        },
        witness_origin=witness.registration.origin,
        capture_case=case.value,
        witness_expires_at=witness.registration.expires_at,
    )
    return witness


@contextmanager
def observe_acceptance(acceptance: "FundingAcceptance", directory: Path | None, *,
                       case: ExecutionRiskCase = ExecutionRiskCase.FEES_FUNDING_BALANCE) -> Iterator[None]:
    """Use under the original journal lock; one session across all waiting ticks.

    A graceful stop exports no final proof. Resume reuses the original reference
    and expiry. Interrupted/broken capture is never repaired into qualification.
    """
    if directory is None and "witness_registration" not in acceptance.state:
        yield
        return
    if acceptance.bot.stop.is_set():
        yield
        return
    terminal = {"VERIFIED", "UNAVAILABLE", "ABORTED", "NEEDS_ATTENTION", "CANCELLED"}
    if acceptance.state.get("status") in terminal:
        # A finished report is immutable evidence of that run; no new session
        # or retrospective registration on an already terminal acceptance.
        yield
        return
    if not acceptance.state:
        try:
            acceptance.tick()  # Original arming/check/budget guards; submits no orders.
        except Exception as error:
            acceptance._handle_failure(error)
            yield
            return
    if acceptance.state.get("status") in terminal:
        yield
        return
    observer = None
    try:
        if acceptance.state.get("witness_status") == AcceptanceCaptureStatus.FAILED.value:
            raise WitnessError(WitnessFailure.CORRUPT_TRACE)
        witness = _setup(acceptance, directory, case)
        if witness.broken:
            raise WitnessError(WitnessFailure.CORRUPT_TRACE)
        observer = witness.observe(acceptance.bot.spot, acceptance.bot.perp)
        observer.__enter__()
    except Exception as error:
        code = (
            error.code.value if isinstance(error, WitnessError) else WitnessFailure.IO_ERROR.value
        )
        try:
            _save(acceptance, AcceptanceCaptureStatus.FAILED, code)
        except Exception:
            pass  # Evidence write failure must not suppress an existing risk-reducing exit.
        if not _claimed(acceptance):
            acceptance._save("ABORTED", reason="WITNESS_UNAVAILABLE_BEFORE_ENTRY")
        # For an existing claim, the original acceptance only manages that
        # attempt; WAITING recovers HOLDING without creating another entry.
        yield
        return
    try:
        yield
    except BaseException as error:
        assert observer is not None
        observer.__exit__(type(error), error, error.__traceback__)
        raise
    else:
        assert observer is not None
        observer.__exit__(None, None, None)
        try:
            if witness.broken:
                raise WitnessError(WitnessFailure.CORRUPT_TRACE)
            if acceptance.state.get("status") in terminal:
                path = witness.registration.reference.path.parent / "trace.json"
                reference = witness.export_trace(path)
                _save(
                    acceptance,
                    AcceptanceCaptureStatus.EXPORTED,
                    witness_trace={"path": str(reference.path), "sha256": reference.sha256},
                    witness_finished_at=utc_timestamp(acceptance.bot.clock()),
                )
            else:
                _save(acceptance, AcceptanceCaptureStatus.STOPPED)
        except Exception as error:
            code = (
                error.code.value
                if isinstance(error, WitnessError)
                else WitnessFailure.IO_ERROR.value
            )
            try:
                _save(acceptance, AcceptanceCaptureStatus.FAILED, code)
            except Exception:
                pass
