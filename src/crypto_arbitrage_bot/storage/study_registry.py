"""Exclusive local preregistration and capture receipts, never a trading permit.

Local hashes detect changes relative to a pinned reference. They are not an
external timestamp, signature, or protection against an operator rewriting all
local evidence. Original SQLite research records are never changed here.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from enum import StrEnum
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.strategy_study import StrategyStudyProtocol
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

REGISTRY_VERSION = 1


class StudyArchiveFailure(StrEnum):
    INVALID_REGISTRATION = "INVALID_REGISTRATION"
    REGISTRATION_CONFLICT = "REGISTRATION_CONFLICT"
    REGISTRATION_NOT_PRIOR = "REGISTRATION_NOT_PRIOR"
    ARCHIVE_IO_ERROR = "ARCHIVE_IO_ERROR"
    ARCHIVE_CORRUPT = "ARCHIVE_CORRUPT"
    ARCHIVE_HASH_MISMATCH = "ARCHIVE_HASH_MISMATCH"
    POLICY_MISMATCH = "POLICY_MISMATCH"
    RECORD_OUTSIDE_STUDY = "RECORD_OUTSIDE_STUDY"
    RECORD_CONFIG_MISMATCH = "RECORD_CONFIG_MISMATCH"
    CAPTURE_TOO_LATE = "CAPTURE_TOO_LATE"
    RECEIPT_CONFLICT = "RECEIPT_CONFLICT"
    DATABASE_READ_FAILED = "DATABASE_READ_FAILED"
    ATTEMPT_LOG_CORRUPT = "ATTEMPT_LOG_CORRUPT"


class StudyArchiveError(RuntimeError):
    def __init__(self, code: StudyArchiveFailure) -> None:
        self.code = code
        # Stable, bounded diagnosis; do not include raw JSON, books or settings.
        super().__init__(code.value)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _duration(value: object) -> timedelta:
    if not isinstance(value, dict) or set(value) != {"microseconds"} or type(value["microseconds"]) is not int:
        raise ValueError("integer microseconds required")
    return timedelta(microseconds=value["microseconds"])


def _protocol(value: dict[str, Any]) -> StrategyStudyProtocol:
    data = dict(value)
    for name in ("registered_at", "study_start", "study_end"):
        data[name] = utc_timestamp(datetime.fromisoformat(data[name]))
    for name in ("max_holding_period", "min_entry_separation", "max_exit_lateness"):
        data[name] = _duration(data[name])
    for name in ("min_mean_net_return", "min_median_net_return", "max_loss_return"):
        if not isinstance(data[name], str):
            raise ValueError("Decimal text required")
        data[name] = Decimal(data[name])
    data["symbols"] = tuple(data["symbols"])
    data["config_fingerprints"] = tuple(tuple(item) for item in data["config_fingerprints"])
    return StrategyStudyProtocol(**data)


def full_research_policy(settings: Settings) -> str:
    """Pin complete research settings, including scenario/exit and sampling rules.

    Only output paths, verbosity and TESTNET configuration are excluded. The
    separate DB binding remains mandatory. Operational timeout/scheduler rules
    are intentionally retained: changing them requires a new study reference.
    """
    if not isinstance(settings, Settings) or settings.research is None:
        raise ValueError("explicit research Settings required")
    # The pinned reference is metadata rather than strategy policy; excluding
    # it avoids a self-referential hash when the Runner adopts its registration.
    research = settings.research.model_dump(exclude={"sqlite_path", "study_registration"})
    return exact_json({"symbols": settings.symbols, "poll_interval_seconds": settings.poll_interval_seconds,
                       "fees": settings.fees.model_dump(), "research": research})


def study_cost_fingerprints(settings: Settings) -> tuple[tuple[str, str], ...]:
    """Hash the existing estimate/decision configuration without recalculating PnL."""
    research = settings.research
    if research is None:
        raise ValueError("explicit research Settings required")
    return tuple((symbol, _hash(exact_json({
        "estimate": research.sizing[symbol].as_estimate(settings.fees.as_rates(), research.data_quality),
        "decision": research.decision,
    }))) for symbol in settings.symbols)


@dataclass(frozen=True)
class RegisteredStudy:
    path: Path
    sha256: str
    protocol: StrategyStudyProtocol
    registered_at: datetime
    database_path: Path
    policy_json: str
    max_capture_delay: timedelta
    max_sampling_gap: timedelta | None = None
    candidate_context_json: str | None = None

    @property
    def receipt_directory(self) -> Path:
        return self.path.with_suffix(".captures")


def load_registered_study(path: Path, expected_sha256: str) -> RegisteredStudy:
    """Read only a pinned registry; never adopt a modified file's new hash."""
    try:
        if (not isinstance(path, Path) or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ValueError("pinned SHA256 and path required")
        text = path.read_text(encoding="utf-8")
        if _hash(text) != expected_sha256:
            raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_HASH_MISMATCH)
        data = json.loads(text)
        if exact_json(data) != text or data["schema_version"] not in (1, 2, 3):
            raise ValueError("unsupported/noncanonical registry")
        expected_fields = {"schema_version", "protocol", "registered_at", "database_path",
                           "policy", "max_capture_delay"}
        if data["schema_version"] in (2, 3):
            expected_fields.add("max_sampling_gap")
        if data["schema_version"] == 3:
            expected_fields.add("candidate_context")
        if set(data) != expected_fields:
            raise ValueError("registry fields differ")
        protocol = _protocol(data["protocol"])
        registered = utc_timestamp(datetime.fromisoformat(data["registered_at"]))
        delay = _duration(data["max_capture_delay"])
        gap = _duration(data["max_sampling_gap"]) if data["schema_version"] in (2, 3) else None
        if gap is not None and not timedelta(0) < gap <= protocol.study_end-protocol.study_start:
            raise ValueError("positive predeclared sampling gap within study window required")
        database = Path(data["database_path"])
        if (not protocol.registered_at <= registered <= protocol.study_start or delay < timedelta(0)
                or not database.is_absolute() or database != database.resolve()):
            raise ValueError("invalid registry timing/path")
        return RegisteredStudy(path.resolve(), expected_sha256, protocol, registered,
                               database, exact_json(data["policy"]), delay, gap,
                               exact_json(data["candidate_context"]) if data["schema_version"] == 3 else None)
    except StudyArchiveError:
        raise
    except OSError as exc:
        raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_IO_ERROR) from exc
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError, DecimalException) as exc:
        raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_CORRUPT) from exc


def _exclusive_write(path: Path, text: str) -> None:
    # An interrupted partial write remains corrupt and fails closed. Never
    # delete/reset evidence or overwrite a conflicting pre-existing registration.
    with path.open("xb") as output:
        output.write(text.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


class StrategyStudyRegistry:
    def __init__(self, directory: Path, *, clock: Callable[[], datetime] = utc_now) -> None:
        self.directory = directory.resolve()
        self.clock = clock

    def verify_policy(self, study: RegisteredStudy, settings: Settings) -> RegisteredStudy:
        current = load_registered_study(study.path, study.sha256)
        if current.candidate_context_json is not None:
            from crypto_arbitrage_bot.candidate_study import verify_candidate_policy
            return verify_candidate_policy(current, settings)
        if (full_research_policy(settings) != current.policy_json or settings.research is None
                or settings.research.sqlite_path.resolve() != current.database_path
                or settings.symbols != current.protocol.symbols):
            raise StudyArchiveError(StudyArchiveFailure.POLICY_MISMATCH)
        return current

    def register(self, protocol: StrategyStudyProtocol, settings: Settings,
                 *, max_capture_delay: timedelta, max_sampling_gap: timedelta | None = None) -> RegisteredStudy:
        """Register before study_start; repeated identical calls preserve the receipt."""
        try:
            policy = full_research_policy(settings)
            research = settings.research
            assert research is not None
            if (not isinstance(protocol, StrategyStudyProtocol)
                    or not isinstance(max_capture_delay, timedelta) or max_capture_delay < timedelta(0)
                    or protocol.symbols != settings.symbols
                    or protocol.config_fingerprints != study_cost_fingerprints(settings)
                    or protocol.formula_version != research.formula_version
                    or protocol.config_version != research.config_version):
                raise StudyArchiveError(StudyArchiveFailure.INVALID_REGISTRATION)
            database = research.sqlite_path.resolve()
            if (max_sampling_gap is not None and (not isinstance(max_sampling_gap, timedelta)
                    or not timedelta(0) < max_sampling_gap <= protocol.study_end-protocol.study_start)):
                raise StudyArchiveError(StudyArchiveFailure.INVALID_REGISTRATION)
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (_hash(protocol.id) + ".json")
            if path.exists():
                prior_text = path.read_text(encoding="utf-8")
                prior = load_registered_study(path, _hash(prior_text))
                if (prior.protocol != protocol or prior.policy_json != policy
                        or prior.database_path != database or prior.max_capture_delay != max_capture_delay):
                    raise StudyArchiveError(StudyArchiveFailure.REGISTRATION_CONFLICT)
                if max_sampling_gap is not None and prior.max_sampling_gap != max_sampling_gap:
                    raise StudyArchiveError(StudyArchiveFailure.REGISTRATION_CONFLICT)
                return prior
            at = utc_timestamp(self.clock())
            if not protocol.registered_at <= at <= protocol.study_start:
                raise StudyArchiveError(StudyArchiveFailure.REGISTRATION_NOT_PRIOR)
            envelope = {"schema_version": 1 if max_sampling_gap is None else 2, "protocol": protocol,
                        "registered_at": at, "database_path": str(database),
                        "policy": json.loads(policy), "max_capture_delay": max_capture_delay}
            if max_sampling_gap is not None:
                envelope["max_sampling_gap"] = max_sampling_gap
            text = exact_json(envelope)
            _exclusive_write(path, text)
            saved = load_registered_study(path, _hash(text))
            if max_sampling_gap is not None:
                from crypto_arbitrage_bot.storage.study_attempts import create_attempt_head
                create_attempt_head(saved)
            return saved
        except StudyArchiveError:
            raise
        except FileExistsError as exc:
            # Racing registrations must explicitly reload the original reference.
            raise StudyArchiveError(StudyArchiveFailure.REGISTRATION_CONFLICT) from exc
        except OSError as exc:
            raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_IO_ERROR) from exc
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError, DecimalException) as exc:
            raise StudyArchiveError(StudyArchiveFailure.INVALID_REGISTRATION) from exc

    def capture(self, study: RegisteredStudy, record: OpportunityRecord, settings: Settings) -> Path:
        """Save an immediate record/policy hash receipt, not a duplicate order book.

        This does not assert that supplied settings were used by a trustworthy
        Runner. Integration must call it with the exact settings used in that
        cycle. Missing receipts are NOT repaired retrospectively after outcomes.
        """
        try:
            current = self.verify_policy(study, settings)
            at = utc_timestamp(self.clock())
            p = current.protocol
            if record.symbol not in p.symbols or not p.study_start <= record.created_at < p.study_end:
                raise StudyArchiveError(StudyArchiveFailure.RECORD_OUTSIDE_STUDY)
            if (record.config_fingerprint != dict(p.config_fingerprints)[record.symbol]
                    or _hash(record.config_json) != record.config_fingerprint
                    or record.formula_version != p.formula_version or record.config_version != p.config_version):
                raise StudyArchiveError(StudyArchiveFailure.RECORD_CONFIG_MISMATCH)
            directory = current.receipt_directory
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (_hash(record.id) + ".json")
            fixed = {"schema_version": REGISTRY_VERSION, "registration_sha256": current.sha256,
                     "opportunity_id": record.id, "record_sha256": _hash(exact_json(record)),
                     "policy_sha256": _hash(current.policy_json)}
            if path.exists():
                prior = json.loads(path.read_text(encoding="utf-8"))
                if any(prior.get(name) != value for name, value in fixed.items()):
                    raise StudyArchiveError(StudyArchiveFailure.RECEIPT_CONFLICT)
                # Verify its original timing; never repair a corrupt/late receipt.
                captured = utc_timestamp(datetime.fromisoformat(prior["captured_at"]))
                if not record.created_at <= captured <= record.created_at + current.max_capture_delay:
                    raise StudyArchiveError(StudyArchiveFailure.CAPTURE_TOO_LATE)
                return path
            if not current.registered_at <= record.created_at <= at <= record.created_at + current.max_capture_delay:
                raise StudyArchiveError(StudyArchiveFailure.CAPTURE_TOO_LATE)
            _exclusive_write(path, exact_json({**fixed, "captured_at": at}))
            return path
        except StudyArchiveError:
            raise
        except FileExistsError as exc:
            raise StudyArchiveError(StudyArchiveFailure.RECEIPT_CONFLICT) from exc
        except OSError as exc:
            raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_IO_ERROR) from exc
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError, DecimalException) as exc:
            raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_CORRUPT) from exc
