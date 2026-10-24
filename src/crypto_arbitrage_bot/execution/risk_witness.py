"""Bounded, redacted, preregistered observation of controlled TESTNET runs.

This tool never creates orders or calls an account endpoint itself. Its context
observes existing gateway calls made by the controlled acceptance coordinator.
Injected transports are always marked OFFLINE_FIXTURE, not native evidence.
"""

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.restart_capture import execution_snapshot, replay_terminal
from crypto_arbitrage_bot.execution.risk_review import (
    execution_policy_fingerprint,
    implementation_fingerprint,
)
from crypto_arbitrage_bot.execution.runner import process_lock
from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind,
    TestnetExecutionClient,
    TestnetExecutionError,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder, OrderReceipt
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.native_identity import (
    NativeIdentity,
    NativeState,
    current_native_identity,
    observe_native_identity,
)

MAX_EVENTS = 200
MAX_REGISTRATION_BYTES = 256000
MAX_EVENT_BYTES = 16384


class WitnessFailure(StrEnum):
    INVALID_REGISTRATION = "INVALID_WITNESS_REGISTRATION"
    ATTEMPT_ALREADY_EXISTS = "WITNESS_ATTEMPT_ALREADY_EXISTS"
    REGISTRATION_CONFLICT = "WITNESS_REGISTRATION_CONFLICT"
    IO_ERROR = "WITNESS_IO_ERROR"
    CORRUPT_TRACE = "CORRUPT_WITNESS_TRACE"
    POLICY_CHANGED = "WITNESS_POLICY_CHANGED"
    EXPIRED = "WITNESS_EXPIRED"
    EVENT_LIMIT = "WITNESS_EVENT_LIMIT"


class WitnessError(RuntimeError):
    def __init__(self, code: WitnessFailure) -> None:
        self.code = code
        super().__init__(code.value)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_once(path: Path, text: str) -> None:
    with path.open("xb") as output:
        output.write(text.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _receipt(value: OrderReceipt) -> dict[str, Any]:
    # Identity/quantity/state only. No signed URLs, headers, API messages,
    # raw account data, trades, keys or secret objects enter this trace.
    return {"market": value.market, "symbol": value.symbol, "order_id": value.order_id,
            "client_order_id": value.client_order_id, "status": value.status,
            "original_quantity": value.original_quantity, "executed_quantity": value.executed_quantity}


@dataclass(frozen=True)
class WitnessRegistration:
    reference: StudyRegistrationSettings
    attempt_id: str
    symbol: str
    case: ExecutionRiskCase
    created_at: datetime
    expires_at: datetime
    origin: str
    journal_path: Path
    policy_sha256: str
    implementation_sha256: str

    @property
    def events_directory(self) -> Path:
        return self.reference.path.parent / "events"


def load_witness(reference: StudyRegistrationSettings) -> WitnessRegistration:
    try:
        if reference.path.is_symlink():
            raise ValueError("regular witness registration required")
        with reference.path.open("rb") as stream:
            raw = stream.read(MAX_REGISTRATION_BYTES + 1)
        if len(raw) > MAX_REGISTRATION_BYTES:
            raise ValueError("registration byte limit exceeded")
        text = raw.decode("utf-8")
        if _digest(text) != reference.sha256:
            raise WitnessError(WitnessFailure.REGISTRATION_CONFLICT)
        data = json.loads(text)
        if exact_json(data) != text or data["schema_version"] != 1 or data["environment"] != "TESTNET":
            raise ValueError("versioned exact TESTNET registration required")
        created, expires = (_time(data[n]) for n in ("created_at", "expires_at"))
        if (not created < expires or data["origin"] not in ("NATIVE_TESTNET", "OFFLINE_FIXTURE")
                or data["symbol"] not in ("BTCUSDT", "ETHUSDT")):
            raise ValueError("invalid witness registration")
        return WitnessRegistration(reference, data["attempt_id"], data["symbol"], ExecutionRiskCase(data["case"]),
                                   created, expires, data["origin"], Path(data["journal_path"]),
                                   data["policy_sha256"], data["implementation_sha256"])
    except WitnessError:
        raise
    except OSError as exc:
        raise WitnessError(WitnessFailure.IO_ERROR) from exc
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise WitnessError(WitnessFailure.CORRUPT_TRACE) from exc


def _time(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


def read_witness_events(registration: WitnessRegistration) -> tuple[dict[str, Any], ...]:
    try:
        directory = registration.events_directory
        if directory.is_symlink():
            raise ValueError("regular event directory required")
        if not directory.exists():
            return ()
        paths: list[Path] = []
        for path in directory.iterdir():
            if len(paths) >= MAX_EVENTS:
                raise WitnessError(WitnessFailure.EVENT_LIMIT)
            paths.append(path)
        paths.sort()
        previous = registration.reference.sha256
        events = []
        for index, path in enumerate(paths):
            if path.name != f"{index:05d}.json" or path.is_symlink() or not path.is_file():
                raise ValueError("missing/out-of-sequence event")
            with path.open("rb") as stream:
                raw = stream.read(MAX_EVENT_BYTES + 1)
            if len(raw) > MAX_EVENT_BYTES:
                raise ValueError("event byte limit exceeded")
            text = raw.decode("utf-8")
            event = json.loads(text)
            if (exact_json(event) != text or event["sequence"] != index or event["previous_sha256"] != previous
                    or event["registration_sha256"] != registration.reference.sha256
                    or not registration.created_at <= _time(event["at"]) < registration.expires_at):
                raise ValueError("altered trace chain")
            previous = _digest(text)
            events.append(event)
        return tuple(events)
    except WitnessError:
        raise
    except OSError as exc:
        raise WitnessError(WitnessFailure.IO_ERROR) from exc
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise WitnessError(WitnessFailure.CORRUPT_TRACE) from exc


class RiskWitness:
    def __init__(self, registration: WitnessRegistration, settings: Settings,
                 *, clock: Callable[[], datetime] = utc_now) -> None:
        self.registration = registration
        self.settings = settings
        self.clock = clock
        prior = read_witness_events(registration)
        self.broken = (any(e.get("capture_complete") is False for e in prior)
                       or sum(e["kind"] == "SESSION_BEGIN" for e in prior) != sum(e["kind"] == "SESSION_END" for e in prior))
        self._verify()

    @classmethod
    def register(cls, directory: Path, settings: Settings, spot: TestnetExecutionClient,
                 perp: TestnetExecutionClient, *, attempt_id: str, symbol: str,
                 case: ExecutionRiskCase, duration: timedelta,
                 clock: Callable[[], datetime] = utc_now) -> "RiskWitness":
        try:
            if (settings.testnet is None or not isinstance(case, ExecutionRiskCase)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attempt_id)
                    or symbol not in settings.symbols or not isinstance(duration, timedelta)
                    or not timedelta(0) < duration <= timedelta(days=1)
                    or spot.market is not ExecutionMarket.SPOT or perp.market is not ExecutionMarket.USDTM):
                raise WitnessError(WitnessFailure.INVALID_REGISTRATION)
            journal = settings.testnet.journal_path.resolve()
            # Do not create, initialize or rewrite the execution journal.
            with closing(sqlite3.connect(journal.as_uri()+"?mode=ro", uri=True, timeout=0)) as db:
                if db.execute("SELECT 1 FROM testnet_pair_attempts WHERE attempt_id=?", (attempt_id,)).fetchone():
                    raise WitnessError(WitnessFailure.ATTEMPT_ALREADY_EXISTS)
            folder = directory.resolve() / _digest(attempt_id+":"+case.value)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "registration.json"
            at = utc_timestamp(clock())
            origin = "NATIVE_TESTNET" if spot.uses_default_transport and perp.uses_default_transport else "OFFLINE_FIXTURE"
            text = exact_json({"schema_version": 1, "environment": "TESTNET", "attempt_id": attempt_id,
                "symbol": symbol, "case": case, "created_at": at, "expires_at": at+duration,
                "journal_path": str(journal), "origin": origin, "policy_sha256": execution_policy_fingerprint(settings),
                "implementation_sha256": implementation_fingerprint()})
            _write_once(path, text)
            reference = StudyRegistrationSettings(path=path, sha256=_digest(text))
            return cls(load_witness(reference), settings, clock=clock)
        except WitnessError:
            raise
        except FileExistsError as exc:
            raise WitnessError(WitnessFailure.REGISTRATION_CONFLICT) from exc
        except (OSError, sqlite3.Error) as exc:
            raise WitnessError(WitnessFailure.IO_ERROR) from exc
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise WitnessError(WitnessFailure.INVALID_REGISTRATION) from exc

    def _verify(self) -> None:
        current = load_witness(self.registration.reference)
        if (current.policy_sha256 != execution_policy_fingerprint(self.settings)
                or current.implementation_sha256 != implementation_fingerprint()):
            raise WitnessError(WitnessFailure.POLICY_CHANGED)
        if not current.created_at <= utc_timestamp(self.clock()) < current.expires_at:
            raise WitnessError(WitnessFailure.EXPIRED)

    def _event(self, kind: str, **values: Any) -> None:
        self._verify()
        previous = read_witness_events(self.registration)
        if len(previous) >= MAX_EVENTS:
            raise WitnessError(WitnessFailure.EVENT_LIMIT)
        directory = self.registration.events_directory
        directory.mkdir(parents=True, exist_ok=True)
        prior = self.registration.reference.sha256 if not previous else _digest(exact_json(previous[-1]))
        text = exact_json({"sequence": len(previous), "previous_sha256": prior,
            "registration_sha256": self.registration.reference.sha256, "at": utc_timestamp(self.clock()),
            "kind": kind, **values})
        if len(text.encode("utf-8")) > MAX_EVENT_BYTES:
            raise WitnessError(WitnessFailure.CORRUPT_TRACE)
        _write_once(directory / f"{len(previous):05d}.json", text)

    def _safe_event(self, kind: str, **values: Any) -> None:
        try:
            self._event(kind, **values)
        except (WitnessError, OSError, ValueError, TypeError):
            self.broken = True

    def _entry_belongs(self, order: IocLimitOrder) -> bool:
        try:
            with closing(sqlite3.connect(self.registration.journal_path.as_uri()+"?mode=ro", uri=True, timeout=0)) as db:
                row = db.execute("SELECT attempt_id FROM testnet_order_intents WHERE client_order_id=?",
                                 (order.client_order_id,)).fetchone()
                return row is not None and row[0] == self.registration.attempt_id
        except sqlite3.Error:
            return False

    @contextmanager
    def observe(self, spot: TestnetExecutionClient, perp: TestnetExecutionClient,
                *, after_spot_ack: Callable[[OrderReceipt], None] | None = None) -> Iterator[None]:
        """Observe one controlled run; never submit an order on context entry/exit.

        Trace failure before a BUY Spot/SELL Perp blocks that entry. Trace failure
        during a reducing order cannot prevent risk reduction or hide its receipt.
        A broken/incomplete trace must not be finalized as engineering proof.
        """
        self._verify()
        native = spot.uses_default_transport and perp.uses_default_transport
        if native != (self.registration.origin == "NATIVE_TESTNET"):
            raise WitnessError(WitnessFailure.REGISTRATION_CONFLICT)
        originals = []
        with process_lock(self.registration.reference.path.with_suffix(".lock")):
            identity = (current_native_identity() if self.registration.case is ExecutionRiskCase.RESTART_IDEMPOTENCY else None)
            self._event("SESSION_BEGIN", **({} if identity is None else {"process_identity": identity.payload()}))
            try:
                for client in (spot, perp):
                    original_place, original_get = client.place_ioc_limit_order, client.get_order
                    def place(order: IocLimitOrder, original: Callable[[IocLimitOrder], OrderReceipt] = original_place) -> OrderReceipt:
                        entry = (order.market is ExecutionMarket.SPOT and order.side.value == "BUY"
                                 or order.market is ExecutionMarket.USDTM and order.side.value == "SELL")
                        if entry and (order.symbol != self.registration.symbol or not self._entry_belongs(order)):
                            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED, "witness symbol mismatch")
                        if order.symbol != self.registration.symbol:
                            self.broken = True  # Observation cannot block another reducing leg.
                        self._safe_event("POST_BEGIN", market=order.market, client_order_id=order.client_order_id,
                                         symbol=order.symbol, side=order.side, quantity=order.quantity)
                        if entry and self.broken:
                            raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED, "witness unavailable before entry")
                        try:
                            receipt = original(order)
                        except TestnetExecutionError as error:
                            self._safe_event("POST_FAILED", market=order.market, client_order_id=order.client_order_id,
                                             diagnostic=error.diagnostic_fields())
                            raise
                        except BaseException as error:
                            self.broken = True
                            self._safe_event("POST_INTERRUPTED", market=order.market,
                                             client_order_id=order.client_order_id, error_type=type(error).__name__)
                            raise
                        self._safe_event("POST_RESULT", receipt=_receipt(receipt))
                        if entry and order.market is ExecutionMarket.SPOT and after_spot_ack is not None:
                            try:
                                after_spot_ack(receipt)
                            except Exception:
                                self._safe_event("LOCAL_ACK_DELIVERY_INTERRUPT", market=receipt.market,
                                                 client_order_id=receipt.client_order_id)
                                # The underlying gateway already returned an
                                # actual receipt. A local delivery fault must
                                # never be mislabeled as rejection before send.
                                raise TestnetExecutionError(ExecutionErrorKind.SUBMISSION_UNKNOWN,
                                    "controlled local acknowledgement delivery interrupted", market=receipt.market,
                                    submission_may_have_reached_exchange=True) from None
                        return receipt
                    def query(symbol: str, client_order_id: str, *, include_base_commission: bool = True,
                              original: Callable[..., OrderReceipt] = original_get) -> OrderReceipt:
                        self._safe_event("QUERY_BEGIN", symbol=symbol, client_order_id=client_order_id)
                        try:
                            receipt = original(symbol, client_order_id, include_base_commission=include_base_commission)
                        except TestnetExecutionError as error:
                            self._safe_event("QUERY_FAILED", client_order_id=client_order_id, diagnostic=error.diagnostic_fields())
                            raise
                        except BaseException as error:
                            self.broken = True
                            self._safe_event("QUERY_INTERRUPTED", client_order_id=client_order_id,
                                             error_type=type(error).__name__)
                            raise
                        self._safe_event("QUERY_RESULT", receipt=_receipt(receipt))
                        return receipt
                    originals.append((client, original_place, original_get))
                    client.place_ioc_limit_order = place  # type: ignore[method-assign]
                    client.get_order = query  # type: ignore[method-assign]
                try:
                    yield
                except BaseException as error:
                    self.broken = True
                    self._safe_event("SESSION_INTERRUPTED", error_type=type(error).__name__)
                    raise
            finally:
                for client, place_method, get_method in originals:
                    client.place_ioc_limit_order = place_method  # type: ignore[method-assign]
                    client.get_order = get_method  # type: ignore[method-assign]
                checkpoint: dict[str, Any] = {}
                if identity is not None:
                    try:
                        checkpoint = {"process_identity": identity.payload(), "terminal_snapshot": execution_snapshot(self.registration.journal_path, self.registration.attempt_id)}
                    except Exception:
                        self.broken = True
                self._safe_event("SESSION_END", capture_complete=not self.broken, **checkpoint)

    def capture_terminal_restart(self) -> dict[str, Any]:
        """Replay the original terminal reconcile in a new native process.

        Only the preregistered restart case is accepted. All database access is
        read-only and both gateways reject requests; no clients or credentials.
        A same-process session, live/unknown old owner or changed snapshot fails.
        """
        self._verify()
        if self.registration.case is not ExecutionRiskCase.RESTART_IDEMPOTENCY or self.broken:
            raise WitnessError(WitnessFailure.CORRUPT_TRACE)
        with process_lock(self.registration.reference.path.with_suffix(".lock")):
            try:
                prior = read_witness_events(self.registration)
                end = prior[-1]
                if end["kind"] != "SESSION_END" or end["capture_complete"] is not True:
                    raise ValueError("complete old terminal session required")
                previous = NativeIdentity.parse(end["process_identity"])
                current = current_native_identity()
                observed = observe_native_identity(previous)
                if (current == previous or not _time(end["at"]) <= current.created_at <= utc_timestamp(self.clock())
                        or observed not in (NativeState.MISSING, NativeState.EXITED, NativeState.REPLACED)):
                    raise ValueError("new native process and terminated prior owner required")
                before = execution_snapshot(self.registration.journal_path, self.registration.attempt_id)
                if before != end["terminal_snapshot"]:
                    raise ValueError("original execution snapshot changed")
            except Exception as exc:
                raise WitnessError(WitnessFailure.CORRUPT_TRACE) from exc
            self._event("SESSION_BEGIN", process_identity=current.payload(), phase="TERMINAL_RESTART")
            after = None
            try:
                self._event("TERMINAL_REPLAY_BEGIN", previous_process=previous.payload(), previous_observation=observed, before=before)
                result, calls = replay_terminal(self.registration.journal_path, self.registration.attempt_id)
                after = execution_snapshot(self.registration.journal_path, self.registration.attempt_id)
                if after != before:
                    raise ValueError("execution source changed during replay")
                self._event("TERMINAL_REPLAY_END", after=after, request_call_count=calls, result_state=result.state)
                return {"state": result.state.value, "request_call_count": calls, "origin": self.registration.origin}
            except Exception as exc:
                self.broken = True
                self._safe_event("SESSION_INTERRUPTED", error_type="TERMINAL_REPLAY_FAILED")
                raise WitnessError(WitnessFailure.CORRUPT_TRACE) from exc
            finally:
                self._safe_event("SESSION_END", capture_complete=not self.broken,
                    process_identity=current.payload(), terminal_snapshot=after)

    def export_trace(self, path: Path) -> StudyRegistrationSettings:
        """Immutable trace export, not an engineering approval or risk certificate."""
        try:
            events = read_witness_events(self.registration)
            if (self.broken or not events or events[-1]["kind"] != "SESSION_END"
                    or events[-1]["capture_complete"] is not True
                    or any(e.get("capture_complete") is False for e in events)):
                raise WitnessError(WitnessFailure.CORRUPT_TRACE)
            text = exact_json({"schema_version": 1, "registration": {
                               "path": str(self.registration.reference.path), "sha256": self.registration.reference.sha256},
                               "origin": self.registration.origin, "events": events})
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_once(path, text)
            return StudyRegistrationSettings(path=path.resolve(), sha256=_digest(text))
        except WitnessError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise WitnessError(WitnessFailure.IO_ERROR) from exc
