"""Durable pre-entry balance evidence, independent of order submission.

Before an execution journal parent exists, two immutable wallet snapshots may
be archived under a proposed attempt. Unused evidence never creates an order.
The caller must capture BEFORE obtaining its final executable market quote.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from crypto_arbitrage_bot.execution.account_balance_audit import (
    AssetBalance,
    BalanceSnapshot,
    capture_balance_snapshot,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionClient,
    TestnetExecutionError,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


class BalanceFrameCode(StrEnum):
    INVALID_FRAME = "INVALID_FRAME"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    CAPTURE_FAILED = "CAPTURE_FAILED"
    MISSING_FRAME = "MISSING_FRAME"


@dataclass(frozen=True)
class BeforeBalanceFrame:
    attempt_id: str
    symbol: str
    quantity: Decimal
    snapshots: tuple[BalanceSnapshot, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.attempt_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.attempt_id):
            raise ValueError("invalid balance frame identity")
        if self.symbol not in ("BTCUSDT", "ETHUSDT"):
            raise ValueError("unsupported frame symbol")
        require_decimal(self.quantity, "balance frame quantity", positive=True)
        if (type(self.schema_version) is not int or self.schema_version != 1 or not isinstance(self.snapshots, tuple)
                or len(self.snapshots) != 2 or any(not isinstance(s, BalanceSnapshot) for s in self.snapshots)
                or {s.market for s in self.snapshots} != set(ExecutionMarket)):
            raise ValueError("both typed testnet wallet snapshots required")


@dataclass(frozen=True)
class FrameResult:
    success: bool
    frame: BeforeBalanceFrame | None = None
    error_code: BalanceFrameCode | None = None
    error_kind: str | None = None
    diagnostics: dict[str, object] | None = None


class BalanceFrameStore:
    """Versioned UTF-8 exact JSON; immutable exclusive file creation.

    Decimal strings and UTC timestamps are decoded back to their original
    types. A corrupt/partial file blocks capture, never overwritten as if new.
    """

    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, attempt_id: str) -> Path:
        if type(attempt_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", attempt_id):
            raise ValueError("invalid balance frame identity")
        return self.directory / (attempt_id + ".before.json")

    def get(self, attempt_id: str, symbol: str, quantity: Decimal) -> FrameResult:
        try:
            path = self._path(attempt_id)
            require_decimal(quantity, "expected frame quantity", positive=True)
            if symbol not in ("BTCUSDT", "ETHUSDT"):
                raise ValueError("unsupported frame symbol")
            if not path.exists():
                return FrameResult(False, error_code=BalanceFrameCode.MISSING_FRAME)
            data = json.loads(path.read_text(encoding="utf-8"))
            if type(data["schema_version"]) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported balance frame version")
            snapshots = []
            for s in data["snapshots"]:
                if any(type(b["amount"]) is not str for b in s["balances"]):
                    raise ValueError("balance amounts must be decimal strings")
                snapshots.append(BalanceSnapshot(ExecutionMarket(s["market"]),
                    datetime.fromisoformat(s["requested_at"]), datetime.fromisoformat(s["received_at"]),
                    tuple(AssetBalance(b["asset"], Decimal(b["amount"])) for b in s["balances"]),
                    None if s["official_updated_at"] is None else datetime.fromisoformat(s["official_updated_at"]),
                    s["raw_balances_json"]))
            if type(data["quantity"]) is not str:
                raise ValueError("quantity must be decimal string")
            frame = BeforeBalanceFrame(data["attempt_id"], data["symbol"], Decimal(data["quantity"]),
                                       tuple(snapshots), data["schema_version"])
            if (frame.attempt_id, frame.symbol, frame.quantity) != (attempt_id, symbol, quantity):
                return FrameResult(False, error_code=BalanceFrameCode.IDENTITY_MISMATCH)
            return FrameResult(True, frame)
        except OSError:
            return FrameResult(False, error_code=BalanceFrameCode.PERSISTENCE_FAILED)
        except (ValueError, KeyError, TypeError, ArithmeticError):
            return FrameResult(False, error_code=BalanceFrameCode.INVALID_FRAME)

    def save(self, frame: BeforeBalanceFrame) -> FrameResult:
        encoded = exact_json(frame).encode("utf-8")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(frame.attempt_id)
            try:
                # Readers fail safely if they encounter a partially written
                # file. No order may proceed until this write has completed.
                with path.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                existing = self.get(frame.attempt_id, frame.symbol, frame.quantity)
                if not existing.success or existing.frame != frame:
                    return FrameResult(False, error_code=BalanceFrameCode.EVIDENCE_CONFLICT)
            return FrameResult(True, frame)
        except OSError:
            return FrameResult(False, error_code=BalanceFrameCode.PERSISTENCE_FAILED)

    def capture_before(self, attempt_id: str, symbol: str, quantity: Decimal,
                       spot: TestnetExecutionClient, perp: TestnetExecutionClient, *,
                       clock: Callable[[], datetime] = utc_now) -> FrameResult:
        """Restart reuses existing evidence; never invents a new 'before'.

        Only account GETs. Failure leaves entry authorization to the caller;
        this method cannot submit, retry, cancel or modify an order.
        """
        existing = self.get(attempt_id, symbol, quantity)
        if existing.error_code is not BalanceFrameCode.MISSING_FRAME:
            return existing
        try:
            if spot.market is not ExecutionMarket.SPOT or perp.market is not ExecutionMarket.USDTM:
                raise ValueError("wrong wallet clients")
            frames = tuple(capture_balance_snapshot(c, clock=clock) for c in (spot, perp))
            return self.save(BeforeBalanceFrame(attempt_id, symbol, quantity, frames))
        except Exception as error:
            # No arbitrary exception strings, account bodies or credentials.
            return FrameResult(False, error_code=BalanceFrameCode.CAPTURE_FAILED,
                               error_kind=type(error).__name__,
                               diagnostics=error.diagnostic_fields() if isinstance(error, TestnetExecutionError) else None)
