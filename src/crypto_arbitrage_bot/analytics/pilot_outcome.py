"""Frozen multisettlement-observation-v1 outcome; never strategy admission."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from enum import StrEnum
from typing import Any

from crypto_arbitrage_bot.analytics.validation_evidence import (
    VALIDATED_METRICS,
    read_validation_attribution,
    require_validation_amounts,
)
from crypto_arbitrage_bot.models.validation_record import ForwardValidationRecord
from crypto_arbitrage_bot.models.validation_task import TaskStatus, ValidationTask
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp

PILOT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "THEUSDT", "TSTUSDT")
PILOT_CANDIDATES = frozenset({"THEUSDT", "TSTUSDT"})
PILOT_HORIZONS = (86400, 259200, 604800)
PILOT_LATENESS = timedelta(seconds=300)
ORIGINAL_METRICS = frozenset({"capital_occupied", "original_entry_basis", "spot_entry_fee",
                              "perp_entry_fee", "other_costs", "original_estimated_net_profit",
                              "original_estimated_net_return"})


class PilotConclusion(StrEnum):
    PENDING_FINAL_OBSERVATIONS = "PENDING_FINAL_OBSERVATIONS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_GO_FOR_CURRENT_PILOT = "NO_GO_FOR_CURRENT_PILOT"
    EXPLORATORY_SIGNAL = "EXPLORATORY_SIGNAL"


class PilotReason(StrEnum):
    MISSING_TASK = "MISSING_TASK"
    DUPLICATE_TASK = "DUPLICATE_TASK"
    UNEXPECTED_SCOPE = "UNEXPECTED_SCOPE"
    TASK_NOT_COMPLETED = "TASK_NOT_COMPLETED"
    MISSING_VALIDATION = "MISSING_VALIDATION"
    DUPLICATE_VALIDATION = "DUPLICATE_VALIDATION"
    INVALID_IDENTITY = "INVALID_IDENTITY"
    INVALID_VALIDATION = "INVALID_VALIDATION"
    INVALID_TIMING = "INVALID_TIMING"
    LATE_VALIDATION = "LATE_VALIDATION"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    MISSING_OR_INVALID_METRIC = "MISSING_OR_INVALID_METRIC"
    INVALID_STORED_DETAILS = "INVALID_STORED_DETAILS"
    ORIGINAL_ENTRY_MISMATCH = "ORIGINAL_ENTRY_MISMATCH"


@dataclass(frozen=True)
class PilotEntry:
    symbol: str
    opportunity_id: str
    created_at: datetime
    entry_time: datetime
    quantity: Decimal
    original_metrics: tuple[tuple[str, Decimal], ...] = ()

    def __post_init__(self) -> None:
        if self.symbol not in PILOT_SYMBOLS or not self.opportunity_id:
            raise ValueError("original pilot symbol and opportunity identity required")
        for name in ("created_at", "entry_time"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name)))
        if self.entry_time > self.created_at:
            raise ValueError("entry must precede its record")
        require_decimal(self.quantity, "original matched quantity", positive=True)
        if (type(self.original_metrics) is not tuple
                or self.original_metrics and (len(self.original_metrics) != len(ORIGINAL_METRICS)
                    or {name for name, _ in self.original_metrics} != ORIGINAL_METRICS)):
            raise ValueError("complete immutable original amount references required")
        for name, value in self.original_metrics:
            require_decimal(value, name, positive=name == "capital_occupied",
                            nonnegative=name in ("spot_entry_fee", "perp_entry_fee", "other_costs"))


def assess_pilot_v1(entries: tuple[PilotEntry, ...], tasks: tuple[ValidationTask, ...],
                    validations: tuple[ForwardValidationRecord, ...], *, evaluated_at: datetime,
                    deadline: datetime) -> dict[str, Any]:
    """Consume saved outcomes without cash-flow recalculation or return errors.

    Same-entry horizons are correlated observations, not independent trades.
    A next-settlement forecast is never compared as a seven-day prediction.
    """
    at, end = utc_timestamp(evaluated_at), utc_timestamp(deadline)
    if (type(entries) is not tuple or len(entries) != 4
            or {e.symbol for e in entries} != set(PILOT_SYMBOLS)
            or len({e.opportunity_id for e in entries}) != 4
            or any(e.created_at > at or e.created_at + timedelta(seconds=max(PILOT_HORIZONS)) > end for e in entries)):
        raise ValueError("the four original past entries and unchanged deadline are required")
    expected = {(e.opportunity_id, h) for e in entries for h in PILOT_HORIZONS}
    unexpected_tasks = tuple(t.id for t in tasks if (t.opportunity_id, t.horizon_seconds) not in expected)
    linked_ids = {t.validation_id for t in tasks if (t.opportunity_id, t.horizon_seconds) in expected}
    unexpected_validations = tuple(v.id for v in validations if v.id not in linked_ids)
    rows: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda e: PILOT_SYMBOLS.index(e.symbol)):
        for horizon in PILOT_HORIZONS:
            due = entry.created_at + timedelta(seconds=horizon)
            matches = [t for t in tasks if t.opportunity_id == entry.opportunity_id and t.horizon_seconds == horizon]
            task = matches[0] if len(matches) == 1 else None
            reasons = []
            attribution: dict[str, Any] = {}
            if task is None:
                reasons.append(PilotReason.MISSING_TASK if not matches else PilotReason.DUPLICATE_TASK)
            elif (task.symbol != entry.symbol or task.due_at != due
                  or task.created_at > at or task.updated_at > at):
                reasons.append(PilotReason.INVALID_IDENTITY)
            if task is not None and not entry.created_at <= task.created_at <= task.updated_at <= at:
                reasons.append(PilotReason.INVALID_TIMING)
            if task is not None and task.status is not TaskStatus.COMPLETED:
                reasons.append(PilotReason.TASK_NOT_COMPLETED)
            if task is not None and task.failure_code is not None:
                reasons.append(PilotReason.TASK_NOT_COMPLETED)
            saved = [] if task is None else [v for v in validations if v.id == task.validation_id]
            value = saved[0] if len(saved) == 1 else None
            if value is None:
                reasons.append(PilotReason.MISSING_VALIDATION if not saved else PilotReason.DUPLICATE_VALIDATION)
            else:
                if value.symbol != entry.symbol or value.opportunity_id != entry.opportunity_id:
                    reasons.append(PilotReason.INVALID_IDENTITY)
                if value.status != "VALIDATED" or value.reason_codes:
                    reasons.append(PilotReason.INVALID_VALIDATION)
                if (value.created_at > at or value.entry_time != entry.entry_time or value.validation_time is None
                        or value.holding_duration is None or value.validation_time > at
                        or value.validation_time < due or value.validation_time <= entry.entry_time
                        or value.holding_duration != value.validation_time-entry.entry_time):
                    reasons.append(PilotReason.INVALID_TIMING)
                elif value.validation_time - due > PILOT_LATENESS:
                    reasons.append(PilotReason.LATE_VALIDATION)
                if value.matched_quantity != entry.quantity:
                    reasons.append(PilotReason.QUANTITY_MISMATCH)
                if any(getattr(value, name) != original for name, original in entry.original_metrics):
                    reasons.append(PilotReason.ORIGINAL_ENTRY_MISMATCH)
                if (value.validation_time is not None and
                        not value.validation_time <= value.created_at <= at
                        or task is not None and value.created_at > task.updated_at):
                    reasons.append(PilotReason.INVALID_TIMING)
                try:
                    require_validation_amounts(value)
                except (ValueError, TypeError, DecimalException):
                    reasons.append(PilotReason.MISSING_OR_INVALID_METRIC)
                try:
                    attribution = read_validation_attribution(value)
                except (ValueError, TypeError, KeyError, DecimalException, OverflowError):
                    reasons.append(PilotReason.INVALID_STORED_DETAILS)
            usable = not reasons
            rows.append({"symbol": entry.symbol, "opportunity_id": entry.opportunity_id,
                "intended_horizon_seconds": horizon, "due_at": due,
                "task_status": None if task is None else task.status.value,
                "task_failure_code": None if task is None else task.failure_code,
                "validation_id": None if task is None else task.validation_id,
                "validation_status": None if value is None else value.status,
                "entry_time": entry.entry_time,
                "validation_time": None if value is None else value.validation_time,
                "holding_duration": None if value is None else value.holding_duration,
                "usable": usable, "reason_codes": tuple(dict.fromkeys(reasons)),
                **{name: attribution.get(name) if usable else None for name in
                   ("funding_event_count", "mid_basis_pnl", "spread_costs", "slippage_costs", "rounding_residual")},
                **{name: getattr(value, name) if usable else None for name in VALIDATED_METRICS}})
    usable_count = sum(bool(r["usable"]) for r in rows)
    complete = usable_count == 12 and not unexpected_tasks and not unexpected_validations
    terminal = all(r["task_status"] in ("COMPLETED", "FAILED") for r in rows)
    final = terminal or at >= end
    candidate_positive = sum(r["usable"] and r["symbol"] in PILOT_CANDIDATES
                             and r["validated_simulated_net_profit"] > 0 for r in rows)
    benchmark_positive = sum(r["usable"] and r["symbol"] not in PILOT_CANDIDATES
                             and r["validated_simulated_net_profit"] > 0 for r in rows)
    conclusion = (PilotConclusion.INSUFFICIENT_EVIDENCE if not complete else
                  PilotConclusion.EXPLORATORY_SIGNAL if candidate_positive else PilotConclusion.NO_GO_FOR_CURRENT_PILOT)
    return {"protocol_version": "multisettlement-observation-v1",
        "scope": "EXPLORATORY_SIMULATED_FORWARD_OUTCOMES_NOT_LIVE_PROFIT_OR_STRATEGY_QUALIFICATION",
        "evaluated_at": at, "original_deadline": end, "phase": "FINAL" if final else "WAITING_OBSERVATIONS",
        "conclusion": conclusion if final else PilotConclusion.PENDING_FINAL_OBSERVATIONS,
        "provisional_conclusion": conclusion, "expected_observations": 12, "original_entry_count": 4,
        "usable_validations": usable_count, "coverage_complete": complete,
        "task_status_counts": dict(Counter(t.status.value for t in tasks)),
        "unexpected_task_ids": unexpected_tasks, "unexpected_validation_ids": unexpected_validations,
        "candidate_positive_outcomes": candidate_positive, "benchmark_positive_outcomes": benchmark_positive,
        "rows": rows, "strategy_qualified": False, "orders_submitted": False,
        "meaning": "Simulated observed exits only. Four correlated entries are not twelve independent trades. "
                   "Missing outcomes are not zero. No same-horizon prediction errors, win rate or trading authorization."}
