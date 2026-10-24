"""Frozen exploratory protocol does not issue a strategy qualification."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.analytics.pilot_outcome import PilotEntry, assess_pilot_v1
from crypto_arbitrage_bot.models.validation_record import ForwardValidationRecord
from crypto_arbitrage_bot.models.validation_task import TaskFailure, TaskStatus, ValidationTask
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

OPENED = datetime(2026, 10, 6, 3, 25, tzinfo=UTC)
DEADLINE = OPENED + timedelta(days=8)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "THEUSDT", "TSTUSDT")
HORIZONS = (86400, 259200, 604800)


def cohort():

    entries = tuple(
        PilotEntry(s, s, OPENED, OPENED - timedelta(milliseconds=10), Decimal(1)) for s in SYMBOLS
    )
    tasks = tuple(
        ValidationTask(
            f"{s}-{h}",
            s,
            s,
            h,
            OPENED + timedelta(seconds=h),
            TaskStatus.PENDING,
            f"{s}-{h}",
            OPENED,
            OPENED,
        )
        for s in SYMBOLS
        for h in HORIZONS
    )
    return entries, tasks


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("outcome reports must remain offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


def saved(entry, task, profit=Decimal("-1"), late=0):
    at = task.due_at + timedelta(seconds=late)
    numbers = {name: Decimal(0) for name in ForwardValidationRecord.DECIMAL_FIELDS}
    numbers.update(
        validated_simulated_net_profit=profit,
        validated_simulated_net_return=profit / 100,
        matched_quantity=entry.quantity,
        capital_occupied=Decimal(100),
    )
    value = ForwardValidationRecord(
        task.validation_id,
        entry.opportunity_id,
        entry.symbol,
        entry.entry_time,
        at,
        at - entry.entry_time,
        "VALIDATED",
        (),
        **numbers,
        created_at=at,
        details_json="{}",
    )
    details = {
        name: getattr(value, name)
        for name in (
            "symbol",
            "opportunity_id",
            "status",
            "reason_codes",
            "entry_time",
            "validation_time",
            "holding_duration",
            "matched_quantity",
            "capital_occupied",
            "spot_pnl",
            "perp_pnl",
            "validated_simulated_net_return",
            "original_entry_basis",
            "basis_change",
        )
    }
    details.update(
        exit_data_quality={"issues": []},
        calculation={
            "symbol": value.symbol,
            "net_profit": profit,
            "exit_basis": value.validated_exit_basis,
            "price_pnl": value.price_pnl,
            "funding_total": value.funding_total,
            "other_costs": value.other_costs,
            "quantity": value.matched_quantity,
            "entry_basis": value.original_entry_basis,
        },
        fees={
            name: getattr(value, name + "_fee")
            for name in ("spot_entry", "perp_entry", "spot_exit", "perp_exit")
        },
        funding_events=[
            {
                "event_id": "observed",
                "symbol": value.symbol,
                "settled_at": at,
                "funding_rate": "0",
                "settlement_mark_price": "100",
            }
        ],
        decomposition={
            name: Decimal(0)
            for name in (
                "mid_basis_pnl",
                "spread_costs",
                "slippage_costs",
                "rounding_residual",
                "price_pnl",
            )
        },
    )
    return replace(value, details_json=exact_json(details))


def completed(positive_symbols=()):
    entries, tasks = cohort()
    by_id = {e.opportunity_id: e for e in entries}
    tasks = tuple(replace(t, status=TaskStatus.COMPLETED, updated_at=t.due_at) for t in tasks)
    values = tuple(
        saved(
            by_id[t.opportunity_id], t, Decimal(1) if t.symbol in positive_symbols else Decimal(-1)
        )
        for t in tasks
    )
    return entries, tasks, values


def assess(entries, tasks, values, at=DEADLINE):
    return assess_pilot_v1(entries, tasks, values, evaluated_at=at, deadline=DEADLINE)


@pytest.mark.parametrize(
    "positive,conclusion,candidate,benchmark",
    [
        ((), "NO_GO_FOR_CURRENT_PILOT", 0, 0),
        (("BTCUSDT",), "NO_GO_FOR_CURRENT_PILOT", 0, 3),
        (("THEUSDT",), "EXPLORATORY_SIGNAL", 3, 0),
        (("THEUSDT", "TSTUSDT"), "EXPLORATORY_SIGNAL", 6, 0),
    ],
)
def test_complete_outcomes_follow_fixed_candidate_and_control_protocol(
    positive, conclusion, candidate, benchmark
):
    result = assess(*completed(positive))
    assert result["phase"] == "FINAL" and result["conclusion"] == conclusion
    assert result["coverage_complete"] and result["usable_validations"] == 12
    assert result["candidate_positive_outcomes"] == candidate
    assert result["benchmark_positive_outcomes"] == benchmark
    assert all(r["funding_event_count"] == 1 for r in result["rows"])
    assert not result["strategy_qualified"]


def test_positive_remaining_results_cannot_erase_four_missed_windows():
    entries, tasks, values = completed(("THEUSDT", "TSTUSDT"))
    tasks = tuple(
        replace(t, status=TaskStatus.FAILED, failure_code=TaskFailure.VALIDATION_WINDOW_MISSED)
        if t.horizon_seconds == 86400
        else t
        for t in tasks
    )
    values = tuple(
        v
        for v in values
        if v.id not in {t.validation_id for t in tasks if t.status is TaskStatus.FAILED}
    )
    result = assess(entries, tasks, values)
    assert result["conclusion"] == "INSUFFICIENT_EVIDENCE"
    assert result["candidate_positive_outcomes"] == 4 and result["usable_validations"] == 8


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing_task", "MISSING_TASK"),
        ("duplicate_task", "DUPLICATE_TASK"),
        ("missing_validation", "MISSING_VALIDATION"),
        ("duplicate_validation", "DUPLICATE_VALIDATION"),
        ("wrong_symbol", "INVALID_IDENTITY"),
        ("wrong_due", "INVALID_IDENTITY"),
        ("quantity", "QUANTITY_MISMATCH"),
        ("null_profit", "MISSING_OR_INVALID_METRIC"),
        ("nan_profit", "MISSING_OR_INVALID_METRIC"),
        ("negative_fee", "MISSING_OR_INVALID_METRIC"),
        ("future_time", "INVALID_TIMING"),
        ("wrong_duration", "INVALID_TIMING"),
        ("failed_status", "INVALID_VALIDATION"),
        ("completed_with_failure", "TASK_NOT_COMPLETED"),
        ("damaged_json", "INVALID_STORED_DETAILS"),
        ("projection_mismatch", "INVALID_STORED_DETAILS"),
    ],
)
def test_unusable_evidence_is_retained_with_null_profit(case, reason):
    entries, tasks, values = completed()
    tasks, values = list(tasks), list(values)
    if case == "missing_task":
        tasks.pop(0)
    elif case == "duplicate_task":
        tasks.append(tasks[0])
    elif case == "missing_validation":
        values.pop(0)
    elif case == "duplicate_validation":
        values.append(values[0])
    elif case == "wrong_symbol":
        tasks[0] = replace(tasks[0], symbol="ETHUSDT")
    elif case == "wrong_due":
        tasks[0] = replace(tasks[0], due_at=OPENED)
    elif case == "quantity":
        values[0] = replace(values[0], matched_quantity=Decimal(2))
    elif case == "null_profit":
        values[0] = replace(values[0], validated_simulated_net_profit=None)
    elif case == "nan_profit":
        values[0] = replace(values[0], validated_simulated_net_profit=Decimal("NaN"))
    elif case == "negative_fee":
        values[0] = replace(values[0], spot_exit_fee=Decimal(-1))
    elif case == "future_time":
        values[0] = replace(values[0], validation_time=DEADLINE + timedelta(seconds=1))
    elif case == "wrong_duration":
        values[0] = replace(values[0], holding_duration=timedelta(seconds=1))
    elif case == "failed_status":
        values[0] = replace(values[0], status="INCOMPLETE")
    elif case == "completed_with_failure":
        tasks[0] = replace(tasks[0], failure_code=TaskFailure.PERSISTENCE_FAILED)
    elif case == "damaged_json":
        values[0] = replace(values[0], details_json="{")
    else:
        values[0] = replace(values[0], validated_simulated_net_profit=Decimal(100))
    result = assess(entries, tuple(tasks), tuple(values))
    row = result["rows"][0]
    assert reason in row["reason_codes"] and not row["usable"]
    assert row["validated_simulated_net_profit"] is None
    assert result["conclusion"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("late,usable", [(0, True), (300, True), (301, False), (-1, False)])
def test_original_lateness_boundary(late, usable):
    entries, tasks, values = completed()
    values = (saved(entries[0], tasks[0], late=late), *values[1:])
    tasks = (replace(tasks[0], updated_at=values[0].created_at), *tasks[1:])
    assert assess(entries, tasks, values)["rows"][0]["usable"] is usable


@pytest.mark.parametrize(
    "case", ["forecast", "duplicate_id", "duplicate_time", "outside", "symbol", "missing"]
)
def test_invalid_funding_details_cannot_be_reported_as_observed(case):
    entries, tasks, values = completed()
    details = json.loads(values[0].details_json)
    event = details["funding_events"][0]
    if case == "forecast":
        event["event_id"] = "FORECAST_ONLY:BTCUSDT"
    elif case == "duplicate_id":
        details["funding_events"].append(dict(event, settled_at=OPENED.isoformat()))
    elif case == "duplicate_time":
        details["funding_events"].append(dict(event, event_id="another"))
    elif case == "outside":
        event["settled_at"] = DEADLINE.isoformat()
    elif case == "symbol":
        event["symbol"] = "ETHUSDT"
    else:
        del details["funding_events"]
    values = (replace(values[0], details_json=exact_json(details)), *values[1:])
    assert "INVALID_STORED_DETAILS" in assess(entries, tasks, values)["rows"][0]["reason_codes"]


def test_deadline_finalizes_report_without_mutating_pending_tasks():
    entries, tasks = cohort()
    result = assess(entries, tasks, ())
    assert result["phase"] == "FINAL" and result["conclusion"] == "INSUFFICIENT_EVIDENCE"
    assert all(t.status is TaskStatus.PENDING for t in tasks)


@pytest.mark.parametrize("case", ["task", "validation"])
def test_extra_evidence_blocks_complete_coverage(case):
    entries, tasks, values = completed()
    if case == "task":
        tasks += (replace(tasks[0], id="extra", horizon_seconds=3600),)
    else:
        values += (replace(values[0], id="extra"),)
    result = assess(entries, tasks, values)
    assert result["conclusion"] == "INSUFFICIENT_EVIDENCE" and not result["coverage_complete"]


@pytest.mark.parametrize("case", ["missing", "duplicate", "future", "deadline"])
def test_replaced_entry_scope_is_rejected(case):
    entries, tasks = cohort()
    if case == "missing":
        entries = entries[:3]
    elif case == "duplicate":
        entries = (entries[0], entries[0], *entries[2:])
    elif case == "future":
        entries = (replace(entries[0], created_at=DEADLINE + timedelta(days=1)), *entries[1:])
    else:
        entries = (replace(entries[0], created_at=DEADLINE - timedelta(days=1)), *entries[1:])
    with pytest.raises(ValueError):
        assess(entries, tasks, ())


def test_four_missed_and_eight_pending_are_not_final_or_zero_profit():
    from dataclasses import replace

    from crypto_arbitrage_bot.analytics.pilot_outcome import assess_pilot_v1

    entries, tasks = cohort()
    tasks = tuple(
        replace(t, status=TaskStatus.FAILED, failure_code=TaskFailure.VALIDATION_WINDOW_MISSED)
        if t.horizon_seconds == 86400
        else t
        for t in tasks
    )
    result = assess_pilot_v1(
        entries, tasks, (), evaluated_at=OPENED + timedelta(days=2), deadline=DEADLINE
    )
    assert result["phase"] == "WAITING_OBSERVATIONS"
    assert result["provisional_conclusion"] == "INSUFFICIENT_EVIDENCE"
    assert result["expected_observations"] == 12 and result["usable_validations"] == 0
    assert result["task_status_counts"] == {"PENDING": 8, "FAILED": 4}
    assert all(r["validated_simulated_net_profit"] is None for r in result["rows"])
    assert result["strategy_qualified"] is False and result["orders_submitted"] is False
