"""Unified read-only supervision of the bound TESTNET controller."""

import json
import sys
from datetime import datetime, timedelta

import pytest
from test_testnet_control import controlled as control_fixture

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.runtime_status import runtime_status
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.fixture
def controller(tmp_path):
    values = control_fixture.__wrapped__(tmp_path)
    bot, _, config, job, *_ = values
    state = json.loads((job / "state.json").read_text(encoding="utf-8"))
    state.update(status="RUNNING", worker_pid=4242)
    (job / "state.json").write_text(exact_json(state), encoding="utf-8")
    (job / "worker.lock").write_bytes(b"0")
    bot.config.journal_path.with_suffix(".lock").write_bytes(b"0")
    return values, load_settings(config)


def report(controller, *, observation=ProcessState.MATCHED, held=True, at=None):
    values, settings = controller
    bot, _, _, job, *_ = values

    def probe(pid, *, matches, **kwargs):
        assert matches(
            (
                sys.executable,
                "-m",
                "crypto_arbitrage_bot.execution.control",
                "_worker",
                "--job-dir",
                str(job.resolve()),
            )
        )
        created = datetime.fromisoformat(
            json.loads((job / "plan.json").read_text(encoding="utf-8"))["created_at"]
        )
        return ProcessObservation(observation, pid, created)

    return runtime_status(
        testnet_settings=settings,
        max_heartbeat_age=timedelta(seconds=10),
        clock=lambda: at or bot.clock() + timedelta(seconds=1),
        probe=probe,
        lock_probe=held if callable(held) else lambda path: held,
    )


def test_unified_status_identifies_bound_controller_without_reading_keys_or_network(controller):
    values, _ = controller
    before = values[0].config.journal_path.read_bytes()
    result = report(controller)
    owner = result["testnet"]["controller"]
    assert owner["status"] == "RUNNING_CONFIRMED" and owner["worker_pid"] == 4242
    assert owner["control_status"] == "RUNNING" and owner["mode"] == "MANAGE_ONLY"
    assert owner["budget_consumed"] == 0 and owner["budget_limit"] == 10
    assert not result["attention_required"] and not result["credentials_read"]
    assert result["testnet"]["exchange_account_flat_verified"] is False
    assert values[0].config.journal_path.read_bytes() == before and not values[1].requests


@pytest.mark.parametrize(
    "observation,held,expected",
    [
        (ProcessState.MISSING, False, "STOPPED_CONFIRMED"),
        (ProcessState.MISSING, True, "OWNER_UNVERIFIED"),
        (ProcessState.MATCHED, False, "OWNER_UNVERIFIED"),
        (ProcessState.UNKNOWN, True, "OWNER_UNVERIFIED"),
        (ProcessState.UNKNOWN, False, "OWNER_UNVERIFIED"),
        (ProcessState.DIFFERENT_IDENTITY, True, "OWNER_UNVERIFIED"),
        (ProcessState.DIFFERENT_IDENTITY, False, "OWNER_UNVERIFIED"),
    ],
)
def test_missing_or_uncertain_controller_requires_attention_without_restart(
    controller, observation, held, expected
):
    before = (controller[0][3] / "state.json").read_bytes()
    result = report(controller, observation=observation, held=held)
    assert result["testnet"]["controller"]["status"] == expected
    assert result["attention_required"]
    assert (controller[0][3] / "state.json").read_bytes() == before
    assert not controller[0][1].requests


def test_worker_lock_without_original_journal_lock_cannot_confirm_management(controller):
    result = report(controller, held=lambda path: path.name == "worker.lock")
    assert result["testnet"]["controller"]["status"] == "OWNER_UNVERIFIED"
    assert result["attention_required"]


def test_alive_but_old_controller_heartbeat_requires_attention(controller):
    result = report(controller, at=controller[0][0].clock() + timedelta(seconds=15))
    assert result["testnet"]["controller"]["status"] == "RUNNING_STALE_HEARTBEAT"
    assert result["attention_required"]


@pytest.mark.parametrize("damage", ["marker", "config", "state", "journal"])
def test_invalid_bound_controller_is_unavailable_not_falsely_running(controller, damage):
    values, _ = controller
    bot, _, config, job, *_ = values
    if damage == "marker":
        bot.config.journal_path.with_suffix(".sqlite.testnet-control.json").write_text(
            '{"directory":"wrong"}', encoding="utf-8"
        )
    elif damage == "config":
        config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    elif damage == "state":
        state = json.loads((job / "state.json").read_text(encoding="utf-8"))
        state["cycles_started"] = -1
        (job / "state.json").write_text(exact_json(state), encoding="utf-8")
    else:
        bot.config.journal_path.unlink()
    result = report(controller)
    assert result["testnet"]["controller"]["status"] == "UNAVAILABLE"
    assert result["attention_required"] and not result["orders_submitted"]
    assert not values[1].requests


def test_known_owner_does_not_hide_an_active_position_or_claim_exchange_flatness(controller):
    from test_testnet_control import pair_plan

    bot = controller[0][0]
    bot.coordinator.open_pair(pair_plan())
    result = report(controller)
    assert result["testnet"]["active_attempts"][0]["state"] == "HEDGED"
    assert "LOCAL_ACTIVE_POSITION_WITHOUT_CONFIRMED_OWNER" not in result["testnet"]["reason_codes"]
    assert result["attention_required"]  # Original conservative active-position alert is preserved.
    assert result["testnet"]["exchange_account_flat_verified"] is False


def test_control_heartbeat_and_cycle_count_do_not_create_repeated_monitor_events(
    controller, tmp_path
):
    from crypto_arbitrage_bot.runtime_status import monitor_status

    values, _ = controller
    bot, _, _, job, elapsed, *_ = values
    before = bot.config.journal_path.read_bytes()

    def wait(seconds):
        elapsed[0] += seconds
        state = json.loads((job / "state.json").read_text(encoding="utf-8"))
        state.update(
            updated_at=bot.clock(),
            cycles_started=state["cycles_started"] + 1,
            cycles_completed=state["cycles_completed"] + 1,
        )
        (job / "state.json").write_text(exact_json(state), encoding="utf-8")
        return False

    result = monitor_status(
        lambda: report(controller),
        tmp_path / "monitor",
        duration_seconds=5,
        interval_seconds=1,
        max_checks=3,
        clock=bot.clock,
        steady_ns=lambda: int(elapsed[0] * 1_000_000_000),
        wait=wait,
    )
    assert result["checks"] == 3 and result["changes"] == 1
    assert bot.config.journal_path.read_bytes() == before and not values[1].requests


def test_live_controller_past_original_budget_is_reported_for_attention(controller):
    values, _ = controller
    bot, _, _, job, *_ = values
    at = bot.clock() + timedelta(seconds=21)
    state = json.loads((job / "state.json").read_text(encoding="utf-8"))
    state["updated_at"] = at
    (job / "state.json").write_text(exact_json(state), encoding="utf-8")
    result = report(controller, at=at)
    assert result["testnet"]["controller"]["status"] == "RUNNING_CONFIRMED"
    assert "CONTROL_BUDGET_ENDED" in result["testnet"]["controller"]["reason_codes"]
    assert result["attention_required"]
