"""Lifecycle acceptance at the Bot/controller interfaces; isolated mock exchange."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState, PairOrderPlan
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.file_lock import ControlError, LockBusyError, file_lock
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.fixture
def controlled(tmp_path):
    from crypto_arbitrage_bot.execution.control import prepare_job

    original = load_settings(Path("docs/testnet-bot.toml"))
    bot, transport = demo_bot(original, tmp_path)
    opened = bot.clock()
    elapsed, waits = [0], []
    bot.clock = lambda: opened + timedelta(seconds=elapsed[0])
    config = tmp_path / "control.toml"
    text = Path("docs/testnet-bot.toml").read_text(encoding="utf-8")
    text = text.replace(
        original.testnet.journal_path.as_posix(), bot.config.journal_path.as_posix()
    )
    text = text.replace(
        original.research.sqlite_path.as_posix(), bot.research.sqlite_path.as_posix()
    )
    config.write_text(text, encoding="utf-8")
    job = tmp_path / "controller"
    prepare_job(config, job, duration_seconds=20, max_cycles=10, clock=bot.clock)

    def wait(seconds):
        waits.append(seconds)
        elapsed[0] += seconds
        return False

    return bot, transport, config, job, elapsed, waits, wait


def controlled_run(controlled, wait=None):
    from crypto_arbitrage_bot.execution.control import run_job

    bot, _, _, job, elapsed, _, original_wait = controlled
    return run_job(
        job,
        clock=bot.clock,
        steady_ns=lambda: int(elapsed[0] * 1_000_000_000),
        wait=original_wait if wait is None else wait,
        factory=lambda settings, credentials: bot,
    )


def pair_plan(name="existing-pair"):
    return PairOrderPlan(
        name,
        IocLimitOrder(
            ExecutionMarket.SPOT,
            "BTCUSDT",
            Side.BUY,
            Decimal("0.001"),
            Decimal("50000"),
            name + "-se",
        ),
        IocLimitOrder(
            ExecutionMarket.USDTM,
            "BTCUSDT",
            Side.SELL,
            Decimal("0.001"),
            Decimal("50009"),
            name + "-pe",
        ),
        IocLimitOrder(
            ExecutionMarket.SPOT,
            "BTCUSDT",
            Side.SELL,
            Decimal("0.001"),
            Decimal("49999"),
            name + "-sx",
        ),
        IocLimitOrder(
            ExecutionMarket.USDTM,
            "BTCUSDT",
            Side.BUY,
            Decimal("0.001"),
            Decimal("50010"),
            name + "-px",
            True,
        ),
    )


def test_managed_cycle_preserves_not_due_pair_and_never_creates_an_entry(tmp_path):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    plan = pair_plan()
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    result = bot.manage_existing_once()
    assert result[0].pair.state is PairExecutionState.HEDGED
    assert len(transport.orders) == 2 and len(bot.journal.plans()) == 1
    assert not bot.repository.list_recent_opportunities()


def test_controlled_session_owns_original_journal_lock_and_blocks_entry(tmp_path):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    other = Bot(bot.settings, bot.spot, bot.perp)
    with bot.controlled_session(manage_only=True):
        assert bot.run_once() == ()
        with pytest.raises(ExecutionGuardError):
            other.run_once()
    assert not bot.repository.list_recent_opportunities() and not transport.orders
    assert other.manage_existing_once() == ()


def test_prepare_and_status_preserve_existing_attempt_budget_without_keys(tmp_path):
    from crypto_arbitrage_bot.execution.control import prepare_job, status

    settings = load_settings(Path("docs/testnet-bot.toml"))
    bot, _ = demo_bot(settings, tmp_path)
    config = tmp_path / "control.toml"
    text = Path("docs/testnet-bot.toml").read_text(encoding="utf-8")
    text = text.replace(
        str(settings.testnet.journal_path).replace("\\", "/"), bot.config.journal_path.as_posix()
    )
    text = text.replace(
        str(settings.research.sqlite_path).replace("\\", "/"), bot.research.sqlite_path.as_posix()
    )
    config.write_text(text, encoding="utf-8")
    directory = tmp_path / "control-job"
    plan = prepare_job(config, directory, duration_seconds=65, max_cycles=10)
    result = status(directory, max_heartbeat_age_seconds=90)
    assert result["journal"]["budget_consumed"] == 0
    assert plan["max_total_attempts"] == settings.testnet.max_total_attempts
    assert result["state"]["cycles_started"] == 0
    assert result["state"]["status"] == "PREPARED"
    assert result["journal"]["exchange_account_flat_verified"] is False


def test_managed_controller_finishes_without_credentials_when_local_journal_has_no_work(tmp_path):
    from crypto_arbitrage_bot.execution.control import prepare_job, run_job

    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    settings = load_settings(Path("docs/testnet-bot.toml"))
    config = tmp_path / "control.toml"
    text = Path("docs/testnet-bot.toml").read_text(encoding="utf-8")
    text = text.replace(
        settings.testnet.journal_path.as_posix(), bot.config.journal_path.as_posix()
    )
    text = text.replace(
        settings.research.sqlite_path.as_posix(), bot.research.sqlite_path.as_posix()
    )
    config.write_text(text, encoding="utf-8")
    job = tmp_path / "control-job"
    prepare_job(config, job, duration_seconds=65, max_cycles=10)
    result = run_job(job)
    assert result["status"] == "FINISHED" and result["cycles_started"] == 0
    assert result["completion_reason"] == "LOCAL_JOURNAL_NO_ACTIVE_ATTEMPTS"
    assert not transport.requests and not transport.orders


def test_controller_waits_for_existing_plan_then_closes_without_another_entry(tmp_path):
    from crypto_arbitrage_bot.execution.control import prepare_job, run_job

    original = load_settings(Path("docs/testnet-bot.toml"))
    bot, transport = demo_bot(original, tmp_path)
    start = bot.clock()
    elapsed, waits = [0], []
    bot.clock = lambda: start + timedelta(seconds=elapsed[0])
    plan = replace(
        pair_plan(),
        scheduled_exit_at=start + timedelta(seconds=2),
        funding_settlement_at=start + timedelta(seconds=1),
        opportunity_id="offline-source",
    )
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    config = tmp_path / "control.toml"
    body = Path("docs/testnet-bot.toml").read_text(encoding="utf-8")
    body = body.replace(
        original.testnet.journal_path.as_posix(), bot.config.journal_path.as_posix()
    )
    body = body.replace(
        original.research.sqlite_path.as_posix(), bot.research.sqlite_path.as_posix()
    )
    config.write_text(body, encoding="utf-8")
    job = tmp_path / "controller"
    prepare_job(config, job, duration_seconds=20, max_cycles=10, clock=bot.clock)

    def wait(seconds):
        waits.append(seconds)
        elapsed[0] += seconds
        return False

    result = run_job(
        job,
        clock=bot.clock,
        steady_ns=lambda: int(elapsed[0] * 1_000_000_000),
        wait=wait,
        factory=lambda settings, credentials: bot,
    )
    assert result["status"] == "FINISHED" and result["cycles_completed"] == 2
    assert waits == [original.poll_interval_seconds]
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
    assert len(transport.orders) == 4 and len(bot.journal.plans()) == 1


def test_stopped_controller_drains_original_plan_without_fast_retries(controlled):
    from crypto_arbitrage_bot.execution.control import request_stop, resume_job

    bot, transport, _, job, _, waits, _ = controlled
    opened = bot.clock()
    plan = replace(
        pair_plan(),
        scheduled_exit_at=opened + timedelta(seconds=2),
        funding_settlement_at=opened + timedelta(seconds=1),
        opportunity_id="offline-source",
    )
    bot.coordinator.open_pair(plan)
    request_stop(job)
    result = controlled_run(controlled)
    assert result["status"] == "STOPPED" and waits == [bot.settings.poll_interval_seconds]
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
    assert len(transport.orders) == 4
    resume_job(job)
    assert controlled_run(controlled)["cycles_started"] == 2


def test_deadline_with_future_exit_preserves_exposure_and_does_not_report_closed(controlled):
    bot, transport, _, _, _, waits, _ = controlled
    bot.coordinator.open_pair(pair_plan())
    result = controlled_run(controlled)
    assert result["status"] == "RECOVERY_REQUIRED" and result["cycles_started"] == 4
    assert waits == [5, 5, 5, 5]
    assert (
        len(transport.orders) == 2
        and bot.journal.get("existing-pair").state is PairExecutionState.HEDGED
    )


def test_restart_keeps_original_deadline_consumption_and_does_not_resubmit_entry(controlled):
    from crypto_arbitrage_bot.execution.control import resume_job, status

    bot, transport, _, job, _, _, _ = controlled
    bot.coordinator.open_pair(pair_plan())

    def interrupted_wait(_):
        raise KeyboardInterrupt()

    first = controlled_run(controlled, interrupted_wait)
    assert first["status"] == "INTERRUPTED" and first["cycles_started"] == 1
    before = status(job, max_heartbeat_age_seconds=90, clock=bot.clock)["plan"]
    resume_job(job)
    second = controlled_run(controlled)
    assert second["cycles_started"] == 4 and second["cycles_completed"] == 4
    assert status(job, max_heartbeat_age_seconds=90, clock=bot.clock)["plan"] == before
    assert len(transport.orders) == 2


def test_budget_ended_job_cannot_be_extended_by_resume(controlled):
    from crypto_arbitrage_bot.execution.control import resume_job

    bot, transport, _, job, _, _, _ = controlled
    bot.coordinator.open_pair(pair_plan())
    controlled_run(controlled)
    before = list(transport.requests)
    resume_job(job)
    result = controlled_run(controlled)
    assert result["status"] == "RECOVERY_REQUIRED" and result["cycles_started"] == 4
    assert transport.requests == before and len(transport.orders) == 2


def test_original_journal_lock_protects_controlled_runner_and_resume(controlled):
    from crypto_arbitrage_bot.execution.control import resume_job
    from crypto_arbitrage_bot.execution.runner import process_lock

    bot, _, _, job, *_ = controlled
    with process_lock(bot.config.journal_path.with_suffix(".lock")):
        with pytest.raises(ExecutionGuardError):
            controlled_run(controlled)
        with pytest.raises(ExecutionGuardError):
            resume_job(job)
    with file_lock(job / "worker.lock"), pytest.raises(LockBusyError):
        controlled_run(controlled)


@pytest.mark.parametrize(
    "observation", [ProcessState.MATCHED, ProcessState.UNKNOWN, ProcessState.DIFFERENT_IDENTITY]
)
def test_live_or_unverified_owner_cannot_be_replaced(controlled, observation):
    import json

    from crypto_arbitrage_bot.execution.control import resume_job

    bot, _, _, job, *_ = controlled
    # Deliberately damaged external metadata, not an alternative production writer.
    path = job / "state.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(status="RUNNING", worker_pid=4242)
    path.write_text(exact_json(value), encoding="utf-8")
    with pytest.raises(ControlError, match="OWNER_NOT_CONFIRMED_MISSING"):
        resume_job(job, probe=lambda *a, **k: ProcessObservation(observation, 4242))


def test_changed_policy_blocks_resume_but_cannot_block_stop(controlled):
    from crypto_arbitrage_bot.execution.control import request_stop, resume_job

    _, _, config, job, *_ = controlled
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "max_total_attempts = 20", "max_total_attempts = 21"
        ),
        encoding="utf-8",
    )
    assert request_stop(job)["status"] == "STOP_REQUESTED"
    with pytest.raises(ControlError, match="POLICY_CHANGED"):
        resume_job(job)


def test_managed_session_stop_blocks_new_research_and_preserves_due_exit(controlled):
    bot, transport, *_ = controlled
    opened = bot.clock()
    plan = replace(
        pair_plan(),
        scheduled_exit_at=opened + timedelta(seconds=1),
        funding_settlement_at=opened,
        opportunity_id="offline-source",
    )
    bot.coordinator.open_pair(plan)
    controlled[4][0] = 2
    with bot.controlled_session(manage_only=False, entry_stop=lambda: True):
        results = bot.run_once()
    assert results[0].pair.state is PairExecutionState.CLOSED
    assert len(transport.orders) == 4 and not bot.repository.list_recent_opportunities()


def test_cli_status_and_stop_work_without_loading_test_credentials(controlled, capsys):
    from crypto_arbitrage_bot.execution.control import main

    job = controlled[3]
    assert main(["status", "--job-dir", str(job), "--max-heartbeat-age-seconds", "90"]) == 0
    assert main(["stop", "--job-dir", str(job)]) == 0
    assert '"credentials_read":false' in capsys.readouterr().out


def test_background_worker_finishes_empty_journal_without_network_or_credentials(controlled):
    import json
    import subprocess
    import sys

    from crypto_arbitrage_bot.execution.control import status

    job = controlled[3]
    before = controlled[0].config.journal_path.read_bytes()
    launched = subprocess.run(
        [
            sys.executable,
            "-m",
            "crypto_arbitrage_bot.execution.control",
            "resume",
            "--job-dir",
            str(job),
            "--background",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert launched.returncode == 0, launched.stdout + launched.stderr
    receipt = json.loads(launched.stdout)
    assert receipt["status"] == "LAUNCHED" and receipt["pid"] > 0
    from threading import Event
    from time import monotonic

    deadline = monotonic() + 15
    while monotonic() < deadline:
        value = status(job, max_heartbeat_age_seconds=90)
        if value["state"]["status"] == "FINISHED":
            break
        Event().wait(0.05)
    assert value["state"]["status"] == "FINISHED"
    assert value["state"]["cycles_started"] == 0
    assert value["journal"]["exchange_account_flat_verified"] is False
    assert controlled[0].config.journal_path.read_bytes() == before
    assert not controlled[1].requests


def test_failed_exit_stays_pending_and_is_visible_in_managed_cycle(controlled, monkeypatch):
    bot, transport, _, _, elapsed, *_ = controlled
    opened = bot.clock()
    plan = replace(
        pair_plan(),
        scheduled_exit_at=opened + timedelta(seconds=1),
        funding_settlement_at=opened,
        opportunity_id="offline-source",
    )
    bot.coordinator.open_pair(plan)
    original_open = transport.open

    def failed_depth(request, *args, **kwargs):
        if "/depth" in request.full_url:
            raise TimeoutError("offline injected transport timeout")
        return original_open(request, *args, **kwargs)

    monkeypatch.setattr(transport, "open", failed_depth)
    elapsed[0] = 2
    cycle = bot.manage_existing_once()[0]
    assert cycle.status == "EXIT_PRIORITY" and cycle.reason == "MANAGED_EXIT_PENDING"
    assert cycle.pair.state is PairExecutionState.HEDGED
    result = controlled_run(controlled)
    assert result["status"] == "RECOVERY_REQUIRED" and result["cycles_started"] <= 4
    assert len(transport.orders) == 2
    assert not bot.repository.list_recent_opportunities()


def test_stop_on_empty_journal_is_reported_stopped_without_loading_credentials(controlled):
    from crypto_arbitrage_bot.execution.control import request_stop, run_job

    request_stop(controlled[3])
    result = run_job(controlled[3])
    assert result["status"] == "STOPPED" and result["cycles_started"] == 0
    assert not controlled[1].requests


@pytest.mark.parametrize(
    "change",
    [
        {"failed_cycles": -1},
        {"failed_cycles": True},
        {"failed_cycles": 1},
        {"interrupted_cycles": -1},
        {"interrupted_cycles": 1},
        {"active_cycle": 0},
        {"active_cycle": True},
    ],
)
def test_invalid_durable_consumption_cannot_be_resumed(controlled, change):
    from crypto_arbitrage_bot.execution.control import resume_job

    job = controlled[3]
    value = json.loads((job / "state.json").read_text(encoding="utf-8"))
    value.update(change)
    (job / "state.json").write_text(exact_json(value), encoding="utf-8")
    with pytest.raises(ControlError, match="POLICY_CHANGED"):
        resume_job(job)
    assert not controlled[1].requests


def test_resume_binds_original_config_not_only_financial_policy(controlled):
    from crypto_arbitrage_bot.execution.control import resume_job

    _, _, config, job, *_ = controlled
    config.write_text(
        config.read_text(encoding="utf-8") + "\n# amended control configuration\n", encoding="utf-8"
    )
    with pytest.raises(ControlError, match="POLICY_CHANGED"):
        resume_job(job)


def test_unsubmitted_plan_is_not_turned_into_new_orders_by_recovery(controlled):
    bot, transport, *_ = controlled
    bot.journal.create(pair_plan())
    result = controlled_run(controlled)
    assert result["status"] == "FINISHED"
    assert bot.journal.get("existing-pair").state is PairExecutionState.FAILED
    assert not transport.orders and not transport.requests


def test_unknown_original_order_is_only_queried_not_resubmitted(controlled):
    bot, transport, *_ = controlled
    plan = pair_plan()
    bot.journal.create(plan)
    bot.journal.record_order(plan.spot_entry)
    bot.journal.update(
        plan.attempt_id,
        state=PairExecutionState.RECOVERY_REQUIRED,
        failure_code="SPOT_SUBMISSION_UNKNOWN",
    )
    result = controlled_run(controlled)
    assert result["status"] == "RECOVERY_REQUIRED"
    assert not transport.orders
    assert all(method == "GET" for method, _ in transport.requests)
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.RECOVERY_REQUIRED


def test_interrupted_cycle_claim_is_consumed_after_resume(controlled, monkeypatch):
    from crypto_arbitrage_bot.execution.control import resume_job

    bot, transport, _, job, *_ = controlled
    plan = pair_plan()
    bot.journal.create(plan)
    bot.journal.record_order(plan.spot_entry)
    bot.journal.update(
        plan.attempt_id,
        state=PairExecutionState.RECOVERY_REQUIRED,
        failure_code="SPOT_SUBMISSION_UNKNOWN",
    )
    original_open = transport.open

    def interrupted(request, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(transport, "open", interrupted)
    first = controlled_run(controlled)
    assert first["status"] == "INTERRUPTED" and first["active_cycle"] == 0
    assert first["cycles_started"] == 1 and first["cycles_completed"] == 0
    monkeypatch.setattr(transport, "open", original_open)
    resume_job(job)
    second = controlled_run(controlled)
    assert second["interrupted_cycles"] == 1 and second["cycles_started"] == 4
    assert second["cycles_completed"] == 3 and second["status"] == "RECOVERY_REQUIRED"
    assert not transport.orders


def test_automatic_control_without_original_admission_cannot_be_started(tmp_path):
    from crypto_arbitrage_bot.execution.control import ControlMode, prepare_job

    settings = load_settings(Path("docs/testnet-bot.toml"))
    bot, transport = demo_bot(settings, tmp_path)
    text = Path("docs/testnet-bot.toml").read_text(encoding="utf-8")
    text = text.replace(
        settings.testnet.journal_path.as_posix(), bot.config.journal_path.as_posix()
    )
    text = text.replace(
        settings.research.sqlite_path.as_posix(), bot.research.sqlite_path.as_posix()
    )
    config = tmp_path / "config.toml"
    config.write_text(text, encoding="utf-8")
    before = bot.config.journal_path.read_bytes()
    with pytest.raises(ControlError, match="ADMISSION_NOT_CONFIGURED"):
        prepare_job(
            config,
            tmp_path / "refused",
            duration_seconds=20,
            max_cycles=4,
            mode=ControlMode.AUTOMATIC,
        )
    assert bot.config.journal_path.read_bytes() == before and not transport.requests


def test_disabled_order_gate_reports_actionable_failure_and_preserves_active_plan(
    controlled, monkeypatch
):
    from crypto_arbitrage_bot.execution.control import run_job

    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    bot, transport, _, job, *_ = controlled
    bot.coordinator.open_pair(pair_plan())
    result = run_job(job)
    assert result["status"] == "FAILED"
    assert result["last_error_code"] == "TESTNET_ORDERS_DISABLED"
    assert result["cycles_started"] == 0
    assert bot.journal.get("existing-pair").state is PairExecutionState.HEDGED
    assert len(transport.orders) == 2
