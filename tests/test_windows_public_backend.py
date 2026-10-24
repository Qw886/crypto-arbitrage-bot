"""Scope/identity/budget tests with mocked OS service; no live jobs or orders."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot import windows_public_backend as module
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

AT = datetime(2026, 10, 12, tzinfo=UTC)


@pytest.fixture
def context(tmp_path, monkeypatch):
    target = tmp_path / "original"
    target.mkdir()
    config = tmp_path / "public.toml"
    config.write_text("original policy", encoding="utf-8")
    plan = {"ends_at": (AT + timedelta(hours=2)).isoformat(), "config_path": str(config)}
    (target / "plan.json").write_text(exact_json(plan), encoding="utf-8")
    monkeypatch.setattr(module, "load_job", lambda path: (plan, SimpleNamespace(testnet=None)))
    monkeypatch.setattr(module, "implementation_fingerprint", lambda: "a" * 64)
    return tmp_path, target, tmp_path / "backend", plan, config


def test_original_public_scope_creates_only_new_backend_files(context):
    root, target, output, plan, config = context
    original = (target / "plan.json").read_bytes()
    value = module.prepare(root, "research", target, output, now=AT)
    assert value["ends_at"] == plan["ends_at"] and value["execution_limit_seconds"] == 7200
    assert value["arguments"] == [
        "-m",
        "crypto_arbitrage_bot.research_control",
        "resume",
        "--job-dir",
        str(target),
    ]
    assert "--background" not in value["arguments"]
    assert module.load_plan(output) == value
    module.write_scripts(output)
    action = (output / "action.ps1").read_text(encoding="utf-8")
    install = (output / "install.ps1").read_text(encoding="utf-8")
    assert "EXPECTED_PLAN_SHA" not in action and module._sha(output / "plan.json") in action
    assert "-WindowStyle Hidden -Wait" in action and "backendGate.ExitCode" in action
    assert "RegisterTaskDefinition" in install and ",2,$null,$null,3)" in install
    assert "EndBoundary=$backendEnd" in install and "Interval='PT1M'" in install
    assert "ToUniversalTime().ToString('o')" in install and "UtcDateTime.ToString('o')" in install
    assert "LogonType=3" in install and "RunLevel=0" in install
    removal = (output / "remove.ps1").read_text(encoding="utf-8")
    status = (output / "status.ps1").read_text(encoding="utf-8")
    assert "BACKEND_DEFINITION_CHANGED" in status and "registered-task.xml" in status
    assert "WITHDRAW_AUTOMATIC_RECOVERY" in removal and "State -eq 4" in removal
    assert "DeleteTask($backendData.task_name,0)" in removal and ".Stop(" not in removal
    assert (target / "plan.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        module.prepare(root, "research", target, output, now=AT)


@pytest.mark.parametrize(
    "change",
    [
        "end",
        "args",
        "file_hash",
        "source",
        "interval",
        "limit_zero",
        "limit_boolean",
        "limit_changed",
        "name",
        "version_boolean",
        "noncanonical",
    ],
)
def test_changed_backend_or_original_scope_is_refused(context, monkeypatch, change):
    root, target, output, plan, config = context
    value = module.prepare(root, "research", target, output, now=AT)
    if change == "file_hash":
        config.write_text("changed", encoding="utf-8")
    elif change == "source":
        monkeypatch.setattr(module, "implementation_fingerprint", lambda: "b" * 64)
    elif change == "end":
        value["ends_at"] = (AT + timedelta(days=1)).isoformat()
    elif change == "args":
        value["arguments"] = ["-m", "crypto_arbitrage_bot.execution.main", "--forever"]
    elif change == "interval":
        value["interval_seconds"] = 1
    elif change.startswith("limit_"):
        value["execution_limit_seconds"] = {
            "limit_zero": 0,
            "limit_boolean": True,
            "limit_changed": 7201,
        }[change]
    elif change == "name":
        value["task_name"] = "SomeOtherApplicationTask"
    elif change == "version_boolean":
        value["schema_version"] = True
    if change == "noncanonical":
        (output / "plan.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    else:
        (output / "plan.json").write_text(exact_json(value), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_plan(output)


@pytest.mark.parametrize("kind", ["testnet", "arbitrary", "execution", "power"])
def test_backend_kind_cannot_launch_other_programs(context, kind):
    root, target, output, *_ = context
    with pytest.raises(ValueError):
        module.prepare(root, kind, target, output, now=AT)
    assert not output.exists()


@pytest.mark.parametrize("where", ["target", "output", "config"])
def test_paths_must_remain_in_explicit_project(context, tmp_path, where):
    root, target, output, plan, config = context
    if where == "target":
        target = root.parent / "elsewhere"
    elif where == "output":
        output = root.parent / "elsewhere"
    else:
        plan["config_path"] = str(root.parent / "elsewhere")
    with pytest.raises(ValueError):
        module.prepare(root, "research", target, output, now=AT)


@pytest.mark.parametrize("end", [AT, AT - timedelta(seconds=1), AT + timedelta(days=9)])
def test_original_deadline_never_expands_to_new_period(context, end):
    root, target, output, plan, config = context
    plan["ends_at"] = end.isoformat()
    with pytest.raises(ValueError):
        module.prepare(root, "research", target, output, now=AT)
    assert not output.exists()


@pytest.mark.parametrize(
    "status",
    ["RUNNING_CONFIRMED", "RUNNING_STALE_HEARTBEAT", "OWNER_UNVERIFIED", "UNAVAILABLE", "FINISHED"],
)
def test_live_stale_or_uncertain_owner_never_restarts(status):
    value = {
        "status": status,
        "process_observation": "MISSING",
        "worker_lock_held": False,
        "budget_consumed": 3,
        "budget_limit": 10,
    }
    assert not module.gate_result(
        value, stop_requested=False, now=AT, ends_at=AT + timedelta(hours=1)
    )


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "reused_pid",
        "locked",
        "lock_unknown",
        "used_unknown",
        "exhausted",
        "overbudget",
        "boolean_used",
        "limit_unknown",
        "stop",
        "expired",
    ],
)
def test_confirmed_missing_is_insufficient_without_original_lock_quota_and_deadline(change):
    value = {
        "status": "STOPPED_CONFIRMED",
        "process_observation": "MISSING",
        "worker_lock_held": False,
        "budget_consumed": 3,
        "budget_limit": 10,
    }
    stop = False
    end = AT + timedelta(hours=1)
    if change == "unknown":
        value["process_observation"] = "UNKNOWN"
    elif change == "reused_pid":
        value["process_observation"] = "DIFFERENT_IDENTITY"
    elif change == "locked":
        value["worker_lock_held"] = True
    elif change == "lock_unknown":
        value["worker_lock_held"] = None
    elif change == "used_unknown":
        value["budget_consumed"] = None
    elif change == "exhausted":
        value["budget_consumed"] = 10
    elif change == "overbudget":
        value["budget_consumed"] = 11
    elif change == "boolean_used":
        value["budget_consumed"] = True
    elif change == "limit_unknown":
        value["budget_limit"] = None
    elif change == "stop":
        stop = True
    else:
        end = AT
    assert not module.gate_result(value, stop_requested=stop, now=AT, ends_at=end)


def test_gate_allows_only_missing_owner_with_remaining_original_budget():
    value = {
        "status": "STOPPED_CONFIRMED",
        "process_observation": "MISSING",
        "worker_lock_held": False,
        "budget_consumed": 3,
        "budget_limit": 10,
    }
    assert module.gate_result(value, stop_requested=False, now=AT, ends_at=AT + timedelta(hours=1))


@pytest.mark.parametrize(
    "report_status,expected", [("RUNNING_CONFIRMED", False), ("STOPPED_CONFIRMED", True)]
)
def test_real_gate_uses_existing_native_status_and_performs_no_resume(
    context, monkeypatch, report_status, expected
):
    root, target, output, *_ = context
    module.prepare(root, "research", target, output, now=AT)
    monkeypatch.setattr(module, "utc_now", lambda: AT)
    monkeypatch.setattr(
        module,
        "research_status",
        lambda *a, **kw: {
            "status": report_status,
            "process_observation": "MISSING",
            "worker_lock_held": False,
            "budget_consumed": 3,
            "budget_limit": 10,
        },
    )
    monkeypatch.setattr(module.subprocess, "run", Mock(side_effect=AssertionError("no launch")))
    assert module.gate(output) is expected
    (target / "stop").touch()
    assert not module.gate(output)


def test_monitors_keep_original_sha_scope_and_frozen_deadline(context, monkeypatch):
    root, target, output, *_ = context
    (target / "monitor-binding.json").write_text("{}", encoding="utf-8")
    plan = {"ends_at": (AT + timedelta(hours=3)).isoformat()}
    monkeypatch.setattr(module.monitor_archive, "load_plan", lambda p, sha: (plan, "c" * 64))
    monkeypatch.setattr(
        module.monitor_archive,
        "load_binding",
        lambda p, sha: {"spec": {"research_config": None, "testnet_config": None}},
    )
    value = module.prepare(root, "monitor", target, output, sha="c" * 64, now=AT)
    assert value["arguments"][1:3] == ["crypto_arbitrage_bot.runtime_status", "resume-monitor"]
    assert value["arguments"][-1] == "c" * 64 and value["execution_limit_seconds"] == 10800
    assert module.load_plan(output) == value


def test_configured_testnet_job_is_not_a_public_target(context, monkeypatch):
    root, target, output, plan, *_ = context
    monkeypatch.setattr(module, "load_job", lambda path: (plan, SimpleNamespace(testnet=object())))
    with pytest.raises(ValueError):
        module.prepare(root, "research", target, output, now=AT)
    assert not output.exists()
