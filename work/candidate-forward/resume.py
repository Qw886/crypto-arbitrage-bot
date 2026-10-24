"""Restore an interrupted public watch without resetting its time or batch budget.

Legacy runs did not save a max-batch/config binding. The first restore must pin
the previously observed configuration hash and batch limit explicitly. This is
an operational recovery receipt, never retrospective strategy qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from crypto_arbitrage_bot.candidate_forward_research import run_due_once
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import file_lock


def _at(value):
    return utc_timestamp(value if isinstance(value, datetime) else datetime.fromisoformat(value))


def _save(directory, state):
    temporary = directory / "state.tmp"
    temporary.write_text(exact_json(state)+"\n", encoding="utf-8")
    temporary.replace(directory / "state.json")


def restore(config_path, output_dir, *, expected_config_sha256, original_max_batches,
            cycle=run_due_once, clock=utc_now, watch=None):
    if (type(original_max_batches) is not int or not 1 <= original_max_batches <= 30000
            or not isinstance(expected_config_sha256, str) or len(expected_config_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_config_sha256)):
        raise ValueError("explicit original budget and lowercase config SHA256 required")
    config_path, output_dir = config_path.resolve(), output_dir.resolve()
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != expected_config_sha256:
        raise ValueError("original public configuration changed")
    settings = load_settings(config_path)
    if settings.testnet is not None or settings.research is None:
        raise ValueError("public research configuration required")
    if not (output_dir / "state.json").is_file():
        raise ValueError("original worker state required")
    if watch is None:
        spec = importlib.util.spec_from_file_location("original_bounded_watch", Path(__file__).with_name("watch.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        watch = module.watch
    with file_lock(output_dir / "worker.lock"):
        state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
        started = _at(state["started_at"])
        duration, poll, used = (state[n] for n in ("duration_seconds", "poll_seconds", "completed_batches"))
        if (type(duration) is not int or not 1 <= duration <= 8*86400
                or type(poll) is not int or not 1 <= poll <= 60
                or type(used) is not int or not 0 <= used <= original_max_batches
                or Path(state["sqlite_path"]).resolve() != settings.research.sqlite_path.resolve()):
            raise ValueError("original watch identity/time/budget invalid")
        deadline = started+timedelta(seconds=duration)
        binding = {"schema_version": 1, "config_path": str(config_path), "config_sha256": expected_config_sha256,
                   "original_started_at": started, "original_deadline": deadline,
                   "original_duration_seconds": duration, "poll_seconds": poll,
                   "original_max_batches": original_max_batches, "sqlite_path": str(settings.research.sqlite_path.resolve()),
                   "meaning": "Recovery of legacy observed parameters; not a preregistered strategy proof."}
        path = output_dir / "recovery-binding.json"
        if path.exists():
            if path.read_text(encoding="utf-8") != exact_json(binding):
                raise ValueError("original recovery binding changed")
        else:
            with path.open("xb") as stream:
                stream.write(exact_json(binding).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        now = utc_timestamp(clock())
        if now < started:
            raise ValueError("clock before original start")
        if now >= deadline or used >= original_max_batches:
            state.update(state="FINISHED", ended_at=now, updated_at=now, worker_pid=None)
            _save(output_dir, state)
            return state
        if (output_dir / "stop").exists():
            state.update(state="STOPPED", ended_at=now, updated_at=now, worker_pid=None)
            _save(output_dir, state)
            return state
        # Preserve evidence written before a crash but not reflected in state.
        while used < original_max_batches and (output_dir / f"batch-{used:05d}.json").exists():
            used += 1
        if state.get("active_batch") is not None:
            state["interrupted_batches"] = state.get("interrupted_batches", 0)+1
            state["active_batch"] = None
        state.update(completed_batches=used, state="RUNNING", ended_at=None, updated_at=now,
                     resumed_at=now, worker_pid=os.getpid(), original_deadline=deadline,
                     recovery_count=state.get("recovery_count", 0)+1)
        _save(output_dir, state)
        if used >= original_max_batches:
            state.update(state="FINISHED", ended_at=now, worker_pid=None)
            _save(output_dir, state)
            return state
        delta = deadline-now
        remaining = delta.days*86400+delta.seconds+bool(delta.microseconds)  # Wall cutoff below prevents rounding overrun.
        segment = output_dir / "recoveries" / uuid4().hex
        def stop():
            at = utc_timestamp(clock())
            return (output_dir / "stop").exists() or at >= deadline or at < started
        def delegated_cycle(_, ignored_output):
            index = state["completed_batches"]
            state.update(completed_batches=index+1, active_batch=index, updated_at=utc_timestamp(clock()))
            _save(output_dir, state)  # Durable consumption before SQLite/market IO.
            try:
                result = cycle(settings, output_dir / f"batch-{index:05d}.json")
                state["task_events"] += len(result["task_events"])
                state["last_error"] = None
                return result
            except BaseException as error:
                state["failed_batches"] += 1
                state["last_error"] = {"type": type(error).__name__}
                raise
            finally:
                state.update(active_batch=None, updated_at=utc_timestamp(clock()))
                _save(output_dir, state)
        try:
            outcome = watch(settings, segment, duration_seconds=remaining, poll_seconds=poll,
                            max_batches=original_max_batches-used, extended_duration=True,
                            cycle=delegated_cycle, stop=stop)
            state["state"] = outcome["state"]
            if utc_timestamp(clock()) >= deadline:
                state["state"] = "FINISHED"
        except BaseException:
            state["state"] = "INTERRUPTED"
            raise
        finally:
            state.update(ended_at=utc_timestamp(clock()), updated_at=utc_timestamp(clock()), worker_pid=None)
            _save(output_dir, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--original-config-sha256", required=True)
    parser.add_argument("--original-max-batches", required=True, type=int)
    args = parser.parse_args()
    try:
        result = restore(args.config, args.output_dir, expected_config_sha256=args.original_config_sha256,
                         original_max_batches=args.original_max_batches)
        print(exact_json(result), flush=True)
        return 0
    except Exception as error:
        print(exact_json({"status": "RESTORE_FAILED", "error_type": type(error).__name__, "orders_submitted": False}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
