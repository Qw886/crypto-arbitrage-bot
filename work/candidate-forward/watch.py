"""Finite local acceptance worker; delegates each batch to tested candidate scheduler."""
import argparse
import time
from pathlib import Path
from threading import Event

from crypto_arbitrage_bot.candidate_forward_research import run_due_once
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now


def watch(settings, output_dir, *, duration_seconds, poll_seconds, max_batches,
          cycle=run_due_once, monotonic=time.monotonic, wait=None, stop=None,
          extended_duration=False):
    if type(extended_duration) is not bool:
        raise ValueError("extended_duration must be an explicit boolean")
    duration_limit = 8 * 86400 if extended_duration else 172800
    batch_limit = 30000 if extended_duration else 10000
    if (any(type(value) is not int for value in (duration_seconds, poll_seconds, max_batches))
            or not 1 <= duration_seconds <= duration_limit or not 1 <= poll_seconds <= 60
            or not 1 <= max_batches <= batch_limit):
        raise ValueError("bounded duration, polling and batch count are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "state.json").exists():
        raise ValueError("existing worker evidence requires bounded resume, not a new watch")
    event = Event()
    wait = event.wait if wait is None else wait
    stop = (lambda: (output_dir / "stop").exists()) if stop is None else stop
    start = monotonic()
    completed = 0
    status = {"state": "RUNNING", "started_at": utc_now(), "ended_at": None,
              "completed_batches": 0, "failed_batches": 0, "task_events": 0,
              "duration_seconds": duration_seconds, "poll_seconds": poll_seconds,
              "extended_duration": extended_duration,
              "sqlite_path": str(settings.research.sqlite_path),
              "meaning": "Public forward research only; no orders. Deadline misses remain failures."}
    def save():
        temporary = output_dir / "state.tmp"
        temporary.write_text(exact_json(status) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "state.json")
    save()
    print(exact_json(status), flush=True)
    try:
        while completed < max_batches and monotonic() - start < duration_seconds and not stop():
            try:
                result = cycle(settings, output_dir / f"batch-{completed:05d}.json")
                status["task_events"] += len(result["task_events"])
                status["last_error"] = None
            except Exception as error:
                status["failed_batches"] += 1
                status["last_error"] = {"type": type(error).__name__, "message": str(error)[:300]}
            completed += 1
            status["completed_batches"] = completed
            status["updated_at"] = utc_now()
            save()
            remaining = duration_seconds - (monotonic() - start)
            if remaining > 0 and completed < max_batches and not stop():
                wait(min(poll_seconds, remaining))
        status["state"] = "STOPPED" if stop() else "FINISHED"
    except KeyboardInterrupt:
        status["state"] = "INTERRUPTED"
    finally:
        status["ended_at"] = utc_now()
        save()
        print(exact_json(status), flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration-seconds", required=True, type=int)
    parser.add_argument("--poll-seconds", required=True, type=int)
    parser.add_argument("--max-batches", required=True, type=int)
    parser.add_argument("--extended-duration", action="store_true",
                        help="Explicit finite research: at most eight days/30000 batches; public GET only")
    args = parser.parse_args()
    watch(load_settings(args.config), args.output_dir, duration_seconds=args.duration_seconds,
          poll_seconds=args.poll_seconds, max_batches=args.max_batches,
          extended_duration=args.extended_duration)
