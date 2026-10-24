"""Bounded read-only stability harness, using the existing Runner unchanged.

Duration stops new rounds; an in-flight round finishes with existing socket
timeouts. This is not a hard process deadline. Memory numbers describe Python
traced allocations, not OS RSS or proof of absence of long-term leaks.
"""

import argparse
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.runner import CycleStatus, ResearchRunner, create_runner
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


@dataclass(frozen=True)
class SoakLimits:
    duration_seconds: int
    max_cycles: int  # Each round includes all configured symbols and due validations.

    def __post_init__(self) -> None:
        for value in (self.duration_seconds, self.max_cycles):
            if type(value) is not int or value <= 0:
                raise ValueError("soak duration and max_cycles must be positive integers")


@dataclass(frozen=True)
class SoakResult:
    rounds: int
    saved_cycles: int
    failed_cycles: int
    interrupted_cycles: int
    unexpected_round_errors: int
    last_error: str | None
    elapsed_ns: int
    traced_start_bytes: int
    traced_end_bytes: int
    traced_peak_bytes: int
    stopped: bool


def run_soak(runner: ResearchRunner, limits: SoakLimits, *,
             clock_ns: Callable[[], int] = monotonic_ns) -> SoakResult:
    """Bounded rounds with interruptible waits; preserve all business decisions.

    Uses settings.poll_interval_seconds (existing TOML/environment system).
    Counters stay constant-space even when many rounds are requested.
    """
    started = clock_ns()
    deadline = started + limits.duration_seconds * 1_000_000_000
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[0]
    rounds = saved = failed = interrupted = errors = 0
    last_error = None
    try:
        while not runner.stop_signal.is_set() and rounds < limits.max_cycles and clock_ns() < deadline:
            rounds += 1
            try:
                results = runner.run_once()  # Includes independently protected due validations.
                for cycle in results:
                    saved += cycle.status is CycleStatus.SAVED
                    failed += cycle.status is CycleStatus.FAILED
                    interrupted += cycle.status is CycleStatus.INTERRUPTED
            except Exception as exc:
                errors += 1
                last_error = f"{type(exc).__name__}: {exc}"
                runner.logger.error(exact_json({"soak_round": rounds, "failure": last_error}))
            remaining = deadline - clock_ns()
            if rounds >= limits.max_cycles or remaining <= 0 or runner.stop_signal.is_set():
                break
            # Float is solely a wall-clock wait duration, never financial data.
            if runner.stop_signal.wait(min(runner.settings.poll_interval_seconds, remaining / 1_000_000_000)):
                break
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        memory_end, memory_peak = tracemalloc.get_traced_memory()
        if own_trace:
            tracemalloc.stop()
    return SoakResult(rounds, saved, failed, interrupted, errors, last_error,
                      clock_ns() - started, memory_start, memory_end, memory_peak, runner.stop_signal.is_set())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded public read-only research soak; no trading")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--duration-seconds", required=True, type=int)
    parser.add_argument("--max-cycles", required=True, type=int)
    args = parser.parse_args(argv)
    limits = SoakLimits(args.duration_seconds, args.max_cycles)
    result = run_soak(create_runner(load_settings(args.config)), limits)
    print(exact_json(result))
    # Scheduler failures remain in its structured task logs/table, not hidden
    # behind this CLI's Runner-only exit code. Audit those as part of acceptance.
    return int(bool(result.failed_cycles or result.interrupted_cycles or result.unexpected_round_errors))


if __name__ == "__main__":
    raise SystemExit(main())
