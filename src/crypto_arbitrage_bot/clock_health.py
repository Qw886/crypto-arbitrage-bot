"""Bounded, read-only operational clock diagnostics using two public time GETs.

No time correction, retry, credentials, private requests, research writes or
qualification. Offset intervals retain network uncertainty; they are not market
quality validation. Child-process deadlines also bound DNS and slow body reads.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from time import monotonic_ns
from typing import Any, TypedDict

from crypto_arbitrage_bot.exchanges.binance_usdtm import fetch_public_time
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

ENDPOINTS = {"spot": PublicEndpoint.SPOT_TIME, "perp": PublicEndpoint.PERP_TIME}
MAX_BODY_BYTES = 4096
MAX_CHILD_BYTES = 8192
OFFSET_BOUND_US = 250_000
MAX_ROUND_TRIP_NS = 2_000_000_000
MAX_WALL_STEP_NS = 100_000_000
MAX_SAMPLE_AGE = timedelta(seconds=15)


class Failure(StrEnum):
    TIMEOUT = "PUBLIC_TIME_TIMEOUT"
    HTTP = "PUBLIC_TIME_HTTP_ERROR"
    IO = "PUBLIC_TIME_IO_ERROR"
    RESPONSE = "PUBLIC_TIME_RESPONSE_INVALID"
    CHILD = "PUBLIC_TIME_CHILD_INVALID"


class Health(StrEnum):
    WITHIN_BOUND = "WITHIN_DIAGNOSTIC_BOUND"
    BEHIND = "LOCAL_CLOCK_BEHIND"
    AHEAD = "LOCAL_CLOCK_AHEAD"
    UNKNOWN = "UNVERIFIED"


class OffsetInterval(TypedDict):
    endpoint: str
    lower_us: int
    upper_us: int


def _timeout(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 10:
        raise ValueError("BOUNDED_CLOCK_TIMEOUT_REQUIRED")
    return value


def _microseconds(value: timedelta) -> int:
    return (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds


@dataclass(frozen=True)
class Sample:
    endpoint: PublicEndpoint
    started_at: datetime
    finished_at: datetime
    elapsed_ns: int
    server_time: datetime | None
    http_status: int | None
    failure: Failure | None = None

    def __post_init__(self) -> None:
        if self.endpoint not in ENDPOINTS.values() or not isinstance(self.endpoint, PublicEndpoint):
            raise ValueError("PUBLIC_TIME_ENDPOINT_REQUIRED")
        for name in ("started_at", "finished_at"):
            object.__setattr__(self, name, utc_timestamp(getattr(self, name)))
        if self.server_time is not None:
            object.__setattr__(self, "server_time", utc_timestamp(self.server_time))
        if type(self.elapsed_ns) is not int or self.elapsed_ns < 0 or (
            self.http_status is not None and (type(self.http_status) is not int or not 100 <= self.http_status <= 599)
        ) or (self.failure is not None and not isinstance(self.failure, Failure)):
            raise ValueError("STRUCTURED_CLOCK_SAMPLE_REQUIRED")
        if self.failure is None and (self.server_time is None or self.http_status != 200):
            raise ValueError("SUCCESSFUL_PUBLIC_TIME_RESPONSE_REQUIRED")
        if self.failure is not None and self.server_time is not None:
            raise ValueError("FAILED_CLOCK_SAMPLE_HAS_NO_TIME")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("DUPLICATE_CLOCK_FIELD")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("NONFINITE_CLOCK_FIELD")


def _decode(body: bytes) -> Any:
    return json.loads(body.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject_constant)


def public_sample(endpoint: PublicEndpoint, timeout_seconds: int) -> Sample:
    """One fixed GET, no redirect/retry/auth; called in a deadline-bound child."""
    timeout = _timeout(timeout_seconds)
    if endpoint not in ENDPOINTS.values() or not isinstance(endpoint, PublicEndpoint):
        raise ValueError("PUBLIC_TIME_ENDPOINT_REQUIRED")
    value = fetch_public_time(prepare_get(endpoint), timeout)
    return Sample(endpoint, value.started_at, value.finished_at, value.elapsed_ns,
                  value.server_time, value.http_status,
                  None if value.failure_code is None else Failure(value.failure_code))


def run_probe(endpoint: PublicEndpoint, timeout_seconds: int) -> Sample:
    """Kill/wait only our own one-shot child after a bounded observation timeout."""
    timeout = _timeout(timeout_seconds)
    name = next((name for name, item in ENDPOINTS.items() if item is endpoint), None)
    if name is None:
        raise ValueError("PUBLIC_TIME_ENDPOINT_REQUIRED")
    start, tick = utc_now(), monotonic_ns()
    source = str(Path(__file__).resolve().parents[1])
    # Windows venv python.exe is a launcher: killing it can leave its real
    # worker alive. Use that interpreter directly, with this venv's libraries.
    executable = sys.executable
    if os.name == "nt":
        base = getattr(sys, "_base_executable", None)
        if not isinstance(base, str) or not Path(base).is_file():
            return Sample(endpoint, start, utc_now(), monotonic_ns()-tick, None, None, Failure.CHILD)
        executable = base
        source += os.pathsep + str(Path(sys.prefix)/"Lib/site-packages")
    env = {**os.environ, "PYTHONPATH": source}
    failure = Failure.CHILD
    try:
        result = subprocess.run(
            [executable, "-m", "crypto_arbitrage_bot.clock_health", "_sample", "--endpoint", name,
             "--timeout-seconds", str(timeout)],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout + 1,
            env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        if result.returncode != 0 or len(result.stdout) > MAX_CHILD_BYTES:
            raise ValueError("BOUNDED_CLOCK_CHILD_REQUIRED")
        value = _decode(result.stdout)
        if not isinstance(value, dict) or set(value) != {
            "endpoint", "started_at", "finished_at", "elapsed_ns", "server_time", "http_status", "failure"
        }:
            raise ValueError("STRUCTURED_CLOCK_CHILD_REQUIRED")
        sample = Sample(PublicEndpoint(value["endpoint"]), datetime.fromisoformat(value["started_at"]),
                        datetime.fromisoformat(value["finished_at"]), value["elapsed_ns"],
                        None if value["server_time"] is None else datetime.fromisoformat(value["server_time"]),
                        value["http_status"], None if value["failure"] is None else Failure(value["failure"]))
        if sample.endpoint is not endpoint or sample.started_at < start or sample.finished_at > utc_now():
            raise ValueError("CLOCK_CHILD_SCOPE_OR_WINDOW_CHANGED")
        return sample
    except subprocess.TimeoutExpired:
        failure = Failure.TIMEOUT
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return Sample(endpoint, start, utc_now(), monotonic_ns() - tick, None, None, failure)


def assess(samples: tuple[Sample, Sample], *, evaluated_at: datetime) -> dict[str, Any]:
    """Classify only when both fresh sources bound one consistent clock offset."""
    at = utc_timestamp(evaluated_at)
    if len(samples) != 2 or {item.endpoint for item in samples} != set(ENDPOINTS.values()):
        raise ValueError("BOTH_PUBLIC_CLOCK_SOURCES_REQUIRED")
    issues: list[str] = []
    intervals: list[OffsetInterval] = []
    for sample in samples:
        if sample.failure is not None:
            issues.append(sample.failure.value)
            continue
        wall_ns = _microseconds(sample.finished_at - sample.started_at) * 1000
        if sample.finished_at > at or at - sample.started_at > MAX_SAMPLE_AGE:
            issues.append("CLOCK_SAMPLE_NOT_FRESH")
        if wall_ns < 0 or abs(wall_ns - sample.elapsed_ns) > MAX_WALL_STEP_NS:
            issues.append("LOCAL_CLOCK_CHANGED_DURING_PROBE")
        if sample.elapsed_ns > MAX_ROUND_TRIP_NS:
            issues.append("CLOCK_ROUND_TRIP_TOO_WIDE")
        assert sample.server_time is not None
        # Conservative +/-1ms accounts for server timestamp quantization.
        intervals.append({"endpoint": sample.endpoint.value,
            "lower_us": _microseconds(sample.server_time - sample.finished_at) - 1000,
            "upper_us": _microseconds(sample.server_time - sample.started_at) + 1000})
    state = Health.UNKNOWN
    if not issues and len(intervals) == 2:
        lower = max(item["lower_us"] for item in intervals)
        upper = min(item["upper_us"] for item in intervals)
        if lower > upper:
            issues.append("PUBLIC_CLOCK_INTERVALS_DISAGREE")
        elif all(item["lower_us"] > OFFSET_BOUND_US for item in intervals):
            state = Health.BEHIND
            issues.append("PUBLIC_TIME_AHEAD_OF_LOCAL_CLOCK")
        elif all(item["upper_us"] < -OFFSET_BOUND_US for item in intervals):
            state = Health.AHEAD
            issues.append("PUBLIC_TIME_BEHIND_LOCAL_CLOCK")
        elif all(-OFFSET_BOUND_US <= item["lower_us"] <= item["upper_us"] <= OFFSET_BOUND_US for item in intervals):
            state = Health.WITHIN_BOUND
        else:
            issues.append("CLOCK_OFFSET_INTERVAL_INDETERMINATE")
    return {"schema_version": 1, "evaluated_at": at, "status": state,
            "attention_required": state is not Health.WITHIN_BOUND,
            "reason_codes": list(dict.fromkeys(issues)), "samples": samples,
            "server_minus_local_intervals": intervals, "diagnostic_bound_us": OFFSET_BOUND_US,
            "max_round_trip_ns": MAX_ROUND_TRIP_NS, "max_wall_step_ns": MAX_WALL_STEP_NS,
            "clock_modified": False, "credentials_read": False, "orders_submitted": False,
            "market_quality_verified": False, "qualification_issued": False}


def check(timeout_seconds: int = 5) -> dict[str, Any]:
    timeout = _timeout(timeout_seconds)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_probe, endpoint, timeout) for endpoint in ENDPOINTS.values()]
        spot, perp = (future.result() for future in futures)
    return assess((spot, perp), evaluated_at=utc_now())


def _output_path(root: Path, output: Path) -> Path:
    path = output.resolve()
    work = (root / "work").resolve()
    if not work.is_relative_to(root.resolve()) or not path.is_relative_to(work) or output.is_symlink() or path.suffix != ".json":
        raise ValueError("NEW_WORK_JSON_REPORT_REQUIRED")
    if path.exists() or not path.parent.is_dir():
        raise ValueError("NEW_WORK_JSON_REPORT_REQUIRED")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only PUBLIC clock health; no time changes or trading")
    parser.add_argument("command", choices=("check", "_sample"))
    parser.add_argument("--endpoint", choices=tuple(ENDPOINTS))
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        _timeout(args.timeout_seconds)
        if args.command == "_sample":
            if args.endpoint is None or args.output is not None:
                raise ValueError("ONE_CLOCK_SAMPLE_REQUIRED")
            print(exact_json(public_sample(ENDPOINTS[args.endpoint], args.timeout_seconds)))
            return 0
        if args.endpoint is not None:
            raise ValueError("BOTH_PUBLIC_CLOCK_SOURCES_REQUIRED")
        output = None if args.output is None else _output_path(Path.cwd(), args.output)
        report = check(args.timeout_seconds)
        body = exact_json(report)
        if output is not None:
            with output.open("x", encoding="utf-8") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        print(body)
        return {Health.WITHIN_BOUND: 0, Health.BEHIND: 1, Health.AHEAD: 1, Health.UNKNOWN: 2}[report["status"]]
    except (OSError, ValueError, TypeError):
        print(exact_json({"status": "CLOCK_CHECK_REFUSED", "credentials_read": False, "orders_submitted": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
