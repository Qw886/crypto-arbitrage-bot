"""UTC wall clock and monotonic timing; no server time requests."""

from datetime import UTC, datetime
from time import monotonic_ns


def utc_now() -> datetime:
    return datetime.now(UTC)


def monotonic_milliseconds() -> int:
    return monotonic_ns() // 1_000_000
