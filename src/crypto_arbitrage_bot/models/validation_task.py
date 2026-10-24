"""Durable scheduling identities and structured failures; no financial math."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskFailure(StrEnum):
    COLLECTION_FAILED = "COLLECTION_FAILED"
    INVALID_EXIT_MARKET_DATA = "INVALID_EXIT_MARKET_DATA"
    FUNDING_UNAVAILABLE = "FUNDING_UNAVAILABLE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    ENTRY_EVIDENCE_INVALID = "ENTRY_EVIDENCE_INVALID"
    RECOVERY_LIMIT_EXCEEDED = "RECOVERY_LIMIT_EXCEEDED"
    INTERRUPTED = "INTERRUPTED"
    VALIDATION_WINDOW_MISSED = "VALIDATION_WINDOW_MISSED"


@dataclass(frozen=True)
class ValidationTask:
    id: str
    opportunity_id: str
    symbol: str
    horizon_seconds: int
    due_at: datetime
    status: TaskStatus
    validation_id: str
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    lease_until: datetime | None = None
    claim_token: str | None = None
    failure_code: TaskFailure | None = None
    failure_reason: str | None = None
