"""Bounded task diagnostics; never serialize market payloads or response bodies."""

import json
from typing import TYPE_CHECKING, Any

from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly, SnapshotSource

if TYPE_CHECKING:
    from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow

MAX_FAILURE_REASON_CHARS = 2048
TRUNCATED_MARKER = "... [truncated]"
_MESSAGE_CHARS = 128


def limit_failure_reason(reason: str | None) -> str | None:
    """Final storage/log guard. Existing short text and None retain their meaning."""
    if reason is None or len(reason) <= MAX_FAILURE_REASON_CHARS:
        return reason
    return reason[:MAX_FAILURE_REASON_CHARS - len(TRUNCATED_MARKER)] + TRUNCATED_MARKER


def _encode(summary: dict[str, Any]) -> str:
    # Only message text is expendable: source/kind/phase/API diagnostics win.
    diagnostics = summary["sources"]
    for source in diagnostics:
        message = source.get("message")
        if message is not None and len(message) > _MESSAGE_CHARS:
            source["message"] = message[:_MESSAGE_CHARS] + TRUNCATED_MARKER
            summary["truncated"] = True
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > MAX_FAILURE_REASON_CHARS:
        # Escaping control characters can expand otherwise short messages.
        for source in diagnostics:
            source.pop("message", None)
        summary["truncated"] = True
        encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Core fields come from finite enums/fixed endpoint names and small codes.
    # Retain a last-resort guard for unexpectedly malformed external objects.
    result = limit_failure_reason(encoded)
    assert result is not None
    return result


def collection_failure_summary(snapshot: SnapshotAssembly) -> str:
    sources = []
    for name in SnapshotSource:
        source = getattr(snapshot.sources, name.value.lower())
        issues = sorted({issue.kind.value for issue in snapshot.issues if issue.source is name})
        error = None if source is None else source.error
        if source is not None and error is None and not issues:
            continue  # Successful observations belong in research evidence, not errors.
        request = None if source is None else source.request
        sources.append({
            "source": name.value,
            "issues": issues,
            "kind": None if error is None else error.kind.value,
            "endpoint": None if request is None else request.endpoint.name,
            "phase": None if error is None or error.phase is None else error.phase.value,
            "api_code": None if error is None else error.api_code,
            "http_status": None if source is None else source.status_code,
            "elapsed_ms": None if source is None else source.elapsed_ms,
            "message": None if error is None else error.message,
        })
    return _encode({"failure_code": "COLLECTION_FAILED", "symbol": snapshot.symbol[:32],
                    "sources": sources, "truncated": len(snapshot.symbol) > 32})


def funding_failure_summary(window: "FundingWindow") -> str:
    return _encode({
        "failure_code": "FUNDING_UNAVAILABLE", "symbol": window.symbol[:32],
        "opened_at": window.opened_at.isoformat(), "closed_at": window.closed_at.isoformat(),
        "sources": [{"source": "FUNDING_HISTORY", "endpoint": "PERP_FUNDING_RATE",
                     "kind": None if window.failure is None else window.failure.value,
                     "http_status": window.pages[-1].status_code if window.pages else None,
                     "message": window.message}],
        "truncated": len(window.symbol) > 32,
    })
