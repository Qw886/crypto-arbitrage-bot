"""Bounded read-only accounting after an existing controlled test finishes."""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot


class AccountingFollowupStatus(StrEnum):
    SAVED = "SAVED"
    SKIPPED = "SKIPPED"
    INVALID_REPORT = "INVALID_REPORT"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class AccountingFollowupResult:
    status: AccountingFollowupStatus
    attempt_id: str | None = None
    accounting_status: str | None = None
    error_code: str | None = None


def reconcile_after_acceptance(bot: "TestnetBot", report_path: Path,
                              *, max_wait_seconds: int = 14400) -> AccountingFollowupResult:
    """No orders, lock acquisition or account requests while waiting.

    Use the exact existing acceptance report; never create/restart execution.
    Accounting has its own outcome file. Failure doesn't alter acceptance or
    position management. SAVED means archived, not necessarily RECONCILED.
    """
    if type(max_wait_seconds) is not int or not 1 <= max_wait_seconds <= 14400:
        raise ValueError("accounting wait must be bounded to four hours")
    outcome_path = report_path.with_suffix(".accounting.json")
    deadline = monotonic_ns() + max_wait_seconds * 1_000_000_000
    def finish(result: AccountingFollowupResult, summary: dict[str, object] | None = None) -> AccountingFollowupResult:
        temporary = outcome_path.with_suffix(outcome_path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(exact_json({"checked_at": bot.clock(), "outcome": result,
            "orders_submitted": False, "accounting_summary": summary}), encoding="utf-8")
        temporary.replace(outcome_path)
        return result
    while not bot.stop.is_set() and monotonic_ns() < deadline:
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                if (data.get("sample_type") != "TESTNET_CONNECTIVITY_ACCEPTANCE"
                    or data.get("journal_path") != str(bot.config.journal_path.resolve())
                    or data.get("symbol") not in bot.settings.symbols
                    or not isinstance(data.get("attempt_id"), str)):
                    return finish(AccountingFollowupResult(AccountingFollowupStatus.INVALID_REPORT))
                attempt_id = data["attempt_id"]
                if data["status"] in ("VERIFIED", "UNAVAILABLE"):
                    plans = [plan for plan, _ in bot.journal.plans() if plan.attempt_id == attempt_id]
                    if len(plans) != 1 or plans[0].symbol != data["symbol"]:
                        return finish(AccountingFollowupResult(AccountingFollowupStatus.INVALID_REPORT, attempt_id))
                    pair = bot.journal.get(attempt_id)
                    if pair is None or pair.state is not PairExecutionState.CLOSED:
                        return finish(AccountingFollowupResult(AccountingFollowupStatus.SKIPPED,
                                                               attempt_id, error_code="ATTEMPT_NOT_CLOSED"))
                    report = reconcile_closed_attempt(bot, attempt_id)
                    if report.symbol != data["symbol"]:
                        return finish(AccountingFollowupResult(AccountingFollowupStatus.INVALID_REPORT, attempt_id))
                    saved = ReconciliationRepository(bot.journal).save(report)
                    return finish(AccountingFollowupResult(AccountingFollowupStatus.SAVED if saved.success else
                        AccountingFollowupStatus.FAILED, attempt_id, report.status.value,
                        None if saved.error_code is None else saved.error_code.value),
                        {"price_pnl": report.price_pnl, "fees": report.fees, "funding_total": report.funding_total,
                         "net_profit": None if report.calculation is None else report.calculation.net_profit,
                         "net_return": report.net_return, "issues": report.issues,
                         "scope": report.accounting_scope})
                if data["status"] in ("ABORTED", "NEEDS_ATTENTION", "CANCELLED"):
                    return finish(AccountingFollowupResult(AccountingFollowupStatus.SKIPPED, attempt_id,
                                                           error_code=data["status"]))
                if data["status"] not in ("WAITING", "HOLDING", "CHECKING_FUNDING"):
                    return finish(AccountingFollowupResult(AccountingFollowupStatus.INVALID_REPORT, attempt_id))
            except (ValueError, KeyError, TypeError):
                return finish(AccountingFollowupResult(AccountingFollowupStatus.INVALID_REPORT))
            except Exception as error:
                return finish(AccountingFollowupResult(AccountingFollowupStatus.FAILED, error_code=type(error).__name__))
        if bot.stop.wait(bot.settings.poll_interval_seconds):
            break
    return finish(AccountingFollowupResult(AccountingFollowupStatus.STOPPED if bot.stop.is_set() else
                                          AccountingFollowupStatus.TIMED_OUT))
