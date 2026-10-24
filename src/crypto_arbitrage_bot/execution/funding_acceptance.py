"""One controlled virtual-money settlement test, never a strategy PASS.

Reuse the execution journal, coordinator, exits and account ledger. No PnL,
prediction, trading threshold or production endpoint is implemented here.
"""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic_ns
from typing import Literal
from uuid import uuid4

from crypto_arbitrage_bot.arbitrage.funding_projection import project_next_funding
from crypto_arbitrage_bot.execution.funding_ledger import reconcile_funding_ledger
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, TestnetBot, process_lock
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind, TestnetExecutionError
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.runner import CycleStatus
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

_DONE = frozenset(("VERIFIED", "UNAVAILABLE", "ABORTED", "NEEDS_ATTENTION", "CANCELLED"))


class FundingAcceptance:
    """A durable single test; restart never creates another attempt.

    Call tick under the shared execution process lock. Automatic opportunity
    selection is bypassed explicitly for connectivity evidence only. The saved
    research Decision is left unchanged, including REJECT.
    """

    def __init__(self, bot: TestnetBot, report_path: Path, *,
                 symbol: Literal["BTCUSDT", "ETHUSDT"] = "ETHUSDT",
                 entry_lead_seconds: int = 180, max_wait_seconds: int = 14400,
                 max_preflight_attempts: int = 1, preflight_retry_seconds: int = 15) -> None:
        if (symbol not in bot.settings.symbols or type(max_wait_seconds) is not int
                or not 1 <= max_wait_seconds <= 28800):
            raise ValueError("acceptance requires a configured symbol and a bounded wait <= 8h")
        if (type(max_preflight_attempts) is not int or not 1 <= max_preflight_attempts <= 5
                or type(preflight_retry_seconds) is not int or not 1 <= preflight_retry_seconds <= 60):
            raise ValueError("preflight retries require 1..5 attempts and 1..60 seconds delay")
        policy = bot.research.scenarios[symbol].funding_projection
        if policy is None or not policy.min_entry_lead_seconds < entry_lead_seconds <= policy.entry_window_seconds:
            raise ValueError("acceptance lead must be inside the configured funding entry window")
        self.bot, self.report_path = bot, report_path.resolve()
        self.symbol, self.policy = symbol, policy
        self.entry_lead_seconds, self.max_wait_seconds = entry_lead_seconds, max_wait_seconds
        self.max_preflight_attempts = max_preflight_attempts
        self.preflight_retry_seconds = preflight_retry_seconds
        self.state: dict[str, object] = {}
        self._phase = "ARMING"
        if self.report_path.exists():
            data = json.loads(self.report_path.read_text(encoding="utf-8"))
            if (data.get("symbol") != symbol or data.get("journal_path") != str(bot.config.journal_path.resolve())
                    or data.get("sample_type") != "TESTNET_CONNECTIVITY_ACCEPTANCE"):
                raise ValueError("acceptance report identity mismatch")
            self.state = data
            for name in ("max_preflight_attempts", "preflight_retry_seconds"):
                if name in data and data[name] != getattr(self, name):
                    raise ValueError("acceptance preflight retry policy mismatch")

    def _save(self, status: str, **fields: object) -> str:
        self.state.update(status=status, updated_at=self.bot.clock(), **fields)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(exact_json(self.state), encoding="utf-8")
        temporary.replace(self.report_path)
        return status

    def _time(self, name: str) -> datetime:
        value = self.state[name]
        return utc_timestamp(value if isinstance(value, datetime) else datetime.fromisoformat(str(value)))

    def tick(self) -> str:
        now = utc_timestamp(self.bot.clock())
        status = str(self.state.get("status", "NEW"))
        if status in _DONE:
            return status
        if status == "NEW":
            self._phase = "ARMING"
            # No mutation of account configuration or orders while arming.
            assembly = self.bot.check(self.symbol)
            for configured_symbol in self.bot.settings.symbols:
                account = self.bot.perp.get_account_state(configured_symbol)
                if account.position_quantity != 0 or account.open_order_count:
                    raise ExecutionGuardError("ACCEPTANCE_REQUIRES_FLAT_DEMO_ACCOUNT")
            assert assembly.snapshot is not None
            window = project_next_funding(assembly.snapshot, now, self.policy)
            entry_at = window.settlement_at - timedelta(seconds=self.entry_lead_seconds)
            if entry_at < now or entry_at - now > timedelta(seconds=self.max_wait_seconds):
                raise ExecutionGuardError("ACCEPTANCE_SETTLEMENT_OUTSIDE_WAIT_WINDOW")
            self.bot._assert_open_allowed(self.symbol)
            for plan, _ in self.bot.journal.plans():
                pair = self.bot.journal.get(plan.attempt_id)
                if pair is not None and pair.state not in (
                    PairExecutionState.CLOSED, PairExecutionState.FLATTENED, PairExecutionState.FAILED,
                ):
                    raise ExecutionGuardError("ACCEPTANCE_REQUIRES_NO_MANAGED_POSITION")
            return self._save("WAITING", sample_type="TESTNET_CONNECTIVITY_ACCEPTANCE", symbol=self.symbol,
                journal_path=str(self.bot.config.journal_path.resolve()), attempt_id=uuid4().hex,
                started_at=now, settlement_at=window.settlement_at, entry_at=entry_at,
                closed_at=window.closed_at, decision_bypassed=True, actual_funding=None,
                max_preflight_attempts=self.max_preflight_attempts,
                preflight_retry_seconds=self.preflight_retry_seconds, preflight_attempts=0)
        if status == "WAITING":
            attempt = str(self.state["attempt_id"])
            if self.bot.journal.get(attempt) is not None:
                return self._save("HOLDING")  # Recover a claim/POST before checking entry lateness.
            if self.bot.stop.is_set():
                return self._save("CANCELLED", reason="STOPPED_BEFORE_ENTRY")
            if now < self._time("entry_at"):
                return status
            if now >= self._time("settlement_at") - timedelta(seconds=self.policy.min_entry_lead_seconds):
                return self._save("ABORTED", reason="ACCEPTANCE_ENTRY_WINDOW_MISSED")
            if self.state.get("next_preflight_at") is not None and now < self._time("next_preflight_at"):
                return status
            attempts = int(str(self.state.get("preflight_attempts", 0)))
            if attempts >= self.max_preflight_attempts:
                return self._save("ABORTED", reason="ACCEPTANCE_PREFLIGHT_ATTEMPTS_EXHAUSTED")
            self._save("WAITING", preflight_attempts=attempts + 1, next_preflight_at=None)
            self._phase = "RESEARCH"
            cycle = self.bot.research_runner._run_symbol(self.symbol)
            if cycle.status is not CycleStatus.SAVED or cycle.opportunity_id is None:
                return self._save("ABORTED", reason="ACCEPTANCE_RESEARCH_FAILED")
            quantity = self.bot.research.sizing[self.symbol].target_quantity
            if quantity is None:
                raise ValueError("acceptance requires explicit configured target quantity")
            self._save("WAITING", opportunity_id=cycle.opportunity_id, original_decision=cycle.decision)
            self._phase = "ENTRY_PREFLIGHT"
            plan = self.bot._plan(self.symbol, quantity, attempt)
            self._phase = "SETTLEMENT_RECHECK"
            assembly = self.bot.collector.collect(self.symbol)
            if not validate_market_data(assembly, self.bot.research.data_quality, evaluated_at=self.bot.clock()).valid:
                return self._save("ABORTED", reason="INVALID_ACCEPTANCE_MARKET_DATA")
            assert assembly.snapshot is not None
            window = project_next_funding(assembly.snapshot, self.bot.clock(), self.policy)
            if window.settlement_at != self._time("settlement_at"):
                return self._save("ABORTED", reason="ACCEPTANCE_SETTLEMENT_CHANGED")
            deadline = window.settlement_at - timedelta(seconds=self.policy.min_entry_lead_seconds)
            plan = replace(plan, spot_entry=replace(plan.spot_entry, submit_before=deadline),
                perp_entry=replace(plan.perp_entry, submit_before=deadline),
                scheduled_exit_at=window.closed_at, funding_settlement_at=window.settlement_at,
                opportunity_id=cycle.opportunity_id,
                projection_json=exact_json({"acceptance_only": True, "decision_bypassed": True,
                    "research_decision": cycle.decision, "forecast_only": True,
                    "snapshot": assembly.snapshot, "funding_assumption": window}))
            self._save("WAITING", opportunity_id=cycle.opportunity_id, original_decision=cycle.decision)
            self._phase = "ENTRY"
            pair = self.bot.coordinator.open_pair(plan)
            if pair.state is not PairExecutionState.HEDGED:
                return self._save("ABORTED" if pair.state in (PairExecutionState.FAILED, PairExecutionState.FLATTENED)
                                  else "NEEDS_ATTENTION", pair=pair, reason="ACCEPTANCE_ENTRY_NOT_HEDGED")
            return self._save("HOLDING", pair=pair)
        attempt = str(self.state["attempt_id"])
        pair = self.bot.journal.get(attempt)
        if pair is None:
            return self._save("NEEDS_ATTENTION", reason="ACCEPTANCE_JOURNAL_MISSING")
        # No scanning/entries here. Existing managed exits and bounded recovery
        # handle only journalled positions when the persisted deadline arrives.
        self._phase = "MANAGED_EXIT"
        self.bot._managed_exits()
        pair = self.bot.journal.get(attempt)
        assert pair is not None
        if pair.state is not PairExecutionState.CLOSED:
            if now > self._time("closed_at") + timedelta(minutes=5):
                return self._save("NEEDS_ATTENTION", pair=pair, reason="ACCEPTANCE_EXIT_NOT_COMPLETE")
            return self._save("HOLDING", pair=pair)
        self._phase = "FUNDING_LEDGER"
        reconcile_funding_ledger(self.bot)
        with self.bot.journal._connect() as db:
            check = db.execute("SELECT status,actual_funding,failure_code FROM testnet_funding_checks WHERE attempt_id=?",
                               (attempt,)).fetchone()
        if check is not None and check["status"] in ("VERIFIED", "UNAVAILABLE"):
            # Independently confirm no residual short/orders before final success.
            self._phase = "ACCOUNT_RECHECK"
            spot = self.bot.spot.get_account_state(self.symbol)
            perp = self.bot.perp.get_account_state(self.symbol)
            if perp.position_quantity != 0 or spot.open_order_count or perp.open_order_count:
                return self._save("NEEDS_ATTENTION", pair=pair, reason="ACCEPTANCE_ACCOUNT_NOT_FLAT")
            return self._save(check["status"], pair=pair, actual_funding=check["actual_funding"],
                              funding_failure=check["failure_code"], perpetual_position=perp.position_quantity,
                              meaning="Virtual account funding evidence; not a profitability test")
        return self._save("CHECKING_FUNDING", pair=pair)

    def _handle_failure(self, error: Exception) -> str:
        fields: dict[str, object] = {"reason": type(error).__name__,
            "error_type": type(error).__name__, "failure_stage": self._phase}
        if isinstance(error, ExecutionGuardError):
            fields.update(reason=error.reason, diagnostics=error.diagnostics)
        elif isinstance(error, TestnetExecutionError):
            fields.update(reason=error.kind.value, diagnostics={
                "market": error.market, "http_status": error.http_status, "api_code": error.api_code,
                "submission_may_have_reached_exchange": error.submission_may_have_reached_exchange,
            })
        # Unknown exceptions may embed signed URLs/keys. Retain type only.
        attempt = str(self.state.get("attempt_id", ""))
        claimed = self.bot.journal.get(attempt) is not None
        transient = isinstance(error, TestnetExecutionError) and error.kind in (
            ExecutionErrorKind.NETWORK_ERROR, ExecutionErrorKind.RATE_LIMITED,
        ) and not error.submission_may_have_reached_exchange
        if isinstance(error, ExecutionGuardError) and error.reason in (
            "INVALID_TESTNET_MARKET_DATA", "MARKET_EXPIRED_DURING_PREFLIGHT",
        ):
            codes = error.diagnostics.get("quality_reason_codes", ())
            stale_codes = (
                "STALE_SPOT_DATA", "STALE_PERP_DATA", "STALE_PREMIUM_DATA",
                "STALE_OPEN_INTEREST_DATA", "SAMPLING_SKEW_TOO_LARGE",
            )
            source_map = error.diagnostics.get("sources", {})
            kinds = [source.get("error_kind") for source in source_map.values()
                     if isinstance(source, dict) and source.get("error_kind") is not None] if isinstance(source_map, dict) else []
            network_source = bool(kinds) and all(kind in ("NETWORK_ERROR", "RATE_LIMITED") for kind in kinds)
            network_codes = stale_codes + (
                "INCOMPLETE_SNAPSHOT", "SOURCE_ERROR", "MISSING_SPOT_DEPTH", "MISSING_PERP_DEPTH",
                "MISSING_SPOT_BID", "MISSING_SPOT_ASK", "MISSING_PERP_BID", "MISSING_PERP_ASK",
                "MISSING_MARK_PRICE", "MISSING_INDEX_PRICE", "MISSING_FUNDING",
                "MISSING_NEXT_FUNDING_TIME", "MISSING_OPEN_INTEREST", "MISSING_TIMESTAMP",
            )
            transient = bool(codes) and isinstance(codes, (tuple, list)) and all(
                code in (network_codes if network_source else stale_codes) for code in codes)
        retry_at = self.bot.clock() + timedelta(seconds=self.preflight_retry_seconds)
        if (transient and not claimed and not self.bot.journal.intents(attempt)
                and self._phase == "ENTRY_PREFLIGHT" and self.state.get("status") == "WAITING"
                and int(str(self.state.get("preflight_attempts", 0))) < self.max_preflight_attempts
                and retry_at < self._time("settlement_at") - timedelta(seconds=self.policy.min_entry_lead_seconds)):
            return self._save("WAITING", next_preflight_at=retry_at, last_preflight_failure=fields)
        return self._save("NEEDS_ATTENTION" if claimed else "ABORTED", **fields)

    def run(self, *, witness_directory: Path | None = None, capture_accounting: bool = False,
            witness_case: ExecutionRiskCase | None = None) -> str:
        from crypto_arbitrage_bot.execution.witness_acceptance import observe_acceptance
        if type(capture_accounting) is not bool:
            raise ValueError("explicit capture_accounting boolean required")
        if capture_accounting and (not self.bot.config.balance_evidence_enabled
                or witness_directory is None and "witness_registration" not in self.state):
            raise ValueError("captured accounting requires preregistration and pre-entry balance evidence")
        if witness_case is not None and (not isinstance(witness_case, ExecutionRiskCase) or not capture_accounting):
            raise ValueError("explicit registered risk case requires captured accounting")
        case = witness_case or ExecutionRiskCase.FEES_FUNDING_BALANCE
        already_terminal = self.state.get("status") in _DONE
        if already_terminal and witness_case is not None and "witness_registration" in self.state:
            from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
            from crypto_arbitrage_bot.execution.risk_witness import load_witness
            reference = self.state["witness_registration"]
            if not isinstance(reference, dict) or load_witness(StudyRegistrationSettings(**reference)).case is not case:
                raise ValueError("acceptance case cannot change an existing registration")
        # Bound the whole acceptance even if the wall clock moves backwards.
        budget_seconds = (self.max_wait_seconds + self.entry_lead_seconds + self.policy.exit_delay_seconds
                          + 300 + self.bot.config.funding_check_max_attempts * self.bot.config.funding_check_interval_seconds)
        deadline_ns = monotonic_ns() + budget_seconds * 1_000_000_000
        with process_lock(self.bot.config.journal_path.with_suffix(".lock")), observe_acceptance(self, witness_directory, case=case):
            while not self.bot.stop.is_set() and monotonic_ns() < deadline_ns:
                try:
                    status = self.tick()
                except Exception as error:
                    status = self._handle_failure(error)
                if status in _DONE:
                    pair = self.bot.journal.get(str(self.state.get("attempt_id", ""))) if capture_accounting else None
                    if capture_accounting and not already_terminal and pair is not None and pair.state in (
                        PairExecutionState.CLOSED, PairExecutionState.FLATTENED, PairExecutionState.FAILED,
                    ):
                        # Wallet observation belongs to the still-open witness.
                        # Reuse original targeted accounting; never reopen a
                        # completed witness or recapture a before-entry frame.
                        from crypto_arbitrage_bot.execution.balance_check_repository import (
                            BalanceCheckRepository,
                            BalanceCheckStatus,
                        )
                        from crypto_arbitrage_bot.execution.balance_ledger import (
                            reconcile_balance_ledger,
                        )
                        attempt = str(self.state["attempt_id"])
                        if self.bot.stop.is_set() or monotonic_ns() >= deadline_ns:
                            outcome: dict[str, object] = {"status": "SKIPPED", "attempt_id": attempt,
                                "error_code": "STOPPED_OR_CAPTURE_TIME_BUDGET_EXHAUSTED"}
                        else:
                            try:
                                reconcile_balance_ledger(self.bot)
                                check = BalanceCheckRepository(self.bot.journal).get(attempt)
                                outcome = {"status": "ARCHIVED" if check is not None
                                    and check.status is BalanceCheckStatus.ARCHIVED else "FAILED",
                                    "attempt_id": attempt, "audit_id": None if check is None else check.audit_id,
                                    "balance_status": None if check is None else check.balance_status,
                                    "error_code": "CAPTURE_ACCOUNTING_UNAVAILABLE" if check is None else check.failure_code}
                            except Exception as error:
                                outcome = {"status": "FAILED", "attempt_id": attempt,
                                    "error_code": type(error).__name__}
                        self._save(status, captured_accounting=outcome)
                    return status
                if self.bot.stop.wait(self.bot.settings.poll_interval_seconds):
                    break
            if not self.bot.stop.is_set():
                return self._save("NEEDS_ATTENTION" if self.bot.journal.get(str(self.state.get("attempt_id", "")))
                                  is not None else "ABORTED", reason="ACCEPTANCE_TIME_BUDGET_EXHAUSTED")
        return str(self.state.get("status", "CANCELLED"))
