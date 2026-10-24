"""Explicit CLI for isolated Binance testnet operation; never production trading."""

import argparse
import os
from decimal import Decimal
from pathlib import Path

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    CredentialStoreError,
    execution_environment,
    setup_credentials,
)
from crypto_arbitrage_bot.execution.runner import BotCycle, ExecutionGuardError, create_testnet_bot
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.logging import configure_logging


def print_cycle(cycle: BotCycle) -> None:
    """Display computed results and dates so operators know why no entry occurs."""
    funding = "未知" if cycle.estimated_funding is None else format(cycle.estimated_funding, ".6f")
    profit = "未知" if cycle.estimated_net_profit is None else format(cycle.estimated_net_profit, ".6f")
    next_time = "未知" if cycle.next_funding_at is None else cycle.next_funding_at.isoformat()
    reason = cycle.reason or cycle.status
    codes = ",".join(cycle.decision_reason_codes)
    print(f"TESTNET {cycle.symbol} {cycle.status} | 预计Funding={funding} USDT | "
          f"预计净收益={profit} USDT | 下次结算(UTC)={next_time} | {reason} {codes}", flush=True)
    if cycle.pair is not None and cycle.planned_exit_at is not None:
        print(f"  虚拟仓位按保存计划管理，预计退出(UTC)={cycle.planned_exit_at.isoformat()}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Binance TESTNET bot: virtual test funds only")
    parser.add_argument("--config", type=Path, default=Path("docs/testnet-funding.toml"))
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIAL_FILE,
                        help="Windows-user encrypted testnet credential file")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--demo", action="store_true", help="Offline mocked round trip; no keys or network")
    commands.add_argument("--setup-credentials", action="store_true", help="Hidden local testnet key setup; no network or orders")
    commands.add_argument("--check", action="store_true", help="Read testnet markets and accounts, submit no orders")
    commands.add_argument("--prepare", action="store_true", help="Set flat USDT-M Demo symbols to exactly 1x; no orders")
    commands.add_argument("--round-trip", action="store_true", help="Explicit test-money entry and immediate close")
    commands.add_argument("--once", action="store_true", help="One research/managed-exit/eligible-entry cycle")
    commands.add_argument("--forever", action="store_true", help="Poll research and managed exits until interrupted")
    commands.add_argument("--funding-acceptance", action="store_true",
                         help="One controlled virtual-money settlement test; explicitly bypasses strategy selection")
    commands.add_argument("--reconcile", action="store_true",
                         help="Read closed test-account fills/fees/funding; submit no orders")
    commands.add_argument("--accounting-history", action="store_true",
                         help="Read existing local accounting revisions; no keys, network or writes")
    commands.add_argument("--reconcile-after-acceptance", action="store_true",
                         help="Bounded read-only accounting after the specified acceptance report finishes")
    commands.add_argument("--export-risk-proof", action="store_true",
                         help="Read existing captured acceptance and native archives; no credentials, network or orders")
    commands.add_argument("--export-captured-case", action="store_true",
                         help="Read pinned preregistration and original trace for one risk case; no orders")
    commands.add_argument("--capture-terminal-restart", action="store_true",
                         help="Replay original terminal recovery in this new native process; journal read-only and gateways blocked")
    parser.add_argument("--capture-registration", type=Path)
    parser.add_argument("--capture-registration-sha256")
    parser.add_argument("--capture-trace", type=Path)
    parser.add_argument("--capture-trace-sha256")
    parser.add_argument("--capture-trace-output", type=Path)
    parser.add_argument("--risk-proof-output", type=Path,
                        help="New immutable file for a single case proof, not an execution qualification")
    parser.add_argument("--attempt-id", help="Accounting for this existing execution attempt only")
    parser.add_argument("--capital", type=Decimal,
                        help="Optional explicit USDT capital budget for accounting return, not an account balance")
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), default="ETHUSDT")
    parser.add_argument("--acceptance-report", type=Path,
                        default=Path("data/testnet/funding-acceptance.json"),
                        help="Durable acceptance report; reuse this exact path to resume")
    parser.add_argument("--acceptance-max-wait-seconds", type=int, default=14400,
                        help="Bounded acceptance wait, 1..28800 seconds; default four hours")
    parser.add_argument("--acceptance-preflight-attempts", type=int, default=1,
                        help="1..5 read-only preflight attempts; never retries order submission")
    parser.add_argument("--account-after-acceptance", action="store_true",
                        help="Archive actual fills/fees after this controlled acceptance finishes")
    parser.add_argument("--witness-directory", type=Path,
                        help="Preregister and capture this controlled acceptance; no budget override or risk qualification")
    parser.add_argument("--capture-accounting", action="store_true",
                        help="Capture existing pre/post wallet accounting before witness closure; requires balance evidence")
    parser.add_argument("--capture-case", choices=[case.value for case in ExecutionRiskCase],
                        help="Fix a risk case before entry; full witness/accounting capture required, no qualification")
    args = parser.parse_args()
    if args.accounting_history and not args.attempt_id:
        parser.error("--accounting-history requires --attempt-id")
    if args.account_after_acceptance and not args.funding_acceptance:
        parser.error("--account-after-acceptance requires --funding-acceptance")
    if args.witness_directory is not None and not args.funding_acceptance:
        parser.error("--witness-directory requires --funding-acceptance")
    if args.capture_accounting and (not args.funding_acceptance or args.witness_directory is None
                                    or args.account_after_acceptance):
        parser.error("--capture-accounting requires witnessed funding acceptance and excludes --account-after-acceptance")
    if args.capture_case is not None and not args.capture_accounting:
        parser.error("--capture-case requires --capture-accounting")
    if (args.risk_proof_output is not None) != (args.export_risk_proof or args.export_captured_case):
        parser.error("--risk-proof-output is required only with a proof export command")
    capture_arguments = (args.capture_registration, args.capture_registration_sha256,
                         args.capture_trace, args.capture_trace_sha256)
    if args.export_captured_case and not all(capture_arguments):
        parser.error("--export-captured-case requires pinned registration and trace")
    if not (args.export_captured_case or args.capture_terminal_restart) and any(value is not None for value in capture_arguments):
        parser.error("capture references require --export-captured-case")
    if args.capture_terminal_restart and (not args.capture_registration or not args.capture_registration_sha256
            or args.capture_trace is not None or args.capture_trace_sha256 is not None):
        parser.error("--capture-terminal-restart requires pinned registration only")
    if (args.capture_trace_output is not None) != args.capture_terminal_restart:
        parser.error("--capture-trace-output is required only with --capture-terminal-restart")
    try:
        if args.setup_credentials:
            setup_credentials(args.credentials_file)
            return 0
        settings = load_settings(args.config)
        if args.capture_accounting and (settings.testnet is None or not settings.testnet.balance_evidence_enabled):
            raise ExecutionGuardError("CAPTURE_ACCOUNTING_REQUIRES_PREENTRY_BALANCE_EVIDENCE")
        if args.accounting_history:
            from crypto_arbitrage_bot.execution.pair_coordinator import SQLitePairExecutionJournal
            from crypto_arbitrage_bot.execution.reconciliation_repository import (
                ReconciliationRepository,
            )
            if settings.testnet is None:
                raise ExecutionGuardError("accounting history requires a TESTNET journal configuration")
            archive = ReconciliationRepository(SQLitePairExecutionJournal(settings.testnet.journal_path))
            revisions = archive.list_report_history(args.attempt_id)
            records = []
            for revision in revisions:
                report = revision["report"]
                calculation = report.get("calculation")
                records.append({**{key: value for key, value in revision.items() if key != "report"},
                    "funding_total": report.get("funding_total"), "price_pnl": report.get("price_pnl"),
                    "fees": report.get("fees"), "issues": report.get("issues"),
                    "net_profit": None if calculation is None else calculation.get("net_profit"),
                    "net_return": report.get("net_return")})
            print(exact_json({"scope": "LOCAL_ACCOUNTING_REPORT_HISTORY", "environment": "TESTNET",
                "status": "FOUND" if records else "NOT_FOUND", "attempt_id": args.attempt_id,
                "records": records, "orders_submitted": False, "credentials_read": False,
                "risk_qualification_issued": False}))
            return 0 if records else 1
        if args.capture_terminal_restart:
            from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
            from crypto_arbitrage_bot.execution.risk_witness import RiskWitness, load_witness
            from crypto_arbitrage_bot.utils.clock import utc_now
            assert args.capture_trace_output is not None
            reference = StudyRegistrationSettings(path=args.capture_registration, sha256=args.capture_registration_sha256)
            try:
                if args.capture_trace_output.exists():
                    raise FileExistsError("NEW_TRACE_OUTPUT_REQUIRED")
                witness = RiskWitness(load_witness(reference), settings, clock=utc_now)
                restart_result = witness.capture_terminal_restart()
                trace = witness.export_trace(args.capture_trace_output)
            except Exception:
                print(exact_json({"stage": "TERMINAL_RESTART_CAPTURE", "success": False,
                    "failure": "TERMINAL_RESTART_CAPTURE_REJECTED", "orders_submitted": False,
                    "credentials_read": False, "risk_qualification_issued": False}))
                return 1
            print(exact_json({"stage": "TERMINAL_RESTART_CAPTURE", "success": True, "result": restart_result,
                "trace": trace.model_dump(mode="json"), "orders_submitted": False,
                "credentials_read": False, "risk_qualification_issued": False}))
            return 0
        if args.export_captured_case:
            from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
            from crypto_arbitrage_bot.execution.captured_case_proof import (
                export_captured_case_proof,
            )
            assert args.risk_proof_output is not None
            exported = export_captured_case_proof(settings,
                StudyRegistrationSettings(path=args.capture_registration, sha256=args.capture_registration_sha256),
                StudyRegistrationSettings(path=args.capture_trace, sha256=args.capture_trace_sha256),
                args.risk_proof_output)
            print(exact_json({"stage": "NATIVE_CAPTURED_CASE_PROOF", "result": {
                "success": exported.success, "failure": exported.failure,
                "reference": None if exported.reference is None else exported.reference.model_dump(mode="json")},
                "orders_submitted": False, "credentials_read": False, "risk_qualification_issued": False}))
            return 0 if exported.success else 1
        if args.export_risk_proof:
            from crypto_arbitrage_bot.execution.acceptance_proof import export_acceptance_proof
            assert args.risk_proof_output is not None
            exported = export_acceptance_proof(settings, args.acceptance_report, args.risk_proof_output)
            print(exact_json({"stage": "NATIVE_ACCEPTANCE_PROOF", "result": {
                "success": exported.success, "failure": exported.failure,
                "reference": None if exported.reference is None else exported.reference.model_dump(mode="json")},
                              "orders_submitted": False, "risk_qualification_issued": False}))
            return 0 if exported.success else 1
        if args.demo:
            from crypto_arbitrage_bot.execution.demo import run_demo
            return run_demo(settings)
        if not (args.check or args.prepare or args.reconcile or args.reconcile_after_acceptance) and os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
            print("TESTNET_ORDERS_DISABLED: explicitly set CAB_TESTNET_ORDERS_ENABLED=true for virtual-money orders")
            return 2
        configure_logging(settings.log_dir, settings.log_level)
        bot = create_testnet_bot(settings, environ=execution_environment(args.credentials_file, os.environ))
        if args.prepare:
            for symbol in settings.symbols:
                bot.prepare(symbol)
                print(exact_json({"environment": "TESTNET", "symbol": symbol,
                                 "status": "PREPARED_1X", "orders_submitted": False}))
        elif args.check:
            for symbol in settings.symbols:
                assembly = bot.check(symbol)
                print(exact_json({"environment": "TESTNET", "symbol": symbol, "snapshot": assembly.status,
                                 "orders_submitted": False}))
        elif args.round_trip:
            assert settings.research is not None
            for symbol in settings.symbols:
                quantity = settings.research.sizing[symbol].target_quantity
                if quantity is None:
                    raise ValueError("round-trip requires explicit target_quantity")
                result = bot.round_trip(symbol, quantity)
                print(exact_json(result))
                if result.state.value != "CLOSED":
                    return 1
        elif args.once:
            cycles = bot.run_once()
            for cycle in cycles:
                print(exact_json(cycle))
            if any(cycle.status in ("RESEARCH_FAILED", "ENTRY_BLOCKED", "RECOVERY_REQUIRED", "FAILED") for cycle in cycles):
                return 1
        elif args.reconcile_after_acceptance:
            from crypto_arbitrage_bot.execution.acceptance_accounting import (
                reconcile_after_acceptance,
            )
            outcome = reconcile_after_acceptance(bot, args.acceptance_report)
            print(exact_json(outcome), flush=True)
            return 0 if outcome.status.value == "SAVED" else 1
        elif args.reconcile:
            from crypto_arbitrage_bot.execution.reconciliation_repository import (
                ReconciliationRepository,
            )
            from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
            archive = ReconciliationRepository(bot.journal)
            attempt_ids = [args.attempt_id] if args.attempt_id else [plan.attempt_id for plan, _ in bot.journal.plans()
                if (pair := bot.journal.get(plan.attempt_id)) is not None and pair.state.value == "CLOSED"]
            all_saved = True
            for attempt_id in attempt_ids:
                accounting = reconcile_closed_attempt(bot, attempt_id, capital=args.capital)
                saved = archive.save(accounting)
                all_saved = all_saved and saved.success
                print(exact_json({"stage": "TESTNET_TRADE_ACCOUNTING", "attempt_id": attempt_id,
                    "status": accounting.status, "issues": accounting.issues, "saved": saved,
                    "net_profit": None if accounting.calculation is None else accounting.calculation.net_profit,
                    "net_return": accounting.net_return, "funding_total": accounting.funding_total,
                    "orders_submitted": False}), flush=True)
            return 0 if all_saved else 1
        elif args.funding_acceptance:
            from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
            print("TESTNET CONNECTIVITY ACCEPTANCE: one virtual pair, not strategy PASS or profit proof; "
                  "waiting for official settlement, then managed exit and account funding check.", flush=True)
            acceptance = FundingAcceptance(bot, args.acceptance_report, symbol=args.symbol,
                max_wait_seconds=args.acceptance_max_wait_seconds,
                max_preflight_attempts=args.acceptance_preflight_attempts)
            if args.capture_accounting:
                if args.capture_case is not None:
                    status = acceptance.run(witness_directory=args.witness_directory, capture_accounting=True,
                                            witness_case=ExecutionRiskCase(args.capture_case))
                else:
                    status = acceptance.run(witness_directory=args.witness_directory, capture_accounting=True)
            else:
                status = acceptance.run() if args.witness_directory is None else acceptance.run(witness_directory=args.witness_directory)
            witness_state = acceptance.state if isinstance(acceptance.state, dict) else {}
            print(exact_json({"stage": "FUNDING_ACCEPTANCE", "status": status,
                              "report": str(args.acceptance_report), "environment": "TESTNET"}), flush=True)
            if args.witness_directory is not None or "witness_registration" in witness_state:
                print(exact_json({"stage": "ACCEPTANCE_WITNESS", "status": witness_state.get("witness_status", "NOT_CAPTURED"),
                    "reason": witness_state.get("witness_failure"), "origin": witness_state.get("witness_origin"),
                    "trace": witness_state.get("witness_trace"), "risk_qualification_issued": False}), flush=True)
            if args.account_after_acceptance:
                from crypto_arbitrage_bot.execution.acceptance_accounting import (
                    AccountingFollowupStatus,
                    reconcile_after_acceptance,
                )
                followup = reconcile_after_acceptance(bot, args.acceptance_report, max_wait_seconds=1)
                print(exact_json({"stage": "ACCEPTANCE_ACCOUNTING", "result": followup,
                                  "orders_submitted": False}), flush=True)
                if (followup.status is not AccountingFollowupStatus.SAVED
                        or followup.accounting_status != "RECONCILED"):
                    return 1
            capture_ok = (args.witness_directory is None and "witness_registration" not in witness_state
                          or witness_state.get("witness_status") == "EXPORTED")
            if args.capture_accounting:
                captured = witness_state.get("captured_accounting")
                linked = (isinstance(captured, dict)
                    and captured.get("attempt_id") == witness_state.get("attempt_id"))
                audit_id = captured.get("audit_id") if isinstance(captured, dict) else None
                audit_saved = (isinstance(audit_id, str) and len(audit_id) == 64
                    and all(char in "0123456789abcdef" for char in audit_id))
                captured_status = captured.get("status") if isinstance(captured, dict) else None
                balance_status = captured.get("balance_status") if isinstance(captured, dict) else None
                summary = {"status": captured_status if linked and captured_status in ("ARCHIVED", "FAILED", "SKIPPED")
                    else "INVALID_CAPTURED_ACCOUNTING", "audit_saved": audit_saved and linked,
                    "balance_status": balance_status if linked and balance_status in ("MATCHED", "MISMATCH", "INVALID", "INCOMPLETE") else None}
                print(exact_json({"stage": "CAPTURED_ACCOUNTING", "result": summary,
                    "orders_submitted": False, "risk_qualification_issued": False}), flush=True)
                capture_ok = (capture_ok and linked and audit_saved and isinstance(captured, dict)
                    and captured_status == "ARCHIVED" and balance_status == "MATCHED"
                    and captured.get("error_code") is None)
            return 0 if status == "VERIFIED" and capture_ok else 1
        else:
            print("TESTNET 虚拟资金运行：预计Funding是假设，实际入账另行核实；未满足门槛不会开仓。", flush=True)
            bot.status_callback = print_cycle
            bot.run_forever()
        return 0
    except KeyboardInterrupt:
        print("Stopped. Persisted open/uncertain positions require restart/check; no blind liquidation.")
        return 130
    except Exception as error:
        # Never echo arbitrary exception URLs, credentials or request bodies.
        details = {"environment": "TESTNET", "status": "FAILED", "error_type": type(error).__name__}
        if isinstance(error, TestnetExecutionError):
            details.update({"kind": error.kind.value, "reason": error.message})
        elif isinstance(error, ExecutionGuardError):
            details["reason"] = str(error)[:240]
        elif isinstance(error, CredentialStoreError):
            details["reason"] = str(error)[:240]
        print(exact_json(details))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
