"""Bounded continuous Demo execution acceptance, never a strategy profit sample.

One explicitly labelled virtual BTC or ETH entry; subsequent strategy entries are
blocked by the existing session-attempt gate, without changing decisions.
Duration bounds the loop; in-flight requests retain existing timeouts.
"""

import argparse
import json
import os
from collections import Counter
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from threading import Event, Timer
from time import monotonic

from restart_acceptance import save
from short_restart_acceptance import exit_timing, first_process

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.runner import (
    ExecutionGuardError,
    create_testnet_bot,
    process_lock,
)
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.logging import configure_logging


def safe_error(error):
    details = {"error_type": type(error).__name__}
    if isinstance(error, GatewayError):
        details.update(reason=error.kind.value, diagnostics=error.diagnostic_fields())
    elif isinstance(error, ExecutionGuardError):
        details.update(reason=error.reason, diagnostics=error.diagnostics)
    return details


def run_acceptance(bot, path, *, duration_seconds=600, drain_seconds=300, entry_symbol="ETHUSDT"):
    if type(duration_seconds) is not int or not 120 <= duration_seconds <= 1800:
        raise ValueError("continuous acceptance must be bounded to 120..1800 seconds")
    if type(drain_seconds) is not int or not 1 <= drain_seconds <= 300:
        raise ValueError("exit drain must be bounded to 1..300 seconds")
    with process_lock(path.with_suffix(".supervisor.lock")):
        if first_process(bot, path, entry_symbol=entry_symbol) != 0:
            return 1  # Entry receipts remain persisted; never repeat an entry.
        evidence = json.loads(path.read_text(encoding="utf-8"))
        attempt = evidence["attempt_id"]
        plans = bot.journal.plans()
        plan, opened = next(item for item in plans if item[0].attempt_id == attempt)
        due = plan.scheduled_exit_at or opened + timedelta(seconds=bot.config.holding_seconds)
        original_config, original_run = bot.config, bot.run_once
        # No new strategy entry even if research returns PASS. Exit gates and
        # original decision results are unchanged; this is a one-pair test.
        bot.config = bot.config.model_copy(update={"max_total_attempts": len(plans)})
        started = monotonic()
        counts = Counter()
        evidence.update(sample_type="TESTNET_CONTINUOUS_ACCEPTANCE", status="RUNNING",
                        pid=os.getpid(), duration_seconds=duration_seconds, rounds=0,
                        process_restart_test=False, configured_exit_due_at=due,
                        session_attempt_limit=len(plans), new_entry_limit=1,
                        continuous_started_at=bot.clock(), round_errors=[])

        def recorded_cycle():
            cycle_started = monotonic()
            try:
                cycles = original_run()
                counts.update(c.status for c in cycles)
                return cycles
            except Exception as error:
                if len(evidence["round_errors"]) < 20:
                    evidence["round_errors"].append(safe_error(error))
                evidence["unexpected_round_errors"] = evidence.get("unexpected_round_errors", 0) + 1
                raise
            finally:
                evidence.update(rounds=evidence["rounds"] + 1, updated_at=bot.clock(),
                                cycle_status_counts=dict(counts),
                                last_cycle_ms=int((monotonic() - cycle_started) * 1000),
                                elapsed_ms=int((monotonic() - started) * 1000),
                                pair_state=bot.journal.get(attempt).state.value)
                save(path, evidence)

        bot.run_once = recorded_cycle
        duration_deadline = Event()
        def stop_at_deadline():
            duration_deadline.set()
            bot.stop.set()
        timer = Timer(duration_seconds, stop_at_deadline)
        timer.daemon = True
        save(path, evidence)
        timer.start()
        try:
            bot.run_forever()  # Existing interval, exit priority and graceful shutdown.
            evidence["graceful_loop_stop"] = bot.stop.is_set()
            evidence["duration_deadline_reached"] = duration_deadline.is_set()
            evidence["loop_elapsed_ms"] = int((monotonic() - started) * 1000)
            # A deadline stops new rounds, not exposure management. Bounded
            # drain uses the original managed exit, never blind liquidation.
            deadline = monotonic() + drain_seconds
            wait = Event()
            while bot.journal.get(attempt).state is not PairExecutionState.CLOSED and monotonic() < deadline:
                with process_lock(bot.config.journal_path.with_suffix(".lock")):
                    bot._managed_exits()
                if bot.journal.get(attempt).state is PairExecutionState.CLOSED:
                    break
                wait.wait(bot.settings.poll_interval_seconds)
            pair = bot.journal.get(attempt)
            if pair.state is not PairExecutionState.CLOSED:
                evidence.update(status="NEEDS_ATTENTION", reason="EXIT_NOT_CLOSED", pair=pair)
                return 1
            accounting = reconcile_closed_attempt(bot, attempt)
            stored = ReconciliationRepository(bot.journal).save(accounting)
            save(path.with_suffix(".accounting.json"), {"report": accounting, "saved": stored})
            spot = {s: bot.spot.get_account_state(s) for s in bot.settings.symbols}
            perp = {s: bot.perp.get_account_state(s) for s in bot.settings.symbols}
            late, on_time = exit_timing(due, accounting.exit_time, Decimal("30"))
            inventory_ok = all(a.base_available == Decimal(evidence["spot_inventory_before"][s])
                               for s, a in spot.items())
            flat = all(a.position_quantity == 0 and not a.open_order_count for a in perp.values())
            no_orders = all(not a.open_order_count for a in spot.values()) and flat
            no_new_entry = len(bot.journal.plans()) == len(plans)
            verified = accounting.price_pnl is not None and accounting.fees is not None
            execution_ok = inventory_ok and no_orders and no_new_entry and verified and stored.success
            passed = execution_ok and on_time and evidence["graceful_loop_stop"]
            evidence.update(status="PASS" if passed else "NEEDS_ATTENTION",
                            execution_status="PASS" if execution_ok else "FAIL",
                            exit_timeliness="PASS" if on_time else "FAIL", pair=pair,
                            actual_exit_time=accounting.exit_time, exit_lateness_seconds=late,
                            no_new_entry=no_new_entry, inventory_restored=inventory_ok,
                            no_open_orders=no_orders, perpetual_positions={s: a.position_quantity for s, a in perp.items()},
                            accounting_status=accounting.status, accounting_issues=accounting.issues,
                            funding_total=accounting.funding_total, price_pnl=accounting.price_pnl,
                            fees=accounting.fees,
                            net_profit=None if accounting.calculation is None else accounting.calculation.net_profit,
                            meaning="Bounded virtual execution/loop acceptance, not profitability proof")
            return 0 if passed else 1
        except Exception as error:
            evidence.update(status="NEEDS_ATTENTION", **safe_error(error))
            return 1
        finally:
            timer.cancel()
            bot.stop.set()
            bot.run_once, bot.config = original_run, original_config
            evidence.update(finished_at=bot.clock(), elapsed_ms=int((monotonic() - started) * 1000))
            save(path, evidence)
            print(exact_json(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--config", type=Path, default=Path("docs/testnet-funding-acceptance.toml"))
    args = parser.parse_args()
    try:
        if os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
            raise ExecutionGuardError("TESTNET_ORDERS_DISABLED")
        if args.report.exists():
            raise ExecutionGuardError("CONTINUOUS_ACCEPTANCE_REPORT_ALREADY_EXISTS")
        settings = load_settings(args.config)
        configure_logging(settings.log_dir, settings.log_level)
        bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
        return run_acceptance(bot, args.report, duration_seconds=args.duration_seconds)
    except Exception as error:
        result = {"status": "NEEDS_ATTENTION", **safe_error(error)}
        save(args.report.with_suffix(".failure.json"), result)
        print(exact_json(result), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
