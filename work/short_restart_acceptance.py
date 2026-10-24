"""Immediate virtual-money restart test using existing managed exit logic.

No Funding settlement is required or invented. First process enters one
explicit connectivity pair; second process reads its journal and exits it.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from time import monotonic
from uuid import uuid4

from restart_acceptance import entry_ids, save

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
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    ReconciliationCode,
    reconcile_closed_attempt,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.logging import configure_logging


def exit_timing(due_at, actual_exit_at, tolerance):
    """Audit execution timeliness without changing orders or financial results."""
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("exit lateness tolerance must be a nonnegative Decimal")
    if actual_exit_at is None:
        return None, False
    late = Decimal(max(0, (actual_exit_at - due_at) // timedelta(microseconds=1))) / Decimal("1000000")
    return late, late <= tolerance


def first_process(bot, path, *, max_exit_lateness_seconds=Decimal("30"), entry_symbol="ETHUSDT"):
    if not max_exit_lateness_seconds.is_finite() or max_exit_lateness_seconds < 0:
        raise ValueError("exit lateness tolerance must be a nonnegative Decimal")
    if path.exists():
        raise ExecutionGuardError("SHORT_ACCEPTANCE_REPORT_ALREADY_EXISTS")
    allowed_quantities = {"BTCUSDT": Decimal("0.001"), "ETHUSDT": Decimal("0.01")}
    if entry_symbol not in allowed_quantities or entry_symbol not in bot.settings.symbols:
        raise ExecutionGuardError("SHORT_ACCEPTANCE_UNSUPPORTED_SYMBOL")
    with process_lock(bot.config.journal_path.with_suffix(".lock")):
        for plan, _ in bot.journal.plans():
            pair = bot.journal.get(plan.attempt_id)
            if pair is not None and pair.state not in (PairExecutionState.CLOSED,
                    PairExecutionState.FLATTENED, PairExecutionState.FAILED):
                raise ExecutionGuardError("SHORT_ACCEPTANCE_REQUIRES_NO_MANAGED_POSITION")
        inventory = {}
        for symbol in bot.settings.symbols:
            spot = bot.spot.get_account_state(symbol)
            perp = bot.perp.get_account_state(symbol)
            if spot.open_order_count or perp.open_order_count or perp.position_quantity != 0:
                raise ExecutionGuardError("SHORT_ACCEPTANCE_REQUIRES_FLAT_DEMO_ACCOUNT")
            inventory[symbol] = spot.base_available
        quantity = bot.research.sizing[entry_symbol].target_quantity
        if quantity != allowed_quantities[entry_symbol]:
            raise ExecutionGuardError("SHORT_ACCEPTANCE_REQUIRES_EXPLICIT_ETH_001" if entry_symbol == "ETHUSDT"
                                      else "SHORT_ACCEPTANCE_REQUIRES_EXPLICIT_BTC_0001")
        attempt = uuid4().hex
        plan = bot._plan(entry_symbol, quantity, attempt)
        evidence = {"status": "ENTRY_PENDING", "sample_type": "TESTNET_SHORT_RESTART_ACCEPTANCE",
            "symbol": entry_symbol, "quantity": quantity, "attempt_id": attempt,
            "first_pid": os.getpid(), "started_at": utc_now(),
            "journal_path": str(bot.config.journal_path.resolve()), "spot_inventory_before": inventory,
            "funding_settlement_test": False, "decision_bypassed": True,
            "max_exit_lateness_seconds": max_exit_lateness_seconds}
        save(path, evidence)  # Never create a second attempt on rerun.
        pair = bot.coordinator.open_pair(plan)
        evidence.update(status="HEDGED_PROCESS_EXITING" if pair.state is PairExecutionState.HEDGED
                        else "NEEDS_ATTENTION", pair=pair, at=utc_now())
        if pair.state is PairExecutionState.HEDGED:
            evidence["entry_ids"] = entry_ids(pair)
        save(path, evidence)
        print(exact_json(evidence), flush=True)
        return 0 if pair.state is PairExecutionState.HEDGED else 1


def resumed_process(bot, path):
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (evidence.get("sample_type") != "TESTNET_SHORT_RESTART_ACCEPTANCE"
            or evidence.get("journal_path") != str(bot.config.journal_path.resolve())
            or evidence.get("symbol") != "ETHUSDT" or evidence.get("first_pid") == os.getpid()):
        raise ExecutionGuardError("SHORT_RESTART_IDENTITY_MISMATCH")
    with process_lock(bot.config.journal_path.with_suffix(".lock")):
        attempt = evidence["attempt_id"]
        plans = [(plan, opened) for plan, opened in bot.journal.plans() if plan.attempt_id == attempt]
        pair = bot.journal.get(attempt)
        if (len(plans) != 1 or pair is None or pair.state is not PairExecutionState.HEDGED
                or tuple(evidence["entry_ids"]) != entry_ids(pair)):
            raise ExecutionGuardError("SHORT_RESTART_REQUIRES_ORIGINAL_HEDGED_PAIR")
        plan, opened_at = plans[0]
        due_at = plan.scheduled_exit_at or opened_at + timedelta(seconds=bot.config.holding_seconds)
        evidence.update(status="RESUMED_HEDGED", second_pid=os.getpid(), resumed_at=utc_now(),
                        opened_at=opened_at, holding_seconds=bot.config.holding_seconds,
                        configured_exit_due_at=due_at)
        save(path, evidence)
        deadline = monotonic() + 300
        while monotonic() < deadline and not bot.stop.is_set():
            bot._managed_exits()  # No research scanning, no new entry selection.
            pair = bot.journal.get(attempt)
            if pair is not None and pair.state is PairExecutionState.CLOSED:
                break
            if bot.stop.wait(bot.settings.poll_interval_seconds):
                break
        pair = bot.journal.get(attempt)
        if pair is None or pair.state is not PairExecutionState.CLOSED:
            evidence.update(status="NEEDS_ATTENTION", reason="SHORT_EXIT_NOT_CLOSED", pair=pair)
            save(path, evidence)
            return 1
        accounting = reconcile_closed_attempt(bot, attempt)
        archive = ReconciliationRepository(bot.journal).save(accounting)
        save(path.with_suffix(".accounting.json"), {"report": accounting, "saved": archive})
        spot = {s: bot.spot.get_account_state(s) for s in bot.settings.symbols}
        perp = {s: bot.perp.get_account_state(s) for s in bot.settings.symbols}
        inventory_restored = all(a.base_available == Decimal(evidence["spot_inventory_before"][s])
                                 for s, a in spot.items())
        flat = all(a.position_quantity == 0 and not a.open_order_count for a in perp.values())
        no_orders = all(not a.open_order_count for a in spot.values())
        unchanged = entry_ids(pair) == tuple(evidence["entry_ids"])
        # Missing Funding evidence stays NULL. It does not invalidate the narrow
        # restart/exit execution test, and is never labelled fully reconciled PnL.
        fills_verified = (accounting.price_pnl is not None and accounting.fees is not None
            and all(issue.code is ReconciliationCode.FUNDING_NOT_CONFIRMED for issue in accounting.issues))
        passed = inventory_restored and flat and no_orders and unchanged and fills_verified and archive.success
        late_seconds, on_time = exit_timing(due_at, accounting.exit_time, Decimal(evidence["max_exit_lateness_seconds"]))
        evidence.update(status="PASS" if passed and on_time else "TIMELINESS_FAILED" if passed else "NEEDS_ATTENTION",
            execution_status="PASS" if passed else "FAIL", exit_timeliness="PASS" if on_time else "FAIL",
            actual_exit_time=accounting.exit_time, exit_lateness_seconds=late_seconds, finished_at=utc_now(),
            pair=pair, entry_ids_unchanged=unchanged, inventory_restored=inventory_restored,
            perpetual_positions={s: a.position_quantity for s, a in perp.items()},
            spot_inventory_after={s: a.base_available for s, a in spot.items()},
            no_open_orders=flat and no_orders, accounting_status=accounting.status,
            accounting_issues=accounting.issues, fills_verified=fills_verified,
            price_pnl=accounting.price_pnl, fees=accounting.fees,
            net_profit=None if accounting.calculation is None else accounting.calculation.net_profit,
            funding_total=accounting.funding_total,
            meaning="Controlled virtual-money process restart/exit only; no profitability proof")
        save(path, evidence)
        print(exact_json(evidence), flush=True)
        return 0 if passed and on_time else 1


def supervise(args):
    with process_lock(args.report.with_suffix(".supervisor.lock")):
        if args.report.exists():
            raise ExecutionGuardError("SHORT_ACCEPTANCE_REPORT_ALREADY_EXISTS")
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                   "--report", str(args.report), "--max-exit-lateness-seconds", str(args.max_exit_lateness_seconds)]
        first = subprocess.run([*command, "--phase", "first"], check=False)
        if first.returncode:
            return first.returncode
        return subprocess.run([*command, "--phase", "resume"], check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("first", "resume", "supervise"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("docs/testnet-funding-acceptance.toml"))
    parser.add_argument("--max-exit-lateness-seconds", type=Decimal, default=Decimal("30"),
                        help="Research acceptance tolerance only; does not change execution settings")
    args = parser.parse_args()
    try:
        if os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
            raise ExecutionGuardError("TESTNET_ORDERS_DISABLED")
        if args.phase == "supervise":
            return supervise(args)
        settings = load_settings(args.config)
        configure_logging(settings.log_dir, settings.log_level)
        bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
        return first_process(bot, args.report, max_exit_lateness_seconds=args.max_exit_lateness_seconds) \
            if args.phase == "first" else resumed_process(bot, args.report)
    except Exception as error:
        failure = {"status": "NEEDS_ATTENTION", "at": utc_now(), "phase": args.phase,
                   "error_type": type(error).__name__}
        if isinstance(error, ExecutionGuardError):
            failure.update(reason=error.reason, diagnostics=error.diagnostics)
        save(args.report.with_suffix(".failure.json"), failure)
        print(exact_json(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
