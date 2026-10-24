"""Bounded, TESTNET-only process-restart acceptance; no new trading rules.

The first child exits normally after a durable HEDGED state. A fresh child
loads the same report/journal and manages its scheduled exit. This tests a
controlled process restart, not forced termination during a POST.
"""

import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_arbitrage_bot.config.settings import load_settings  # noqa: E402
from crypto_arbitrage_bot.execution.acceptance_accounting import (
    reconcile_after_acceptance,  # noqa: E402
)
from crypto_arbitrage_bot.execution.credential_store import (  # noqa: E402
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance  # noqa: E402
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState  # noqa: E402
from crypto_arbitrage_bot.execution.runner import (  # noqa: E402
    ExecutionGuardError,
    create_testnet_bot,
    process_lock,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json  # noqa: E402
from crypto_arbitrage_bot.utils.clock import utc_now  # noqa: E402
from crypto_arbitrage_bot.utils.logging import configure_logging  # noqa: E402


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(exact_json(data), encoding="utf-8")
    temporary.replace(path)


def entry_ids(pair):
    if pair is None or pair.spot_entry is None or pair.perp_entry is None:
        raise ExecutionGuardError("RESTART_ENTRY_RECEIPTS_MISSING")
    return (pair.spot_entry.client_order_id, pair.perp_entry.client_order_id)


def stop_after_hedged(acceptance, evidence_path):
    original_tick = acceptance.tick

    def tick():
        status = original_tick()
        if status == "HOLDING":
            pair = acceptance.bot.journal.get(str(acceptance.state["attempt_id"]))
            if pair is None or pair.state is not PairExecutionState.HEDGED:
                raise ExecutionGuardError("RESTART_REQUIRES_DURABLE_HEDGED_PAIR")
            save(evidence_path, {"status": "FIRST_PROCESS_EXITING_HEDGED", "pid": os.getpid(),
                "at": utc_now(), "attempt_id": pair.attempt_id, "entry_ids": entry_ids(pair),
                "pair": pair, "restart_kind": "CONTROLLED_PROCESS_RESTART"})
            acceptance.bot.stop.set()
        return status

    acceptance.tick = tick


def run_worker(args):
    settings = load_settings(args.config)
    configure_logging(settings.log_dir, settings.log_level)
    bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
    acceptance = FundingAcceptance(bot, args.report, symbol="ETHUSDT", max_wait_seconds=28800,
                                   max_preflight_attempts=5)
    evidence_path = args.report.with_suffix(".restart.json")
    if args.phase == "arm":
        if args.report.exists():
            raise ExecutionGuardError("ARM_REQUIRES_NEW_REPORT")
        # Arming calls existing flat-account/market guards and persists dates.
        # It does not enter WAITING's order-submission path.
        with process_lock(bot.config.journal_path.with_suffix(".lock")):
            status = acceptance.tick()
            baselines = {symbol: bot.spot.get_account_state(symbol).base_available
                         for symbol in settings.symbols}
            save(evidence_path, {"status": "ARMED", "at": utc_now(), "pid": os.getpid(),
                "attempt_id": acceptance.state["attempt_id"], "spot_inventory_before": baselines,
                "orders_submitted": False})
        print(exact_json(acceptance.state), flush=True)
        return 0 if status == "WAITING" else 1
    if os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
        raise ExecutionGuardError("TESTNET_ORDERS_DISABLED")
    if not args.report.exists():
        raise ExecutionGuardError("RESTART_REQUIRES_ARMED_REPORT")
    if args.phase == "first":
        stop_after_hedged(acceptance, args.report.with_suffix(".first.json"))
        status = acceptance.run()
        return 0 if status == "HOLDING" else 1
    initial = bot.journal.get(str(acceptance.state["attempt_id"]))
    if initial is None or initial.state is not PairExecutionState.HEDGED:
        raise ExecutionGuardError("RESTART_REQUIRES_DURABLE_HEDGED_PAIR")
    first = json.loads(args.report.with_suffix(".first.json").read_text(encoding="utf-8"))
    if first["pid"] == os.getpid() or tuple(first["entry_ids"]) != entry_ids(initial):
        raise ExecutionGuardError("RESTART_PROCESS_OR_ENTRY_ID_MISMATCH")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.update(status="RESUMED_HEDGED", first_pid=first["pid"], second_pid=os.getpid(),
        resumed_at=utc_now(), entry_ids=entry_ids(initial), restart_kind="CONTROLLED_PROCESS_RESTART")
    save(evidence_path, evidence)
    status = acceptance.run()
    # A failed acceptance never claims success or invents account balances.
    if status != "VERIFIED":
        evidence.update(status="NOT_VERIFIED", acceptance_status=status, finished_at=utc_now())
        save(evidence_path, evidence)
        return 1
    followup = reconcile_after_acceptance(bot, args.report, max_wait_seconds=1)
    final = bot.journal.get(initial.attempt_id)
    spot_after = {symbol: bot.spot.get_account_state(symbol) for symbol in settings.symbols}
    perp_after = {symbol: bot.perp.get_account_state(symbol) for symbol in settings.symbols}
    evidence.update(finished_at=utc_now(), acceptance_status=status, accounting=followup,
        entry_ids_unchanged=entry_ids(final) == entry_ids(initial),
        spot_inventory_after={s: a.base_available for s, a in spot_after.items()},
        perpetual_positions={s: a.position_quantity for s, a in perp_after.items()},
        open_orders={s: spot_after[s].open_order_count + perp_after[s].open_order_count for s in settings.symbols},
        final_pair=final)
    inventories_match = all(a.base_available == Decimal(evidence["spot_inventory_before"][s])
                            for s, a in spot_after.items())
    passed = (final is not None and final.state is PairExecutionState.CLOSED
        and evidence["entry_ids_unchanged"] and inventories_match
        and all(a.position_quantity == 0 for a in perp_after.values())
        and all(v == 0 for v in evidence["open_orders"].values())
        and followup.status.value == "SAVED" and followup.accounting_status == "RECONCILED")
    evidence.update(status="PASS" if passed else "NEEDS_ATTENTION", inventory_restored=inventories_match)
    save(evidence_path, evidence)
    print(exact_json(evidence), flush=True)
    return 0 if passed else 1


def supervise(args):
    # A separate lock prevents two supervisors from handing off the same pair.
    with process_lock(args.report.with_suffix(".supervisor.lock")):
        if not args.report.exists() or not args.report.with_suffix(".restart.json").exists():
            raise ExecutionGuardError("RESTART_REQUIRES_ARMED_REPORT_AND_BASELINE")
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                   "--report", str(args.report)]
        first = subprocess.run([*command, "--phase", "first"], check=False)
        if first.returncode:
            return first.returncode
        # run() released the journal lock and the OS process has actually exited.
        return subprocess.run([*command, "--phase", "resume"], check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/testnet-funding-acceptance.toml"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--phase", choices=("arm", "first", "resume", "supervise"), required=True)
    args = parser.parse_args()
    try:
        return supervise(args) if args.phase == "supervise" else run_worker(args)
    except Exception as error:
        # Do not print arbitrary exceptions containing signed URLs or credentials.
        failure = {"status": "NEEDS_ATTENTION", "phase": args.phase, "at": utc_now(),
                   "error_type": type(error).__name__}
        if isinstance(error, ExecutionGuardError):
            failure["reason"] = error.reason
        save(args.report.with_suffix(".harness-failure.json"), failure)
        print(exact_json(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
