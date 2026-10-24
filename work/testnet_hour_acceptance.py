"""One-hour bounded TESTNET observation: two isolated 30-minute virtual sessions.

Only BTC 0.001 and ETH 0.01 connectivity entries, never strategy profit evidence.
Uses existing managed exits, reconciliation and the unchanged acceptance config.
No entry retries, no additional strategy entries, no mainnet execution.
"""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

from restart_acceptance import save
from testnet_continuous_acceptance import run_acceptance, safe_error

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.runner import (
    ExecutionGuardError,
    create_testnet_bot,
    process_lock,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.logging import configure_logging


def run_hour(bot_factory, path):
    """A failed/early/late first session blocks the second connectivity entry."""
    with process_lock(path.with_suffix(".supervisor.lock")):
        if path.exists():
            raise ExecutionGuardError("HOUR_ACCEPTANCE_REPORT_ALREADY_EXISTS")
        if any(path.with_name(f"{path.stem}-{symbol}.json").exists() for symbol in ("BTCUSDT", "ETHUSDT")):
            raise ExecutionGuardError("HOUR_SESSION_REPORT_ALREADY_EXISTS")
        started = monotonic()
        evidence = {"status": "RUNNING", "sample_type": "TESTNET_ONE_HOUR_ACCEPTANCE",
                    "started_at": utc_now(), "pid": os.getpid(), "planned_observation_seconds": 3600,
                    "new_entry_limit": 2, "decision_bypassed": True, "sessions": [],
                    "meaning": "Virtual execution/observation only, not profitability proof"}
        save(path, evidence)
        try:
            for symbol in ("BTCUSDT", "ETHUSDT"):
                session_path = path.with_name(f"{path.stem}-{symbol}.json")
                if session_path.exists():
                    raise ExecutionGuardError("HOUR_SESSION_REPORT_ALREADY_EXISTS")
                evidence.update(current_symbol=symbol, updated_at=utc_now())
                save(path, evidence)
                bot = bot_factory()  # Fresh stop event; original journal/account guards.
                result = run_acceptance(bot, session_path, duration_seconds=1800, entry_symbol=symbol)
                session = json.loads(session_path.read_text(encoding="utf-8"))
                # Existing narrow execution PASS alone does not prove a full duration.
                duration_ok = (session.get("duration_deadline_reached") is True
                               and session.get("loop_elapsed_ms", 0) >= 1800000)
                session_ok = result == 0 and session.get("status") == "PASS" and duration_ok
                evidence["sessions"].append({"symbol": symbol, "report": str(session_path),
                    "status": session.get("status"), "duration_complete": duration_ok,
                    "attempt_id": session.get("attempt_id"), "rounds": session.get("rounds"),
                    "cycle_status_counts": session.get("cycle_status_counts"),
                    "exit_lateness_seconds": session.get("exit_lateness_seconds")})
                if not session_ok:
                    evidence.update(status="NEEDS_ATTENTION", reason="SESSION_FAILED_OR_STOPPED_EARLY")
                    return 1
                save(path, evidence)
            evidence.update(status="PASS", observation_duration_complete=True)
            return 0
        except Exception as error:
            evidence.update(status="NEEDS_ATTENTION", **safe_error(error))
            return 1
        except KeyboardInterrupt:
            evidence.update(status="NEEDS_ATTENTION", reason="INTERRUPTED")
            return 1
        finally:
            evidence.update(finished_at=utc_now(), elapsed_ms=int((monotonic() - started) * 1000))
            save(path, evidence)
            print(exact_json(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("docs/testnet-funding-acceptance.toml"))
    args = parser.parse_args()
    try:
        if os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
            raise ExecutionGuardError("TESTNET_ORDERS_DISABLED")
        settings = load_settings(args.config)
        configure_logging(Path("logs/testnet") / args.report.stem, settings.log_level)
        environment = execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ)
        probe = create_testnet_bot(settings, environ=environment)
        if len(probe.journal.plans()) + 2 > probe.config.max_total_attempts:
            raise ExecutionGuardError("HOUR_ACCEPTANCE_REQUIRES_TWO_REMAINING_ATTEMPTS")
        return run_hour(lambda: create_testnet_bot(settings, environ=environment), args.report)
    except Exception as error:
        failure = {"status": "NEEDS_ATTENTION", **safe_error(error)}
        save(args.report.with_suffix(".failure.json"), failure)
        print(exact_json(failure), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
