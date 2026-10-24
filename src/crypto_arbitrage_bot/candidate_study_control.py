"""Prepare fixed public candidate studies and read all resulting assessments."""

import argparse
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from crypto_arbitrage_bot.analytics.registered_study import read_registered_study
from crypto_arbitrage_bot.candidate_diagnostics import DiagnosticLimits, diagnose_candidate_exits
from crypto_arbitrage_bot.candidate_exit_audit import ExitAuditLimits, audit_candidate_exits
from crypto_arbitrage_bot.candidate_study import load_index, prepare_candidate_index, write_new
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed candidate preregistration and read-only results; no orders")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--symbols", nargs="+", required=True)
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--output-config", type=Path, required=True)
    prepare.add_argument("--id", required=True)
    prepare.add_argument("--study-start", type=datetime.fromisoformat, required=True)
    prepare.add_argument("--study-end", type=datetime.fromisoformat, required=True)
    prepare.add_argument("--entry-separation-seconds", type=int, required=True)
    prepare.add_argument("--minimum-entries", type=int, required=True)
    prepare.add_argument("--minimum-days", type=int, required=True)
    prepare.add_argument("--mean-return", type=Decimal, required=True)
    prepare.add_argument("--median-return", type=Decimal, required=True)
    prepare.add_argument("--maximum-loss-return", type=Decimal, required=True)
    prepare.add_argument("--entry-interval-seconds", type=int, required=True)
    prepare.add_argument("--maximum-sampling-gap-seconds", type=int, required=True)
    prepare.add_argument("--maximum-capture-delay-seconds", type=int, required=True)
    report = sub.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    diagnostics = sub.add_parser("diagnostics", help="Read failed tasks and public exit archives; no refresh or qualification")
    diagnostics.add_argument("--config", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument("--max-files", type=int, default=5000)
    diagnostics.add_argument("--max-tasks", type=int, default=10000)
    diagnostics.add_argument("--max-total-bytes", type=int, default=512*1024*1024)
    diagnostics.add_argument("--max-seconds", type=int, default=60)
    audit = sub.add_parser("audit-exits", help="Recompute archived public exits with original parsers/validator; no orders")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--max-files", type=int, default=5000)
    audit.add_argument("--max-records", type=int, default=10000)
    audit.add_argument("--max-total-bytes", type=int, default=512*1024*1024)
    audit.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command != "prepare" and args.output.exists():
            raise FileExistsError("NEW_REPORT_OUTPUT_REQUIRED")
        if args.command == "prepare":
            if args.output_config.exists():
                raise ValueError("NEW_BOUND_CONFIG_REQUIRED")
            path, sha = prepare_candidate_index(settings, args.directory, symbols=tuple(args.symbols),
                id_prefix=args.id, study_start=args.study_start, study_end=args.study_end,
                min_entry_separation=timedelta(seconds=args.entry_separation_seconds),
                min_entries=args.minimum_entries, min_days=args.minimum_days,
                min_mean_return=args.mean_return, min_median_return=args.median_return,
                max_loss_return=args.maximum_loss_return,
                max_capture_delay=timedelta(seconds=args.maximum_capture_delay_seconds),
                max_sampling_gap=timedelta(seconds=args.maximum_sampling_gap_seconds),
                entry_interval_seconds=args.entry_interval_seconds)
            # Only the new copy gains a reference. Existing policy and source
            # TOML bytes remain unchanged; no live study is rebound.
            text = args.config.read_text(encoding="utf-8")
            import json
            text += "\n[research.study_registration]\npath = " + json.dumps(str(path)) + "\nsha256 = " + json.dumps(sha) + "\n"
            args.output_config.parent.mkdir(parents=True, exist_ok=True)
            with args.output_config.open("x", encoding="utf-8") as output:
                output.write(text)
            load_index(load_settings(args.output_config))
            result = {"status": "CANDIDATES_PREREGISTERED", "index": str(path), "sha256": sha,
                      "bound_config": str(args.output_config.resolve()), "orders_submitted": False}
        elif args.command == "audit-exits":
            result = audit_candidate_exits(settings, evaluated_at=utc_now(), limits=ExitAuditLimits(
                args.max_files, args.max_records, args.max_total_bytes, args.max_seconds))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_new(args.output, result)
        elif args.command == "diagnostics":
            result = diagnose_candidate_exits(settings, evaluated_at=utc_now(), limits=DiagnosticLimits(
                args.max_files, args.max_tasks, args.max_total_bytes, args.max_seconds))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_new(args.output, result)
        else:
            index, studies = load_index(settings)
            now = utc_now()
            assessments = {symbol: read_registered_study(study.path, study.sha256, evaluated_at=now)
                           for symbol, study in studies.items()}
            result = {"status": "CANDIDATE_STUDIES_READ", "evaluated_at": now,
                      "symbols": index["symbols"], "evidence": assessments,
                      "trade_allowed": False, "engineering_qualification_issued": False,
                      "credentials_read": False, "orders_submitted": False}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_new(args.output, result)
        print(exact_json(result))
        return 1 if result.get("status") == "INCOMPLETE" else 0
    except Exception as error:
        print(exact_json({"status": "UNAVAILABLE", "error_type": type(error).__name__,
                          "orders_submitted": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
