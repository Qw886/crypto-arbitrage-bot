"""Finite common-period Funding/reference-cost research; no trading or SQL writes."""

import argparse
import hashlib
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from http.client import HTTPException
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.analytics.funding_comparison import (
    FundingComparisonStatus,
    compare_historical_funding,
    cost_reference_from_opportunity,
)
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.exchanges.funding_history import (
    BinanceFundingHistoryClient,
    FundingFailure,
    FundingWindow,
)
from crypto_arbitrage_bot.funding_info_research import read_funding_profile
from crypto_arbitrage_bot.models.funding_info import FundingAdjustment
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.storage.study_registry import full_research_policy
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


def research_funding_comparison(
    settings: Settings, opportunity_ids: tuple[str, ...], *, closed_at: datetime,
    lookback_seconds: tuple[int, ...], output: Path, clock: Callable[[], datetime] = utc_now,
) -> dict[str, Any]:
    """Explicit references/cutoff; keep per-source failures and never retry a trade.

    A dated estimated exit is a cost reference, not a historical exit. Funding
    comes only from actual history. Current metadata cannot reconstruct a past
    settlement schedule. Both artifacts are new immutable observations.
    """
    research = settings.research
    if research is None or research.validation_scheduler is None:
        raise ValueError("explicit research/history pagination configuration required")
    parameters_output = output.with_suffix(".parameters.json")
    if output.exists() or parameters_output.exists():
        raise FileExistsError("comparison or parameter evidence already exists")
    if (type(opportunity_ids) is not tuple or not 1 <= len(opportunity_ids) <= 10
            or any(type(i) is not str or not i for i in opportunity_ids)
            or len(set(opportunity_ids)) != len(opportunity_ids)):
        raise ValueError("provide distinct explicit opportunity ids, not best-outcome selection")
    if (type(lookback_seconds) is not tuple or not 1 <= len(lookback_seconds) <= 16
            or any(type(s) is not int or not 1 <= s <= 604800 for s in lookback_seconds)
            or len(set(lookback_seconds)) != len(lookback_seconds)):
        raise ValueError("explicit unique lookbacks, each no longer than seven days, required")
    closed, now = utc_timestamp(closed_at), utc_timestamp(clock())
    opened = closed - timedelta(seconds=max(lookback_seconds))
    if not datetime(1970, 1, 1, tzinfo=UTC) <= opened < closed <= now:
        raise ValueError("history must be a fully observed, nonnegative past window")
    repository = SQLiteOpportunityRepository(SQLiteConfig(research.sqlite_path))
    records = tuple(repository.get_opportunity_by_id(i) for i in opportunity_ids)
    if any(r is None for r in records):
        raise ValueError("an explicit reference opportunity is missing")
    present = tuple(r for r in records if r is not None)
    references = tuple(cost_reference_from_opportunity(record) for record in present)
    symbols = tuple(e.reference.symbol for e in references)
    if len(set(symbols)) != len(symbols) or any(e.reference.record_created_at > now for e in references):
        raise ValueError("references must be unique same-symbol recorded past entries")
    scheduler = research.validation_scheduler
    history_sources = []
    checked_windows = []
    for record, evidence in zip(present, references, strict=True):
        client = BinanceFundingHistoryClient(timeout_seconds=research.timeout_seconds,
            page_limit=scheduler.funding_page_limit, max_pages=scheduler.funding_max_pages,
            research_instrument=evidence.reference.research_instrument)
        try:
            window = client.fetch_window(record.symbol, opened, closed)
        except (OSError, HTTPException) as error:
            # Includes a failed socket close; no exception body/proxy value leaks.
            window = FundingWindow(record.symbol, opened, closed, failure=FundingFailure.NETWORK_ERROR,
                                   message=f"history transport failed: {type(error).__name__}")
        history_sources.append(window)
        expected = record.next_funding_time
        checked = window
        if window.failure is None:
            # Match one known archived next settlement only, just as the existing
            # validator does. Do not manufacture a full 4h/8h historical schedule.
            if expected is None:
                checked = replace(window, events=(), failure=FundingFailure.INVALID_RESPONSE,
                                  message="archived next settlement time unavailable")
            elif opened < expected <= closed:
                tolerance = timedelta(milliseconds=scheduler.funding_time_tolerance_ms)
                if sum(abs(e.settled_at - expected) <= tolerance for e in window.events) != 1:
                    checked = replace(window, events=(), failure=FundingFailure.INVALID_RESPONSE,
                                      message="known archived settlement absent or ambiguous in history")
        checked_windows.append(checked)
    parameters = read_funding_profile(settings, symbols, parameters_output, evaluated_at=clock)
    adjustments = tuple(FundingAdjustment(p["symbol"], p["adjusted_funding_rate_cap"],
        p["adjusted_funding_rate_floor"], p["funding_interval_hours"], p["observed_at"], p["disclaimer"])
        for p in parameters["profiles"] if p["adjustment_status"] == "REPORTED")
    at = utc_timestamp(clock())
    comparison = compare_historical_funding(tuple(e.reference for e in references), tuple(checked_windows),
        lookbacks=tuple(timedelta(seconds=s) for s in lookback_seconds), evaluated_at=at, adjustments=adjustments)
    status = comparison.status
    if parameters["status"] != "COMPLETE" and status is FundingComparisonStatus.COMPLETE:
        status = FundingComparisonStatus.PARTIAL
    report = {"schema_version": "common-period-funding-coverage-v1", "created_at": at,
        "scope": "DESCRIPTIVE_HISTORICAL_FUNDING_TO_DATED_REFERENCE_COST_NOT_NET_PROFIT",
        "status": status, "research_config_version": research.config_version,
        "research_policy": full_research_policy(settings), "requested_opportunity_ids": opportunity_ids,
        "lookback_seconds": lookback_seconds, "references": references, "history_sources": history_sources,
        "checked_history_failures": tuple({"symbol": w.symbol, "failure": w.failure, "reason": w.message}
                                           for w in checked_windows if w.failure is not None),
        "parameter_evidence": str(parameters_output.resolve()),
        "parameter_evidence_sha256": hashlib.sha256(parameters_output.read_bytes()).hexdigest(),
        "parameter_status": parameters["status"], "parameter_reason_codes": parameters["reason_codes"],
        "comparison": comparison, "orders_submitted": False, "credentials_read": False,
        "meaning": comparison.interpretation}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(exact_json(report) + "\n")
    horizons = []
    for index, horizon in enumerate(comparison.horizons):
        rows = []
        for row in comparison.rows:
            value = row.study.slices[index] if row.study.slices else None
            rows.append({"symbol": row.reference.symbol, "status": row.study.status,
                "reference_costs": row.reference.round_trip_costs,
                "settlement_count": None if value is None else value.settlement_count,
                "observed_funding": None if value is None else value.historical_short_funding_at_reference_quantity,
                "cost_coverage_ratio": None if value is None else value.historical_funding_to_reference_cost_ratio})
        horizons.append({"duration_seconds": lookback_seconds[index], "ranked_symbols": horizon.ranked_symbols,
                         "unranked_symbols": horizon.unranked_symbols, "rows": rows})
    return {"status": status, "history_closed_at": closed, "horizons": horizons,
            "parameter_status": parameters["status"], "history_failures": report["checked_history_failures"],
            "evidence": str(output.resolve()), "orders_submitted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--opportunity-ids", required=True, nargs="+")
    parser.add_argument("--closed-at", required=True, help="One explicit UTC/history cutoff shared by all symbols")
    parser.add_argument("--lookback-seconds", required=True, nargs="+", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = research_funding_comparison(load_settings(args.config), tuple(args.opportunity_ids),
            closed_at=datetime.fromisoformat(args.closed_at), lookback_seconds=tuple(args.lookback_seconds), output=args.output)
        print(exact_json(result))
        return 0 if result["status"] is FundingComparisonStatus.COMPLETE else 1
    except (OSError, ValueError, TypeError, KeyError, RepositoryError) as error:
        print(exact_json({"status": "FUNDING_COMPARISON_FAILED", "error_type": type(error).__name__,
                          "orders_submitted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
