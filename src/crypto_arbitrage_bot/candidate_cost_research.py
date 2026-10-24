"""Finite public candidate cost research with the existing SQLite builder/repository."""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from crypto_arbitrage_bot.arbitrage.candidate_estimate import compare_candidate_windows
from crypto_arbitrage_bot.candidate_research import collect_catalog
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.market_data.candidate_catalog import CandidateCatalog
from crypto_arbitrage_bot.market_data.candidate_market import (
    CandidateMarketCollector,
    CandidateObservation,
)
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.persistence import PersistenceStatus, build_and_save_opportunity
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.utils.clock import utc_now


def research_once(settings: Settings, symbols: tuple[str, ...], output: Path,
                  *, catalog: CandidateCatalog | None = None) -> dict[str, object]:
    """No scheduler/executor. One capture per symbol, explicitly configured windows.

    Metadata is retained once in JSON; source books/estimate/scenarios/Decision
    use the existing lossless SQLite opportunity codec. Entry time is the
    estimate's explicit opening time; created_at remains the actual save time.
    Future exits and funding hypotheses are not forward-validation observations.
    """
    if not 1 <= len(symbols) <= 10 or len(set(symbols)) != len(symbols):
        raise ValueError("provide 1..10 distinct candidate symbols")
    if output.exists():
        raise FileExistsError("evidence file already exists")
    research = settings.research
    if (research is None or research.candidate_catalog is None or research.candidate_economics is None
            or research.validation_scheduler is None):
        raise ValueError("candidate catalog/economics and bounded funding settings are required")
    catalog = collect_catalog(settings) if catalog is None else catalog
    scheduler = research.validation_scheduler
    collector = CandidateMarketCollector(
        catalog, research.candidate_catalog, research.data_quality,
        timeout_seconds=research.timeout_seconds, funding_page_limit=scheduler.funding_page_limit,
        funding_max_pages=scheduler.funding_max_pages,
    )
    repository = SQLiteOpportunityRepository(SQLiteConfig(research.sqlite_path))
    repository.initialize()
    captures: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for symbol in symbols:
        stage = "COLLECTION"
        observation: CandidateObservation | None = None
        try:
            observation = collector.collect(symbol)
            if observation.candidate.symbol != symbol or observation.assembly.symbol != symbol:
                raise ValueError("collected candidate does not match the requested symbol")
            stage = "COST_COMPARISON"
            comparisons = compare_candidate_windows(
                catalog, research.candidate_catalog, observation, research.candidate_economics,
                settings.fees.as_rates(), research.data_quality, research.decision,
                evaluated_at=utc_now(),
            )
            windows: list[dict[str, object]] = []
            for comparison in comparisons:
                stage = "PERSISTENCE"
                baseline = comparison.scenarios[0].estimate
                saved = build_and_save_opportunity(
                    repository, baseline, comparison.decision, created_at=utc_now(),
                    formula_version=research.formula_version,
                    config_version=f"{research.config_version}:{comparison.profile_name}",
                )
                windows.append({
                    "profile": comparison.profile_name,
                    "configured_holding_seconds": comparison.configured_holding_seconds,
                    "assumed_holding_duration": comparison.assumed_holding_duration,
                    "entry_time": baseline.scenario.opened_at, "assumed_exit_time": baseline.scenario.closed_at,
                    "estimate_status": baseline.status, "decision": comparison.decision.status,
                    "net_profit": baseline.net_profit, "net_return": baseline.net_return,
                    "funding_assumption": baseline.scenario.funding_events,
                    "funding_total": baseline.estimated_funding,
                    "spot_fees": None if baseline.fees is None else baseline.fees.spot_total,
                    "perp_fees": None if baseline.fees is None else baseline.fees.perp_total,
                    "persistence_status": saved.status, "opportunity_id": saved.saved_id,
                    "failure": saved.failure,
                })
            captures.append({"symbol": symbol, "candidate": observation.candidate, "windows": windows})
            summaries.append({"symbol": symbol, "windows": windows,
                              "status": "SAVED" if all(w["persistence_status"] is PersistenceStatus.SAVED
                                                        for w in windows) else "PERSISTENCE_FAILED"})
        except Exception as error:
            failure: dict[str, object] = {"symbol": symbol, "stage": stage, "error_type": type(error).__name__,
                       "message": str(error)[:300], "status": "FAILED"}
            if observation is not None:
                failure["quality_reason_codes"] = validate_market_data(
                    observation.assembly, research.data_quality, evaluated_at=utc_now()).reason_codes
            captures.append({**failure, "observation": observation})
            summaries.append(failure)
    evidence = {
        "schema_version": "candidate-cost-research-v1", "created_at": utc_now(),
        "meaning": "Public research hypotheses, not later observations, actual trades or proven profitability. No orders.",
        "formula_version": research.formula_version, "config_version": research.config_version,
        "catalog": catalog, "criteria": research.candidate_catalog.model_dump(),
        "economics": research.candidate_economics.model_dump(), "fees": settings.fees.model_dump(),
        "quality": research.data_quality, "decision_config": research.decision,
        "sqlite_path": str(repository.config.database_path), "captures": captures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((exact_json(evidence) + "\n").encode("utf-8"))
    return {"evidence": str(output.resolve()), "sqlite_path": str(repository.config.database_path),
            "captures": summaries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        output = args.output or Path("data/research/candidate-costs") / f"{uuid4().hex}.json"
        result = research_once(settings, tuple(args.symbols), output)
        print(exact_json(result))
        captures = result["captures"]
        assert isinstance(captures, list)
        return 0 if all(item["status"] == "SAVED" for item in captures) else 1
    except (OSError, ValueError, RepositoryError) as error:
        print(exact_json({"status": "RESEARCH_FAILED", "error_type": type(error).__name__,
                          "message": str(error)[:300]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
