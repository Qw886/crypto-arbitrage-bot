"""Finite qualified Funding preselection followed by the original cost engine."""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from crypto_arbitrage_bot.candidate_cost_research import research_once
from crypto_arbitrage_bot.candidate_research import collect_catalog
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.market_data.funding_screen import ScreenStatus, screen_candidates
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.utils.clock import utc_now


def research_screen_once(settings: Settings, output: Path) -> dict[str, object]:
    if output.exists() or output.with_suffix(".costs.json").exists():
        raise FileExistsError("screen/cost evidence already exists")
    research = settings.research
    if (research is None or research.funding_screen is None or research.candidate_catalog is None
            or research.candidate_economics is None or research.validation_scheduler is None):
        raise ValueError("funding screen, catalog, economics and scheduler configuration are required")
    catalog = collect_catalog(settings)
    source = BinanceUsdtmClient(research.timeout_seconds).fetch_premium_catalog()
    screen = screen_candidates(catalog, research.candidate_catalog, source,
                               research.funding_screen, research.data_quality, evaluated_at=utc_now())
    costs = None
    details: list[dict[str, object]] = []
    collection_failures: list[dict[str, object]] = []
    if screen.status is ScreenStatus.COMPLETE:
        symbols = tuple(item.candidate.symbol for item in screen.selected)
        costs = research_once(settings, symbols, output.with_suffix(".costs.json"), catalog=catalog)
        repository = SQLiteOpportunityRepository(SQLiteConfig(research.sqlite_path))
        captures = costs["captures"]
        assert isinstance(captures, list)
        collection_failures = [item for item in captures if item["status"] != "SAVED"]
        for capture in captures:
            for window in capture.get("windows", ()):
                record_id = window.get("opportunity_id")
                if record_id is None:
                    continue
                record = repository.get_opportunity_by_id(record_id)
                if record is None:
                    raise ValueError("saved research record is missing during readback")
                details.append({"opportunity_id": record.id, "symbol": record.symbol,
                    "decision": record.decision, "reason_codes": record.decision_reason_codes,
                    "scenario_outcomes": tuple({"type": s.scenario_type, "net_profit": s.net_profit,
                        "net_return": s.net_return, "funding_total": s.funding_total} for s in record.scenarios)})
    research_status = screen.status
    if costs is not None:
        captures = costs["captures"]
        assert isinstance(captures, list)
        if not all(item["status"] == "SAVED" for item in captures):
            research_status = ScreenStatus.PARTIAL
    report = {"schema_version": "funding-screen-research-v1", "created_at": utc_now(),
              "meaning": "Observed funding preselection, then hypothetical research estimates. No orders or guaranteed payments.",
              "research_status": research_status, "catalog": catalog, "premium_source": source, "screen": screen, "costs": costs,
              "persisted_research_outcomes": details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((exact_json(report) + "\n").encode("utf-8"))
    return {"screen_status": screen.status, "research_status": research_status, "qualified_count": screen.qualified_count,
            "selected": tuple({"symbol": item.candidate.symbol, "observed_funding_rate": item.premium.last_funding_rate,
                              "next_funding_time": item.premium.next_funding_time} for item in screen.selected),
            "reason_codes": screen.reason_codes, "cost_evidence": None if costs is None else costs["evidence"],
            "collection_failures": collection_failures,
            "outcomes": details, "evidence": str(output.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output or Path("data/research/funding-screen") / f"{uuid4().hex}.json"
        result = research_screen_once(load_settings(args.config), output)
        print(exact_json(result))
        return 0 if result["research_status"] is ScreenStatus.COMPLETE and result["outcomes"] else 1
    except (OSError, ValueError, RepositoryError) as error:
        print(exact_json({"status": "FUNDING_SCREEN_FAILED", "error_type": type(error).__name__,
                          "message": str(error)[:300]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
