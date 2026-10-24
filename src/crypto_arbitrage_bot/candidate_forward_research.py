"""One bounded batch of due candidate validations, using the original scheduler."""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from crypto_arbitrage_bot.candidate_research import collect_catalog
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.market_data.candidate_catalog import CandidateCatalog
from crypto_arbitrage_bot.market_data.candidate_market import CandidateMarketCollector
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.validation_scheduler import ValidationScheduler


class CandidateValidationSources:
    """Fresh public evidence only. No caches of prices, exits or funding events."""

    def __init__(self, settings: Settings) -> None:
        research = settings.research
        if research is None or research.candidate_catalog is None or research.validation_scheduler is None:
            raise ValueError("candidate catalog and validation scheduler configuration are required")
        self.settings = settings
        self.catalog: CandidateCatalog | None = None
        self.collector: CandidateMarketCollector | None = None
        self.catalogs: list[CandidateCatalog] = []
        self.captures: list[dict[str, object]] = []
        self.spot_instrument: ResearchInstrument | None = None

    def _select_instrument(self, symbol: str) -> None:
        identity = ResearchInstrument(symbol, symbol.removesuffix("USDT"))
        if identity != self.spot_instrument:
            # A catalog scoped to BTC cannot authorize an ETH observation.
            self.catalog, self.collector = None, None
            self.spot_instrument = identity

    def _current(self) -> CandidateMarketCollector:
        research = self.settings.research
        assert research is not None and research.candidate_catalog is not None
        scheduler = research.validation_scheduler
        assert scheduler is not None
        now = utc_now()
        # Catalog assembly time is not a new metadata observation. A slow
        # response already consumes part of the original request-age budget.
        age = None if self.catalog is None else now - min(
            self.catalog.evaluated_at, *(source.requested_at for source in self.catalog.sources))
        if (age is None or age < timedelta(0)
                or age > timedelta(seconds=research.candidate_catalog.max_metadata_age_seconds)):
            # Clear the old catalog before I/O; refresh failure cannot fall back.
            self.catalog = None
            self.collector = None
            catalog = (collect_catalog(self.settings) if self.spot_instrument is None else
                       collect_catalog(self.settings, spot_instrument=self.spot_instrument))
            # Keep failed refresh evidence even if qualification rejects it.
            # Retention is not a cache and cannot authorize a market read.
            self.catalogs.append(catalog)
            collector = CandidateMarketCollector(
                catalog, research.candidate_catalog, research.data_quality,
                timeout_seconds=research.timeout_seconds,
                funding_page_limit=scheduler.funding_page_limit,
                funding_max_pages=scheduler.funding_max_pages,
            )
            self.catalog, self.collector = catalog, collector
        assert self.collector is not None
        return self.collector

    def collect(self, symbol: str) -> SnapshotAssembly:
        self._select_instrument(symbol)
        observation = self._current().collect(symbol)
        if observation.candidate.symbol != symbol or observation.assembly.symbol != symbol:
            raise ValueError("candidate exit symbol does not match the requested opportunity")
        self.captures.append({"phase": "EXIT_MARKET", "symbol": symbol, "observation": observation})
        return observation.assembly

    def fetch_window(self, symbol: str, opened_at: datetime, closed_at: datetime) -> FundingWindow:
        self._select_instrument(symbol)
        window = self._current().fetch_funding_window(symbol, opened_at, closed_at)
        self.captures.append({"phase": "ACTUAL_FUNDING", "symbol": symbol, "window": window})
        return window


class _TaskEvents(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[object] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(json.loads(record.getMessage()))


def run_due_once(settings: Settings, output: Path) -> dict[str, object]:
    """Reconcile/safely claim one existing bounded task batch; no polling or orders.

    Original scheduler owns due times, lateness, actual settlement coverage,
    validator invocation, fixed validation IDs, leases and persistence semantics.
    Evidence retains raw subsequent books and history pages outside the compact
    validation table. Original Opportunities/Decisions are never changed.
    """
    if output.exists():
        raise FileExistsError("validation evidence file already exists")
    sources = CandidateValidationSources(settings)
    research = settings.research
    assert research is not None and research.validation_scheduler is not None
    if not research.sqlite_path.is_file():
        raise ValueError("existing candidate Opportunity database is required")
    config = SQLiteConfig(research.sqlite_path)
    events = _TaskEvents()
    logger = logging.Logger("candidate-forward-research", level=logging.INFO)
    logger.addHandler(events)
    scheduler = ValidationScheduler(
        research.validation_scheduler, SQLiteOpportunityRepository(config),
        SQLiteForwardValidationRepository(config), SQLiteValidationTaskRepository(config),
        sources, sources, clock=lambda: utc_now(), logger=logger,
    )
    started_at = utc_now()
    scheduler.process_due()
    report = {
        "schema_version": "candidate-forward-research-v1",
        "meaning": "Observed later public books and actual settlements; simulated outcomes, not actual trades or realized profit.",
        "started_at": started_at, "ended_at": utc_now(), "sqlite_path": str(config.database_path),
        "scheduler_config": research.validation_scheduler.model_dump(),
        "catalogs": sources.catalogs, "captures": sources.captures, "task_events": events.events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write((exact_json(report) + "\n").encode("utf-8"))
    return {"evidence": str(output.resolve()), "sqlite_path": str(config.database_path),
            "public_captures": len(sources.captures), "task_events": events.events}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output or Path("data/research/candidate-forward") / f"{uuid4().hex}.json"
        result = run_due_once(load_settings(args.config), output)
        print(exact_json(result))
        events = result["task_events"]
        assert isinstance(events, list)
        return 1 if any(isinstance(item, dict) and (
            item.get("status") == "FAILED" or item.get("failure") is not None) for item in events) else 0
    except (OSError, ValueError, RepositoryError) as error:
        print(exact_json({"status": "FORWARD_RESEARCH_FAILED", "error_type": type(error).__name__,
                          "message": str(error)[:300]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
