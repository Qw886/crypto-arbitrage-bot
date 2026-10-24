"""One-shot public candidate books and observed funding history, never trading."""

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from crypto_arbitrage_bot.candidate_research import collect_catalog
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.market_data.candidate_market import CandidateMarketCollector
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", required=True, help="1..10 catalog-qualified USDT symbols")
    parser.add_argument("--history-seconds", type=int, required=True, help="observed lookback; 1..30 days")
    parser.add_argument("--output", type=Path, help="new evidence file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if (not 1 <= len(args.symbols) <= 10 or len(set(args.symbols)) != len(args.symbols)
                or not 1 <= args.history_seconds <= 30 * 86400):
            raise ValueError("symbols/history must be distinct and bounded")
        if args.output is not None and args.output.exists():
            raise FileExistsError("evidence output already exists")
        settings = load_settings(args.config)
        research = settings.research
        if research is None or research.candidate_catalog is None or research.validation_scheduler is None:
            raise ValueError("candidate catalog and funding pagination configuration are required")
        catalog = collect_catalog(settings)
        scheduler = research.validation_scheduler
        collector = CandidateMarketCollector(
            catalog, research.candidate_catalog, research.data_quality,
            timeout_seconds=research.timeout_seconds,
            funding_page_limit=scheduler.funding_page_limit, funding_max_pages=scheduler.funding_max_pages,
        )
        closed = utc_now()
        opened = closed - timedelta(seconds=args.history_seconds)
        captures: list[dict[str, object]] = []
        summaries: list[dict[str, object]] = []
        for symbol in args.symbols:
            try:
                observation = collector.collect(symbol)
                funding = collector.fetch_funding_window(symbol, opened, closed)
                captures.append({"symbol": symbol, "observation": observation, "funding_history": funding})
                summaries.append({"symbol": symbol, "complete": observation.assembly.complete,
                                  "data_quality": observation.data_quality.status,
                                  "quality_reasons": observation.data_quality.reason_codes,
                                  "funding_failure": funding.failure, "settlement_count": len(funding.events)})
            except (OSError, ValueError) as error:
                failure = {"symbol": symbol, "error_type": type(error).__name__, "message": str(error)[:300]}
                captures.append(failure)
                summaries.append(failure)
        output = args.output or research.sqlite_path.parent / "candidate-markets" / f"{uuid4().hex}.json"
        evidence = {"schema_version": "public-candidate-market-v1", "created_at": utc_now(),
                    "config_version": research.config_version,
                    "meaning": "Research-only observed quotes/history. Historical funding is not future income. No orders.",
                    "history_opened_at": opened, "history_closed_at": closed,
                    "criteria": research.candidate_catalog.model_dump(), "catalog": catalog,
                    "quality_config": research.data_quality, "captures": captures}
        body = (exact_json(evidence) + "\n").encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(body)
        print(exact_json({"captures": summaries, "evidence": str(output.resolve())}))
        return 0 if all(item.get("complete") is True and item.get("data_quality") == "VALID"
                        and item.get("funding_failure") is None for item in summaries) else 1
    except (OSError, ValueError) as error:
        print(exact_json({"status": "CAPTURE_FAILED", "error_type": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
