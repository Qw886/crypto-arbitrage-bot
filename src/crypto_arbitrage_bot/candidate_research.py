"""One public metadata catalog capture using the existing TOML/env settings."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.exchange_metadata import BinanceMetadataClient
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CandidateCatalog,
    CatalogStatus,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now


def collect_catalog(settings: Settings, *, client: BinanceMetadataClient | None = None,
                    spot_instrument: ResearchInstrument | None = None) -> CandidateCatalog:
    if settings.research is None or settings.research.candidate_catalog is None:
        raise ValueError("research.candidate_catalog configuration is required")
    if spot_instrument is not None and not isinstance(spot_instrument, ResearchInstrument):
        raise TypeError("Spot metadata scope requires an explicit research instrument")
    transport = client or BinanceMetadataClient(timeout_seconds=settings.research.timeout_seconds)
    # Independent public GETs start together; a slow Spot body must not defer
    # the Perp request and manufacture sampling skew. Freshness/skew thresholds
    # and original source times remain unchanged.
    with ThreadPoolExecutor(max_workers=2) as pool:
        spot_task = (pool.submit(transport.fetch, PublicEndpoint.SPOT_EXCHANGE_INFO)
                     if spot_instrument is None else pool.submit(transport.fetch, PublicEndpoint.SPOT_EXCHANGE_INFO,
                                                                 research_instrument=spot_instrument))
        perp_task = pool.submit(transport.fetch, PublicEndpoint.PERP_EXCHANGE_INFO)
        spot, perp = spot_task.result(), perp_task.result()
    return build_candidate_catalog(spot, perp, settings.research.candidate_catalog, now=utc_now())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new evidence JSON file; existing files are never replaced")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError("catalog evidence output already exists")
        settings = load_settings(args.config)
        catalog = collect_catalog(settings)
        assert settings.research is not None
        output = args.output or settings.research.sqlite_path.parent / "catalog" / f"{uuid4().hex}.json"
        evidence = {"schema_version": "public-candidate-catalog-v1", "config_version": settings.research.config_version,
                    "criteria": settings.research.candidate_catalog.model_dump() if settings.research.candidate_catalog else None,
                    "meaning": "Metadata research candidates only; depth, costs and profitability remain unverified. No orders.",
                    "catalog": catalog}
        body = (exact_json(evidence) + "\n").encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(body)
        print(exact_json({"status": catalog.status, "candidates": len(catalog.candidates),
                          "excluded": len(catalog.exclusions), "reason_codes": catalog.reason_codes,
                          "sources": [{"endpoint": s.endpoint, "status_code": s.status_code, "error": s.error}
                                      for s in catalog.sources], "evidence": str(output.resolve())}))
        return 0 if catalog.status is CatalogStatus.COMPLETE else 1
    except (OSError, ValueError) as error:
        print(exact_json({"status": "CATALOG_FAILED", "error_type": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
