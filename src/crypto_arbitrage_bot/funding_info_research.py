"""One finite public Funding configuration observation, never a PnL forecast."""

import argparse
import hashlib
import sys
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_registry import full_research_policy
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class FundingProfileReason(StrEnum):
    SOURCE_FAILED = "SOURCE_FAILED"
    INVALID_SOURCE_IDENTITY = "INVALID_SOURCE_IDENTITY"
    INVALID_SOURCE_TIME = "INVALID_SOURCE_TIME"
    STALE_SOURCE = "STALE_SOURCE"


def read_funding_profile(settings: Settings, symbols: tuple[str, ...], output: Path, *,
                         evaluated_at: Callable[[], datetime] = utc_now) -> dict[str, Any]:
    """Observe adjusted caps/floors/intervals; preserve source and UNKNOWN fields.

    Explicit USDT research identities do not grant catalog eligibility, an entry,
    or a future Funding payment. Empty official arrays are valid observations.
    This independent evidence file never changes SQLite or ongoing study rules.
    """
    if settings.research is None:
        raise ValueError("explicit research configuration required")
    if output.exists():
        raise FileExistsError("funding profile evidence already exists")
    if (type(symbols) is not tuple or not 1 <= len(symbols) <= 100
            or any(type(s) is not str or not s.endswith("USDT") for s in symbols)
            or len(set(symbols)) != len(symbols)):
        raise ValueError("provide distinct explicit USDT research symbols")
    instruments = tuple(ResearchInstrument(s, s[:-4]) for s in symbols)
    source = BinanceUsdtmClient(settings.research.timeout_seconds).fetch_funding_info()
    at = utc_timestamp(evaluated_at())
    reasons = []
    if not source.ok:
        reasons.append(FundingProfileReason.SOURCE_FAILED)
    if (source.request.endpoint is not PublicEndpoint.PERP_FUNDING_INFO
            or source.request.params or source.request.research_instrument is not None):
        reasons.append(FundingProfileReason.INVALID_SOURCE_IDENTITY)
    if source.received_at is None or not source.started_at <= source.received_at <= source.finished_at <= at:
        reasons.append(FundingProfileReason.INVALID_SOURCE_TIME)
    elif at - source.started_at > settings.research.data_quality.max_data_age:
        reasons.append(FundingProfileReason.STALE_SOURCE)
    available = not reasons
    entries = {row.symbol: row for row in source.data or ()} if available else {}
    profiles = []
    for instrument in instruments:
        info = entries.get(instrument.symbol)
        profiles.append({"symbol": instrument.symbol,
            "adjustment_status": "SOURCE_UNAVAILABLE" if not available else "NOT_REPORTED" if info is None else "REPORTED",
            "funding_interval_hours": None if info is None else info.funding_interval_hours,
            "adjusted_funding_rate_cap": None if info is None else info.adjusted_funding_rate_cap,
            "adjusted_funding_rate_floor": None if info is None else info.adjusted_funding_rate_floor,
            "observed_at": None if not available else source.received_at,
            "official_time": None, "disclaimer": None if info is None else info.disclaimer})
    source_sha = None if source.raw_response is None else hashlib.sha256(source.raw_response).hexdigest()
    policy = full_research_policy(settings)
    report = {"schema_version": "funding-profile-observation-v1", "created_at": at,
        "scope": "PUBLIC_FUNDING_CONFIGURATION_NOT_SETTLEMENT_OR_PNL",
        "status": "COMPLETE" if available else "UNAVAILABLE", "reason_codes": tuple(reasons),
        "research_config_version": settings.research.config_version,
        "research_policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
        "research_policy": policy, "requested_symbols": symbols,
        "source": source, "source_sha256": source_sha, "profiles": profiles,
        "orders_submitted": False, "credentials_read": False,
        "meaning": "Observed official adjustments only. Absent symbols remain unknown, not assumed 8h. "
                   "No settlement, hourly extrapolation, PnL, eligibility or trade authorization. "
                   "No official event time is supplied by this endpoint."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(exact_json(report) + "\n")
    return {key: report[key] for key in ("status", "reason_codes", "profiles", "source_sha256")} | {
        "evidence": str(output.resolve()), "source_error": source.error, "orders_submitted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path, help="New evidence file; never overwrites an old observation")
    args = parser.parse_args(argv)
    try:
        result = read_funding_profile(load_settings(args.config), tuple(args.symbols), args.output)
        print(exact_json(result))
        return 0 if result["status"] == "COMPLETE" else 1
    except (OSError, ValueError) as error:
        print(exact_json({"status": "FUNDING_PROFILE_FAILED", "error_type": type(error).__name__,
                          "orders_submitted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
