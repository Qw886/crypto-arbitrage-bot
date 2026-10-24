"""Public metadata candidates only: no price forecasts, PnL, decisions or orders."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.arbitrage.sizing import QuantityRules
from crypto_arbitrage_bot.config.research import CandidateCatalogSettings
from crypto_arbitrage_bot.exchanges.binance_spot import _decode_body
from crypto_arbitrage_bot.exchanges.binance_usdtm import _decimal, _milliseconds
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.exchange_metadata import MetadataSource
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


class CatalogStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INVALID = "INVALID"


class CatalogReason(StrEnum):
    SOURCE_FAILED = "SOURCE_FAILED"
    SOURCE_IDENTITY_INVALID = "SOURCE_IDENTITY_INVALID"
    SOURCE_STALE = "SOURCE_STALE"
    SAMPLING_SKEW_TOO_LARGE = "SAMPLING_SKEW_TOO_LARGE"
    MALFORMED_METADATA = "MALFORMED_METADATA"
    DUPLICATE_SYMBOL = "DUPLICATE_SYMBOL"
    SPOT_UNAVAILABLE = "SPOT_UNAVAILABLE"
    SPOT_NOT_TRADING = "SPOT_NOT_TRADING"
    PERP_NOT_TRADING = "PERP_NOT_TRADING"
    NOT_USDT_PERPETUAL = "NOT_USDT_PERPETUAL"
    ASSET_MISMATCH = "ASSET_MISMATCH"
    UNIT_MAPPING_UNVERIFIED = "UNIT_MAPPING_UNVERIFIED"
    ASSET_BLOCKED = "ASSET_BLOCKED"
    PERP_TOO_NEW = "PERP_TOO_NEW"
    LIFETIME_TOO_SHORT = "LIFETIME_TOO_SHORT"
    INVALID_FILTERS = "INVALID_FILTERS"


@dataclass(frozen=True)
class ResearchCandidate:
    symbol: str
    base_asset: str
    spot_rules: QuantityRules
    perp_rules: QuantityRules
    spot_tick_size: Decimal
    perp_tick_size: Decimal
    perp_onboard_at: datetime
    perp_delivery_at: datetime
    benchmark: bool


@dataclass(frozen=True)
class CatalogExclusion:
    symbol: str
    reason_codes: tuple[CatalogReason, ...]


@dataclass(frozen=True)
class CandidateCatalog:
    evaluated_at: datetime
    status: CatalogStatus
    candidates: tuple[ResearchCandidate, ...]
    exclusions: tuple[CatalogExclusion, ...]
    reason_codes: tuple[CatalogReason, ...]
    sources: tuple[MetadataSource, MetadataSource]


def _index(source: MetadataSource) -> dict[str, dict[str, object]]:
    payload = _decode_body(source.raw_response or b"")
    values = payload.get("symbols")
    if not isinstance(values, list):
        raise ValueError(CatalogReason.MALFORMED_METADATA)
    indexed: dict[str, dict[str, object]] = {}
    for value in values:
        if (not isinstance(value, dict) or type(value.get("symbol")) is not str
                or not value["symbol"]):
            raise ValueError(CatalogReason.MALFORMED_METADATA)
        symbol = value["symbol"]
        assert isinstance(symbol, str)
        if symbol in indexed:
            raise ValueError(CatalogReason.DUPLICATE_SYMBOL)
        indexed[symbol] = value
    return indexed


def _filters(row: dict[str, object], *, perp: bool) -> tuple[QuantityRules, Decimal]:
    values = row.get("filters")
    if not isinstance(values, list):
        raise ValueError("missing filters")
    filters: dict[str, dict[str, object]] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("filterType"), str):
            raise ValueError("invalid filter")
        kind = value["filterType"]
        assert isinstance(kind, str)
        if kind in filters:
            raise ValueError("duplicate filter")
        filters[kind] = value
    lot = filters["LOT_SIZE"]
    notional = filters["MIN_NOTIONAL"] if perp else filters.get("NOTIONAL", filters.get("MIN_NOTIONAL"))
    if notional is None:
        raise ValueError("missing notional filter")
    rules = QuantityRules(_decimal(lot.get("stepSize"), "stepSize"),
                          _decimal(lot.get("minQty"), "minQty"),
                          _decimal(notional.get("notional" if perp else "minNotional"), "notional"),
                          _decimal(lot.get("maxQty"), "maxQty"))
    tick = require_decimal(_decimal(filters["PRICE_FILTER"].get("tickSize"), "tickSize"),
                           "tickSize", positive=True)
    return rules, tick


def build_candidate_catalog(spot: MetadataSource, perp: MetadataSource,
                            config: CandidateCatalogSettings, *, now: datetime) -> CandidateCatalog:
    """Metadata availability is not liquidity, profitability or authority to trade."""
    at = utc_timestamp(now)
    reasons: list[CatalogReason] = []
    if spot.endpoint is not PublicEndpoint.SPOT_EXCHANGE_INFO or perp.endpoint is not PublicEndpoint.PERP_EXCHANGE_INFO:
        reasons.append(CatalogReason.SOURCE_IDENTITY_INVALID)
    for source in (spot, perp):
        if (source.error is not None or source.status_code != 200 or source.raw_response is None
                or source.body_truncated or source.received_at is None):
            reasons.append(CatalogReason.SOURCE_FAILED)
        if (source.finished_at > at or
                not timedelta(0) <= at - source.requested_at <= timedelta(seconds=config.max_metadata_age_seconds)):
            reasons.append(CatalogReason.SOURCE_STALE)
    if abs(spot.requested_at - perp.requested_at) > timedelta(seconds=config.max_sampling_skew_seconds):
        reasons.append(CatalogReason.SAMPLING_SKEW_TOO_LARGE)
    def result(candidates: tuple[ResearchCandidate, ...] = (),
               excluded: tuple[CatalogExclusion, ...] = ()) -> CandidateCatalog:
        return CandidateCatalog(at, CatalogStatus.INVALID if reasons else CatalogStatus.COMPLETE,
                                candidates, excluded, tuple(dict.fromkeys(reasons)), (spot, perp))
    if reasons:
        return result()
    try:
        spots, perps = _index(spot), _index(perp)
        if spot.research_instrument is not None and set(spots) != {spot.research_instrument.symbol}:
            raise ValueError("scoped metadata symbols disagree with source identity")
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        reason = str(exc)
        reasons.append(CatalogReason(reason) if reason in CatalogReason else CatalogReason.MALFORMED_METADATA)
        return result()
    candidates, exclusions = [], []
    for symbol, p in sorted(perps.items()):
        if p.get("quoteAsset") != "USDT":
            continue  # The catalog covers USDT instruments only.
        issues: list[CatalogReason] = []
        s = spots.get(symbol)
        base = p.get("baseAsset")
        if p.get("status") != "TRADING":
            issues.append(CatalogReason.PERP_NOT_TRADING)
        if (p.get("contractType") != "PERPETUAL" or p.get("marginAsset") != "USDT"
                or p.get("underlyingType") != "COIN"):
            issues.append(CatalogReason.NOT_USDT_PERPETUAL)
        if s is None:
            issues.append(CatalogReason.SPOT_UNAVAILABLE)
        elif s.get("status") != "TRADING" or s.get("isSpotTradingAllowed") is not True:
            issues.append(CatalogReason.SPOT_NOT_TRADING)
        if (not isinstance(base, str) or not re.fullmatch(r"[A-Z0-9]{1,32}", base)
                or p.get("pair") != symbol or symbol != str(base) + "USDT"
                or s is not None and (s.get("baseAsset") != base or s.get("quoteAsset") != "USDT")):
            issues.append(CatalogReason.ASSET_MISMATCH)
        if isinstance(base, str):
            if re.match(r"^(?:(?:10|100|1000|10000|100000|1000000)[A-Z]|[1-9][0-9]*M[A-Z])", base):
                issues.append(CatalogReason.UNIT_MAPPING_UNVERIFIED)
            if base in config.blocked_assets:
                issues.append(CatalogReason.ASSET_BLOCKED)
        try:
            onboard = _milliseconds(p.get("onboardDate"), "onboardDate")
            delivery = _milliseconds(p.get("deliveryDate"), "deliveryDate")
            if p.get("onboardDate") == 0 or onboard >= delivery:
                issues.append(CatalogReason.MALFORMED_METADATA)
            if at - onboard < timedelta(seconds=config.min_perp_age_seconds):
                issues.append(CatalogReason.PERP_TOO_NEW)
            if delivery - at < timedelta(seconds=config.min_remaining_lifetime_seconds):
                issues.append(CatalogReason.LIFETIME_TOO_SHORT)
        except (ValueError, TypeError, OverflowError):
            issues.append(CatalogReason.MALFORMED_METADATA)
        if issues:
            exclusions.append(CatalogExclusion(symbol, tuple(dict.fromkeys(issues))))
            continue
        assert s is not None and isinstance(base, str)
        try:
            sr, st = _filters(s, perp=False)
            pr, pt = _filters(p, perp=True)
        except (ValueError, TypeError, KeyError, ArithmeticError):
            exclusions.append(CatalogExclusion(symbol, (CatalogReason.INVALID_FILTERS,)))
            continue
        candidates.append(ResearchCandidate(symbol, base, sr, pr, st, pt, onboard, delivery,
                                            symbol in SUPPORTED_SYMBOLS))
    return result(tuple(candidates), tuple(exclusions))
