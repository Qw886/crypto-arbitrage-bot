"""Metadata research selection is offline and never expands execution authority."""

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from crypto_arbitrage_bot.config.research import CandidateCatalogSettings
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.exchange_metadata import (
    MetadataError,
    MetadataErrorKind,
    MetadataSource,
)
from crypto_arbitrage_bot.exchanges.http_client import RequestPhase, prepare_get
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
from crypto_arbitrage_bot.market_data.candidate_catalog import CatalogReason as Reason
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CatalogStatus,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS, Side

NOW = datetime(2026, 10, 5, tzinfo=UTC)
MS = int((NOW - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1))


def instrument(asset="SOL", perp=False):
    row = {"symbol": asset + "USDT", "baseAsset": asset, "quoteAsset": "USDT", "status": "TRADING",
           "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.000001", "minQty": "0.000001",
                        "maxQty": "10000.000001"}, {"filterType": "PRICE_FILTER", "tickSize": "0.00000001"},
                       {"filterType": "MIN_NOTIONAL", "notional" if perp else "minNotional": "5.000000000000000001"}]}
    if perp:
        row.update(pair=row["symbol"], contractType="PERPETUAL", marginAsset="USDT",
                   onboardDate=MS - 30 * 86400000, deliveryDate=MS + 365 * 86400000,
                   underlyingType="COIN")
    else:
        row["isSpotTradingAllowed"] = True
    return row


def source(rows, perp=False, **kwargs):
    values = {"endpoint": PublicEndpoint.PERP_EXCHANGE_INFO if perp else PublicEndpoint.SPOT_EXCHANGE_INFO,
              "requested_at": NOW - timedelta(seconds=1), "received_at": NOW, "finished_at": NOW,
              "official_time": None, "elapsed_ms": 1000, "status_code": 200,
              "raw_response": json.dumps({"symbols": rows}).encode(), "body_truncated": False, "error": None}
    values.update(kwargs)
    return MetadataSource(**values)


@pytest.fixture
def config():
    return CandidateCatalogSettings(min_perp_age_seconds=86400, min_remaining_lifetime_seconds=86400,
                                    max_metadata_age_seconds=10, max_sampling_skew_seconds=2)


def build(config, spot=None, perp=None):
    return build_candidate_catalog(source(spot or [instrument()]),
                                   source(perp or [instrument(perp=True)], True), config, now=NOW)


def test_btc_eth_and_new_research_asset_use_exact_decimal_rules(config):
    assets = ["BTC", "ETH", "SOL"]
    s, p = [instrument(a) for a in assets], [instrument(a, True) for a in assets]
    before = copy.deepcopy((s, p))
    result = build(config, s, p)
    assert result.status is CatalogStatus.COMPLETE
    assert [c.symbol for c in result.candidates] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert [c.benchmark for c in result.candidates] == [True, True, False]
    assert all(c.spot_rules.min_notional == Decimal("5.000000000000000001") for c in result.candidates)
    assert all(c.perp_tick_size == Decimal(".00000001") for c in result.candidates)
    assert (s, p) == before
    assert all(s.official_time is None for s in result.sources)


@pytest.mark.parametrize("field,value,reason", [("status", "BREAK", Reason.PERP_NOT_TRADING),
    ("contractType", "CURRENT_QUARTER", Reason.NOT_USDT_PERPETUAL),
    ("marginAsset", "USDC", Reason.NOT_USDT_PERPETUAL), ("pair", "BTCUSDT", Reason.ASSET_MISMATCH),
    ("baseAsset", "BTC", Reason.ASSET_MISMATCH), ("baseAsset", None, Reason.ASSET_MISMATCH),
    ("onboardDate", None, Reason.MALFORMED_METADATA), ("onboardDate", 0, Reason.MALFORMED_METADATA),
    ("deliveryDate", None, Reason.MALFORMED_METADATA),
    ("onboardDate", MS + 1, Reason.PERP_TOO_NEW), ("deliveryDate", MS, Reason.LIFETIME_TOO_SHORT)])
def test_perp_metadata_exclusions_are_structured(config, field, value, reason):
    p = instrument(perp=True)
    p[field] = value
    result = build(config, perp=[p])
    assert not result.candidates and reason in result.exclusions[0].reason_codes


@pytest.mark.parametrize("field,value,reason", [("status", "HALT", Reason.SPOT_NOT_TRADING),
    ("isSpotTradingAllowed", False, Reason.SPOT_NOT_TRADING),
    ("isSpotTradingAllowed", "true", Reason.SPOT_NOT_TRADING),
    ("baseAsset", "OTHER", Reason.ASSET_MISMATCH), ("quoteAsset", "USDC", Reason.ASSET_MISMATCH)])
def test_spot_metadata_exclusions(config, field, value, reason):
    s = instrument()
    s[field] = value
    result = build(config, spot=[s])
    assert not result.candidates and reason in result.exclusions[0].reason_codes


def test_multiplier_contract_is_not_paired_with_unscaled_spot(config):
    result = build(config, spot=[instrument("PEPE")], perp=[instrument("1000PEPE", True)])
    assert not result.candidates
    assert set(result.exclusions[0].reason_codes) >= {Reason.SPOT_UNAVAILABLE, Reason.UNIT_MAPPING_UNVERIFIED}


def test_multiplier_is_unverified_even_if_same_symbol_is_present(config):
    result = build(config, spot=[instrument("1000PEPE")], perp=[instrument("1000PEPE", True)])
    assert result.exclusions[0].reason_codes == (Reason.UNIT_MAPPING_UNVERIFIED,)


def test_real_asset_name_with_digit_is_not_guessed_as_contract_multiplier(config):
    assert build(config, [instrument("1INCH")], [instrument("1INCH", True)]).candidates[0].base_asset == "1INCH"


def test_asset_denylist_is_configured(config):
    config = config.model_copy(update={"blocked_assets": ("SOL",)})
    assert build(config).exclusions[0].reason_codes == (Reason.ASSET_BLOCKED,)


@pytest.mark.parametrize("side", [False, True])
@pytest.mark.parametrize("change", ["step_zero", "tick_zero", "float", "negative", "missing", "duplicate", "badmax"])
def test_invalid_quantity_and_price_filters_never_become_candidates(config, side, change):
    s, p = instrument(), instrument(perp=True)
    filters = (p if side else s)["filters"]
    if change == "step_zero":
        filters[0]["stepSize"] = "0"
    elif change == "tick_zero":
        filters[1]["tickSize"] = "0"
    elif change == "float":
        filters[0]["stepSize"] = .001
    elif change == "negative":
        filters[0]["minQty"] = "-1"
    elif change == "missing":
        filters.pop()
    elif change == "duplicate":
        filters.append(copy.deepcopy(filters[0]))
    else:
        filters[0]["maxQty"] = "0.0000001"
    result = build(config, [s], [p])
    assert not result.candidates and result.exclusions[0].reason_codes == (Reason.INVALID_FILTERS,)


def test_spot_notional_filter_is_supported_without_precision_heuristics(config):
    s = instrument()
    s["filters"][-1]["filterType"] = "NOTIONAL"
    s.update(quantityPrecision=0, pricePrecision=0)
    candidate = build(config, [s]).candidates[0]
    assert candidate.spot_rules.step_size == Decimal(".000001")
    assert candidate.spot_tick_size == Decimal(".00000001")


@pytest.mark.parametrize("side", [False, True])
def test_failed_source_exposes_no_partial_candidates(config, side):
    s, p = source([instrument()]), source([instrument(perp=True)], True)
    failure = MetadataError(MetadataErrorKind.NETWORK_ERROR, RequestPhase.REQUEST)
    if side:
        p = replace(p, error=failure)
    else:
        s = replace(s, error=failure)
    result = build_candidate_catalog(s, p, config, now=NOW)
    assert result.status is CatalogStatus.INVALID and not result.candidates
    assert Reason.SOURCE_FAILED in result.reason_codes


@pytest.mark.parametrize("age,valid", [(10, True), (11, False)])
def test_metadata_age_boundary(config, age, valid):
    s = source([instrument()], requested_at=NOW - timedelta(seconds=age))
    p = source([instrument(perp=True)], True, requested_at=s.requested_at)
    result = build_candidate_catalog(s, p, config, now=NOW)
    assert (result.status is CatalogStatus.COMPLETE) is valid


def test_future_request_and_sampling_skew_are_rejected(config):
    s = source([instrument()], requested_at=NOW + timedelta(seconds=1),
               received_at=NOW + timedelta(seconds=1), finished_at=NOW + timedelta(seconds=1))
    p = source([instrument(perp=True)], True, requested_at=NOW - timedelta(seconds=5))
    result = build_candidate_catalog(s, p, config, now=NOW)
    assert set(result.reason_codes) == {Reason.SOURCE_STALE, Reason.SAMPLING_SKEW_TOO_LARGE}


@pytest.mark.parametrize("change", ["swapped", "duplicate", "malformed", "bad_json"])
def test_ambiguous_sources_never_yield_research_candidates(config, change):
    s, p = source([instrument()]), source([instrument(perp=True)], True)
    if change == "swapped":
        s, p = p, s
    elif change == "duplicate":
        s = source([instrument(), instrument()])
    elif change == "malformed":
        s = source([{"symbol": None}])
    else:
        s = replace(s, raw_response=b"{")
    result = build_candidate_catalog(s, p, config, now=NOW)
    assert result.status is CatalogStatus.INVALID and not result.candidates


def test_multiple_symbol_failures_are_kept(config):
    p = instrument(perp=True)
    p.update(status="BREAK", contractType="CURRENT_QUARTER", deliveryDate=MS)
    result = build(config, [instrument("BTC")], [p])
    assert set(result.exclusions[0].reason_codes) >= {Reason.PERP_NOT_TRADING, Reason.NOT_USDT_PERPETUAL,
                                                    Reason.SPOT_UNAVAILABLE, Reason.LIFETIME_TOO_SHORT}


def test_no_new_asset_is_added_to_public_depth_or_execution_authority(config):
    assert build(config).candidates[0].symbol == "SOLUSDT"
    assert set(SUPPORTED_SYMBOLS) == {"BTCUSDT", "ETHUSDT"}
    with pytest.raises(ValueError):
        prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "SOLUSDT"})
    with pytest.raises(ValueError):
        IocLimitOrder(ExecutionMarket.SPOT, "SOLUSDT", Side.BUY, Decimal("1"), Decimal("1"), "research-only")
    assert load_settings(Path("docs/research-soak.toml")).symbols == ("BTCUSDT", "ETHUSDT")


@pytest.mark.parametrize("field,value", [("min_perp_age_seconds", -1), ("min_perp_age_seconds", True),
    ("min_remaining_lifetime_seconds", 0), ("max_metadata_age_seconds", .1),
    ("max_sampling_skew_seconds", 61), ("blocked_assets", ["SOL", "SOL"]),
    ("blocked_assets", ["sol"]), ("blocked_assets", ["SOL&x=y"])])
def test_catalog_config_rejects_ambiguous_or_unsafe_inputs(config, field, value):
    data = config.model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        CandidateCatalogSettings(**data)


def test_empty_successful_metadata_catalog_is_explicit(config):
    result = build_candidate_catalog(source([]), source([], True), config, now=NOW)
    assert result.status is CatalogStatus.COMPLETE and result.candidates == result.exclusions == ()


@pytest.mark.parametrize("value", ["INDEX", None])
def test_non_coin_or_unverified_underlying_is_excluded(config, value):
    p = instrument(perp=True)
    p["underlyingType"] = value
    assert Reason.NOT_USDT_PERPETUAL in build(config, perp=[p]).exclusions[0].reason_codes


def test_contradictory_lifetime_is_not_a_research_candidate(config):
    p = instrument(perp=True)
    p.update(onboardDate=MS + 10 * 86400000, deliveryDate=MS + 9 * 86400000)
    assert Reason.MALFORMED_METADATA in build(config, perp=[p]).exclusions[0].reason_codes


def test_megatoken_unit_hint_is_not_guessed_as_an_unscaled_asset(config):
    result = build(config, [instrument("1MBABYDOGE")], [instrument("1MBABYDOGE", True)])
    assert not result.candidates and result.exclusions[0].reason_codes == (Reason.UNIT_MAPPING_UNVERIFIED,)
