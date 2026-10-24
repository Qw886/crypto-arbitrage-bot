"""Decode retained public bytes with original parsers, without sending requests."""

import base64
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.config.research import (
    CandidateCatalogSettings,
    ValidationSchedulerSettings,
)
from crypto_arbitrage_bot.exchanges.binance_spot import MAX_BODY_BYTES, _decode_body, _levels
from crypto_arbitrage_bot.exchanges.binance_usdtm import (
    _decimal,
    _milliseconds,
    _parse_depth,
    _parse_open_interest,
    _parse_premium,
    _reject_nonfinite_json,
    _unique_json_object,
)
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.exchange_metadata import MAX_METADATA_BYTES, MetadataSource
from crypto_arbitrage_bot.exchanges.funding_history import _epoch_ms
from crypto_arbitrage_bot.exchanges.http_client import PublicGetRequest
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CatalogStatus,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.market import Market, OrderBook
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly, SnapshotSources
from crypto_arbitrage_bot.models.spot import SpotDepthResult
from crypto_arbitrage_bot.models.usdtm import UsdtmFetchResult
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


def time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("PUBLIC_EVIDENCE_TIME_REQUIRED")
    return utc_timestamp(datetime.fromisoformat(value))


def raw_bytes(value: Any, maximum: int = MAX_BODY_BYTES) -> bytes:
    if (
        not isinstance(value, dict)
        or set(value) != {"base64"}
        or not isinstance(value["base64"], str)
        or len(value["base64"]) > 4 * ((maximum + 2) // 3)
    ):
        raise ValueError("BOUNDED_PUBLIC_BODY_REQUIRED")
    raw = base64.b64decode(value["base64"], validate=True)
    if len(raw) > maximum:
        raise ValueError("PUBLIC_BODY_LIMIT")
    return raw


def _metadata(value: dict[str, Any]) -> MetadataSource:
    if (
        value["error"] is not None
        or value["body_truncated"] is not False
        or value["status_code"] != 200
    ):
        raise ValueError("COMPLETE_METADATA_REQUIRED")
    identity = value["research_instrument"]
    if identity is not None:
        identity = ResearchInstrument(**identity)
    body = raw_bytes(value["raw_response"], MAX_METADATA_BYTES)
    parsed = _decode_body(body)
    official = None if value["official_time"] is None else time(value["official_time"])
    if official != (
        None if "serverTime" not in parsed else _milliseconds(parsed["serverTime"], "serverTime")
    ):
        raise ValueError("METADATA_OFFICIAL_TIME_DIFFERS")
    return MetadataSource(
        PublicEndpoint(value["endpoint"]),
        time(value["requested_at"]),
        time(value["received_at"]),
        time(value["finished_at"]),
        official,
        value["elapsed_ms"],
        200,
        body,
        False,
        None,
        identity,
    )


def check_catalogs(values: Any, criteria: CandidateCatalogSettings, frozen_candidate: Any) -> None:
    if not isinstance(values, list) or len(values) > 2:
        raise ValueError("BOUNDED_METADATA_CAPTURES_REQUIRED")
    for value in values:
        if len(value["sources"]) != 2:
            raise ValueError("TWO_METADATA_SOURCES_REQUIRED")
        catalog = build_candidate_catalog(
            _metadata(value["sources"][0]),
            _metadata(value["sources"][1]),
            criteria,
            now=time(value["evaluated_at"]),
        )
        if catalog.status is not CatalogStatus.COMPLETE or exact_json(catalog) != exact_json(value):
            raise ValueError("RAW_METADATA_PROJECTION_DIFFERS")
        selected = [
            item for item in catalog.candidates if item.symbol == frozen_candidate["symbol"]
        ]
        if len(selected) != 1 or exact_json(selected[0]) != exact_json(frozen_candidate):
            raise ValueError("FROZEN_CANDIDATE_DIFFERS")


def _request(
    value: dict[str, Any], endpoint: PublicEndpoint, identity: ResearchInstrument
) -> PublicGetRequest:
    if PublicEndpoint(value["endpoint"]) is not endpoint or value[
        "research_instrument"
    ] != json.loads(exact_json(identity)):
        raise ValueError("PUBLIC_REQUEST_IDENTITY_DIFFERS")
    params = value["params"]
    pairs = (
        tuple(params.items()) if isinstance(params, dict) else tuple(tuple(pair) for pair in params)
    )
    return PublicGetRequest(endpoint, pairs, research_instrument=identity)


def _source(
    value: dict[str, Any], endpoint: PublicEndpoint, identity: ResearchInstrument
) -> SpotDepthResult | UsdtmFetchResult[Any]:
    request = _request(value["request"], endpoint, identity)
    if (
        value["error"] is not None
        or value["cleanup_error"] is not None
        or value["body_truncated"] is not False
        or type(value["status_code"]) is not int
        or value["status_code"] != 200
    ):
        raise ValueError("SUCCESSFUL_PUBLIC_SOURCE_REQUIRED")
    started, received, finished = (
        time(value[key]) for key in ("started_at", "received_at", "finished_at")
    )
    if (
        not started <= received <= finished
        or type(value["elapsed_ms"]) is not int
        or value["elapsed_ms"] < 0
    ):
        raise ValueError("PUBLIC_SOURCE_TIMING_DIFFERS")
    body = raw_bytes(value["raw_response"])
    headers = tuple(tuple(pair) for pair in value["response_headers"])
    if len(headers) > 100 or any(
        len(pair) != 2 or any(not isinstance(part, str) for part in pair) for pair in headers
    ):
        raise ValueError("BOUNDED_PUBLIC_HEADERS_REQUIRED")
    for name, length in headers:
        if name.lower() == "content-length" and (
            not length.isascii() or not length.isdigit() or int(length) != len(body)
        ):
            raise ValueError("PUBLIC_BODY_LENGTH_DIFFERS")
    payload = _decode_body(body)
    params = dict(request.params)
    limit = params.get("limit", 100)
    if type(limit) is not int:
        raise ValueError("PUBLIC_DEPTH_LIMIT_REQUIRED")
    if endpoint is PublicEndpoint.SPOT_DEPTH:
        if "symbol" in payload and payload["symbol"] != identity.symbol:
            raise ValueError("SPOT_BODY_SYMBOL_DIFFERS")
        update = payload["lastUpdateId"]
        if type(update) is not int or update < 0:
            raise ValueError("SPOT_UPDATE_ID_REQUIRED")
        bids, asks = _levels(payload["bids"], "bids"), _levels(payload["asks"], "asks")
        if len(bids) > limit or len(asks) > limit or bids[0].price > asks[0].price:
            raise ValueError("SPOT_DEPTH_DIFFERS")
        book = OrderBook(
            identity.symbol, Market.SPOT, bids, asks, received, research_instrument=identity
        )
        if exact_json(book) != exact_json(value["order_book"]) or update != value["last_update_id"]:
            raise ValueError("RAW_SPOT_PROJECTION_DIFFERS")
        return SpotDepthResult(
            request,
            started,
            finished,
            received,
            value["elapsed_ms"],
            200,
            headers,
            body,
            False,
            book,
            update,
            None,
        )
    data: Any
    if endpoint is PublicEndpoint.PERP_DEPTH:
        data = _parse_depth(payload, identity.symbol, limit, received, research_instrument=identity)
    elif endpoint is PublicEndpoint.PERP_PREMIUM_INDEX:
        data = _parse_premium(payload, identity.symbol, received, research_instrument=identity)
    else:
        data = _parse_open_interest(
            payload, identity.symbol, received, research_instrument=identity
        )
    if exact_json(data) != exact_json(value["data"]):
        raise ValueError("RAW_PERP_PROJECTION_DIFFERS")
    return UsdtmFetchResult(
        request,
        started,
        finished,
        received,
        value["elapsed_ms"],
        200,
        headers,
        body,
        False,
        data,
        None,
    )


def decode_market(observation: dict[str, Any], identity: ResearchInstrument) -> SnapshotAssembly:
    value = observation["assembly"]
    sources = value["sources"]
    if set(sources) != {"spot_depth", "perp_depth", "premium_index", "open_interest"}:
        raise ValueError("FOUR_PUBLIC_SOURCES_REQUIRED")
    spot = _source(sources["spot_depth"], PublicEndpoint.SPOT_DEPTH, identity)
    assert isinstance(spot, SpotDepthResult)
    perp = _source(sources["perp_depth"], PublicEndpoint.PERP_DEPTH, identity)
    premium = _source(sources["premium_index"], PublicEndpoint.PERP_PREMIUM_INDEX, identity)
    interest = _source(sources["open_interest"], PublicEndpoint.PERP_OPEN_INTEREST, identity)
    assert (
        isinstance(perp, UsdtmFetchResult)
        and isinstance(premium, UsdtmFetchResult)
        and isinstance(interest, UsdtmFetchResult)
    )
    assembly = assemble_snapshot(
        identity.symbol,
        SnapshotSources(spot, perp, premium, interest),
        assembled_at=time(value["assembled_at"]),
        research_instrument=identity,
    )
    if not assembly.complete or exact_json(assembly) != exact_json(value):
        raise ValueError("RAW_ASSEMBLY_PROJECTION_DIFFERS")
    return assembly


def decode_funding(
    window: dict[str, Any],
    *,
    symbol: str,
    opened_at: datetime,
    closed_at: datetime,
    criteria: ValidationSchedulerSettings,
) -> tuple[FundingEvent, ...]:
    if (
        window["symbol"] != symbol
        or window["failure"] is not None
        or window["message"] is not None
        or time(window["opened_at"]) != opened_at
        or time(window["closed_at"]) != closed_at
    ):
        raise ValueError("FUNDING_WINDOW_DIFFERS")
    pages = window["pages"]
    if not isinstance(pages, list) or not 1 <= len(pages) <= criteria.funding_max_pages:
        raise ValueError("COMPLETE_FUNDING_PAGES_REQUIRED")
    cursor, end = _epoch_ms(opened_at) + 1, _epoch_ms(closed_at)
    events = []
    previous_receive = closed_at
    complete = False
    for page in pages:
        if complete or type(page["status_code"]) is not int or page["status_code"] != 200:
            raise ValueError("FUNDING_PAGE_SEQUENCE_DIFFERS")
        requested, received = time(page["requested_at"]), time(page["received_at"])
        if not previous_receive <= requested <= received:
            raise ValueError("FUNDING_PAGE_TIMING_DIFFERS")
        previous_receive = received
        payload = json.loads(
            raw_bytes(page["raw_response"]),
            parse_float=Decimal,
            parse_constant=_reject_nonfinite_json,
            object_pairs_hook=_unique_json_object,
        )
        if not isinstance(payload, list) or len(payload) > criteria.funding_page_limit:
            raise ValueError("BOUNDED_FUNDING_PAGE_REQUIRED")
        previous = cursor - 1
        for row in payload:
            settled = _milliseconds(row["fundingTime"], "fundingTime")
            millis = _epoch_ms(settled)
            if row["symbol"] != symbol or not previous < millis <= end:
                raise ValueError("FUNDING_EVENT_SEQUENCE_DIFFERS")
            events.append(
                FundingEvent(
                    f"{symbol}:{millis}",
                    settled,
                    _decimal(row["markPrice"], "markPrice"),
                    _decimal(row["fundingRate"], "fundingRate"),
                    symbol,
                )
            )
            previous = millis
        complete = len(payload) < criteria.funding_page_limit or previous == end
        cursor = previous + 1
    if not complete or exact_json(tuple(events)) != exact_json(window["events"]):
        raise ValueError("RAW_FUNDING_COVERAGE_OR_PROJECTION_DIFFERS")
    return tuple(events)
