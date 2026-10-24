"""Strict official fundingInfo array parsing; absent symbols have no guessed defaults."""

import json
import re
from datetime import datetime
from decimal import Decimal

from crypto_arbitrage_bot.exchanges.binance_spot import _reject_nonfinite_json, _unique_json_object
from crypto_arbitrage_bot.models.funding_info import FundingAdjustment


def decode_funding_info(body: bytes) -> dict[str, object]:
    value = json.loads(body.decode("utf-8"), parse_float=Decimal,
                       parse_constant=_reject_nonfinite_json, object_pairs_hook=_unique_json_object)
    if isinstance(value, dict) and type(value.get("code")) is int and value["code"] < 0:
        return value
    # Empty array is a legitimate adjustment catalog, unlike an empty body.
    if not isinstance(value, list) or len(value) > 5000:
        raise ValueError("funding adjustments require a bounded array")
    return {"entries": value}


def parse_funding_info(payload: dict[str, object], observed_at: datetime) -> tuple[FundingAdjustment, ...]:
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise ValueError("funding adjustment entries missing")
    entries = []
    symbols = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("funding adjustment must be an object")
        symbol, interval = row.get("symbol"), row.get("fundingIntervalHours")
        if type(symbol) is not str or symbol in symbols or type(interval) is not int:
            raise ValueError("funding symbol or interval missing/invalid/duplicate")
        rates = []
        for name in ("adjustedFundingRateCap", "adjustedFundingRateFloor"):
            value = row.get(name)
            if type(value) is not str or re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value) is None:
                raise ValueError("funding cap/floor must be exact Decimal strings")
            rates.append(Decimal(value))
        disclaimer = row.get("disclaimer")
        if disclaimer is not None and type(disclaimer) is not bool:
            raise ValueError("invalid funding disclaimer")
        entries.append(FundingAdjustment(symbol, rates[0], rates[1], interval, observed_at, disclaimer))
        symbols.add(symbol)
    return tuple(entries)
