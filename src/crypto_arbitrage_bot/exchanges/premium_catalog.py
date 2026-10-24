"""Strict bulk payload shape only; asset-specific rates/prices parse after qualification."""

import json
from datetime import datetime
from decimal import Decimal

from crypto_arbitrage_bot.exchanges.binance_spot import _reject_nonfinite_json, _unique_json_object


def decode_catalog(body: bytes) -> dict[str, object]:
    value = json.loads(body.decode("utf-8"), parse_float=Decimal,
                       parse_constant=_reject_nonfinite_json, object_pairs_hook=_unique_json_object)
    if isinstance(value, dict) and type(value.get("code")) is int and value["code"] < 0:
        return value
    if not isinstance(value, list) or not 1 <= len(value) <= 5000:
        raise ValueError("premium catalog requires a bounded nonempty array")
    return {"entries": value}


def parse_catalog(payload: dict[str, object], observed_at: datetime) -> tuple[dict[str, object], ...]:
    del observed_at  # UTC request/receive times stay on the surrounding fetch result.
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise ValueError("premium catalog entries missing")
    symbols: set[str] = set()
    entries = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("symbol")) is not str or not row["symbol"]:
            raise ValueError("premium catalog symbol missing")
        symbol = row["symbol"]
        if symbol in symbols:
            raise ValueError("duplicate premium catalog symbol")
        symbols.add(symbol)
        entries.append(row)
    return tuple(entries)
