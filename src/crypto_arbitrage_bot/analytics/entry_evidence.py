"""Decode only recorded entry evidence; never turn forecast exits into observations."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from crypto_arbitrage_bot.arbitrage.fees import FeeRates
from crypto_arbitrage_bot.models.calculation import FillSlice, FillStatus, SimulatedFill, TradeLeg
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig, OpenInterestTimeSemantics
from crypto_arbitrage_bot.models.market import BidAsk, require_market_symbol
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp


@dataclass(frozen=True)
class EntryEvidence:
    opened_at: datetime
    spot: SimulatedFill
    perp: SimulatedFill
    spot_fee: Decimal
    perp_fee: Decimal
    rates: FeeRates
    other_costs: Decimal
    quality_config: DataQualityConfig
    spot_quote: BidAsk
    perp_quote: BidAsk
    entry_basis: Decimal
    research_instrument: ResearchInstrument | None = None


def _decimal(value: Any) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("stored Decimal must be text")
    return require_decimal(Decimal(value), "stored value")


def _fill(value: dict[str, Any], leg: TradeLeg) -> SimulatedFill:
    result = SimulatedFill(
        value["symbol"], TradeLeg(value["leg"]), _decimal(value["requested_quantity"]),
        _decimal(value["filled_quantity"]), _decimal(value["quote_notional"]),
        _decimal(value["vwap"]), FillStatus(value["status"]),
        tuple(FillSlice(_decimal(part["price"]), _decimal(part["quantity"])) for part in value["fills"]),
        _decimal(value["best_price"]),
    )
    if result.leg is not leg or not result.complete:
        raise ValueError("recorded entry must be a complete fill of the expected leg")
    return result


def _duration(value: Any) -> timedelta:
    micros = value["microseconds"]
    if type(micros) is not int:
        raise ValueError("quality threshold must use integer microseconds")
    return timedelta(microseconds=micros)


def decode_entry(record: OpportunityRecord) -> EntryEvidence:
    """Missing evidence raises KeyError; malformed/inconsistent evidence ValueError.

    No pickling, dynamic object restoration, forecast exits or forecast funding.
    Stored entry fees are retained; only future exit fees use the saved rates.
    """
    details = json.loads(record.details_json)
    estimate = details["estimate"]
    snapshot = details.get("snapshot")
    identity = None
    if snapshot is not None:
        value = snapshot.get("research_instrument")
        if value is not None:
            if not isinstance(value, dict) or set(value) != {"symbol", "base_asset"}:
                raise ValueError("malformed recorded research identity")
            identity = ResearchInstrument(value["symbol"], value["base_asset"])
        for name in ("spot_book", "perp_book"):
            book = snapshot[name]
            if book.get("research_instrument") != value or book["symbol"] != record.symbol:
                raise ValueError("recorded entry book identity disagrees with opportunity")
        if snapshot["symbol"] != record.symbol:
            raise ValueError("recorded snapshot symbol disagrees with opportunity")
    require_market_symbol(record.symbol, identity)
    config = json.loads(record.config_json)["estimate"]
    if estimate.get("fees") is None or len(estimate.get("fills", ())) < 3:
        raise KeyError("entry fills/fees are missing")
    if record.matched_quantity is None or record.capital_occupied is None:
        raise KeyError("recorded quantity/capital are missing")
    require_decimal(record.matched_quantity, "matched_quantity", positive=True)
    require_decimal(record.capital_occupied, "capital_occupied", positive=True)
    if record.snapshot_status != "COMPLETE" or record.data_quality_status != "VALID":
        raise ValueError("original entry market data was not complete and valid")
    quotes = (record.spot_bid, record.spot_ask, record.perp_bid, record.perp_ask)
    if any(value is None for value in quotes):
        raise KeyError("entry quotes are missing")
    assert record.spot_bid is not None and record.spot_ask is not None
    assert record.perp_bid is not None and record.perp_ask is not None
    spot = _fill(estimate["fills"][0], TradeLeg.SPOT_ENTRY)
    perp = _fill(estimate["fills"][2], TradeLeg.PERP_ENTRY)
    spot_fee, perp_fee = _decimal(estimate["fees"]["spot_entry"]), _decimal(estimate["fees"]["perp_entry"])
    require_decimal(spot_fee, "entry spot fee", nonnegative=True)
    require_decimal(perp_fee, "entry perp fee", nonnegative=True)
    other = _decimal(config["other_costs"])
    require_decimal(other, "other_costs", nonnegative=True)
    rates = FeeRates(*(_decimal(config["fee_rates"][name]) for name in
                      ("spot_entry", "spot_exit", "perp_entry", "perp_exit")))
    quality = DataQualityConfig(_duration(config["data_quality"]["max_data_age"]),
                               _duration(config["data_quality"]["max_sampling_skew"]),
                               OpenInterestTimeSemantics(config["data_quality"].get(
                                   "open_interest_time_semantics",
                                   OpenInterestTimeSemantics.EXCHANGE_EVENT)))
    if estimate.get("calculation") is None:
        raise KeyError("original calculation is missing")
    calculation = estimate["calculation"]
    if (_decimal(calculation["net_profit"]) != record.net_profit
            or _decimal(estimate["net_return"]) != record.net_return
            or _decimal(estimate["capital_occupied"]) != record.capital_occupied):
        raise ValueError("recorded comparison metrics disagree with original estimate")
    return EntryEvidence(
        utc_timestamp(datetime.fromisoformat(estimate["scenario"]["opened_at"])),
        spot, perp, spot_fee, perp_fee, rates, other, quality,
        BidAsk(record.spot_bid, record.spot_ask), BidAsk(record.perp_bid, record.perp_ask),
        _decimal(calculation["entry_basis"]),
        identity,
    )
