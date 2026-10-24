"""Deterministic offline transport demonstration; no sockets or real balances."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.runner import TestnetBot
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetCredentials,
    TestnetExecutionClient,
    TestnetExecutionPolicy,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now


class DemoResponse:
    status = 200

    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode()

    def read(self, limit: int) -> bytes:
        return self.body[:limit]

    def __enter__(self) -> "DemoResponse":
        return self

    def __exit__(self, *_: object) -> None:
        pass


class DemoTransport:
    """Fake exchange implementing only the testnet client's fixed operations."""

    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        self.clock = clock
        self.base: dict[str, Decimal] = {"BTCUSDT": Decimal("0"), "ETHUSDT": Decimal("0")}
        self.positions = dict(self.base)
        self.orders: dict[str, dict[str, object]] = {}
        self.requests: list[tuple[str, str]] = []

    def open(self, fullurl, data=None, timeout=5):  # type: ignore[no-untyped-def]
        url = urlsplit(fullurl.full_url)
        self.requests.append((fullurl.get_method(), fullurl.full_url))
        if url.hostname not in ("testnet.binance.vision", "demo-fapi.binance.com"):
            raise ValueError("offline demo rejected non-testnet host")
        params = parse_qs((fullurl.data or b"").decode() if fullurl.get_method() == "POST" else url.query)
        symbol = params.get("symbol", ["BTCUSDT"])[0]
        spot = url.hostname == "testnet.binance.vision"
        from datetime import timedelta
        ms = (self.clock() - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)
        price = Decimal("50000") if symbol == "BTCUSDT" else Decimal("2000")
        if url.path.endswith("/time"):
            result: object = {"serverTime": ms}
        elif url.path.endswith("/exchangeInfo"):
            result = {"symbols": [{"symbol": s, "status": "TRADING", "filters": [
                {"filterType": "PRICE_FILTER", "minPrice": "0", "maxPrice": "1000000", "tickSize": "0.1"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "100", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
            ]} for s in self.base]}
        elif url.path.endswith("/depth"):
            ask = price if spot else price + 10
            result = {"lastUpdateId": 1, "bids": [[str(ask - 1), "10"]], "asks": [[str(ask), "10"]]}
            if not spot:
                result.update({"E": ms, "T": ms})
        elif url.path.endswith("/premiumIndex"):
            result = {"symbol": symbol, "markPrice": str(price + 10), "indexPrice": str(price),
                      "lastFundingRate": "0.0001", "nextFundingTime": ms + 28800000, "time": ms}
        elif url.path.endswith("/openInterest"):
            result = {"symbol": symbol, "openInterest": "100", "time": ms}
        elif url.path.endswith("/account"):
            result = ({"canTrade": True, "balances": [{"asset": "USDT", "free": "10000"},
                       *({"asset": s[:-4], "free": str(q)} for s, q in self.base.items())]} if spot else
                      {"canTrade": True, "availableBalance": "10000", "multiAssetsMargin": False})
        elif url.path.endswith("/openOrders"):
            result = []
        elif url.path.endswith("/positionRisk"):
            result = [{"symbol": symbol, "positionSide": "BOTH", "positionAmt": str(self.positions[symbol]), "leverage": "1"}]
        elif url.path.endswith("/positionSide/dual"):
            result = {"dualSidePosition": False}
        elif url.path.endswith("/order") and fullurl.get_method() == "POST":
            quantity = Decimal(params["quantity"][0])
            order_price = Decimal(params["price"][0])
            sign = Decimal("1") if params["side"][0] == "BUY" else Decimal("-1")
            if spot:
                self.base[symbol] += sign * quantity
            else:
                self.positions[symbol] += sign * quantity
            result = {"symbol": symbol, "orderId": len(self.orders) + 1, "clientOrderId": params["newClientOrderId"][0],
                      "status": "FILLED", "origQty": str(quantity), "executedQty": str(quantity),
                      "cummulativeQuoteQty": str(quantity * order_price), "avgPrice": str(order_price)}
            result.update(side=params['side'][0], type=params['type'][0],
                          timeInForce=params['timeInForce'][0], price=params['price'][0])
            if not spot:
                result.update(reduceOnly=params.get('reduceOnly', ['false'])[0] == 'true', positionSide='BOTH')
            if spot:
                result["fills"] = [{"price": str(order_price), "qty": str(quantity), "commission": "0", "commissionAsset": "USDT"}]
            self.orders[params["newClientOrderId"][0]] = result
        elif url.path.endswith("/order"):
            result = self.orders[params["origClientOrderId"][0]]
        elif url.path.endswith("/myTrades"):
            row = next(row for row in self.orders.values() if str(row["orderId"]) == params["orderId"][0])
            result = [{"symbol": symbol, "orderId": row["orderId"], "qty": row["executedQty"],
                       "commission": "0", "commissionAsset": "USDT"}]
        else:
            raise ValueError("unsupported offline route")
        return DemoResponse(result)


def demo_bot(settings: Settings, directory: Path) -> tuple[TestnetBot, DemoTransport]:
    assert settings.testnet is not None and settings.research is not None
    settings = settings.model_copy(update={
        "testnet": settings.testnet.model_copy(update={"journal_path": directory / "execution.sqlite"}),
        "research": settings.research.model_copy(update={"sqlite_path": directory / "research.sqlite"}),
    })
    transport = DemoTransport()
    assert settings.testnet is not None
    policy = TestnetExecutionPolicy(settings.testnet.max_order_notional)
    clients = [TestnetExecutionClient(market, TestnetCredentials("offline-key", "offline-secret"), policy,
                                      opener=transport) for market in (ExecutionMarket.SPOT, ExecutionMarket.USDTM)]
    return TestnetBot(settings, *clients), transport


def run_demo(settings: Settings) -> int:
    directory = Path("data/testnet") / f"offline-demo-{uuid4().hex}"
    directory.mkdir(parents=True)
    bot, transport = demo_bot(settings, directory)
    print("OFFLINE MOCK DEMO: no network, no credentials, no actual exchange fills")
    for cycle in bot.run_once():
        print(exact_json(cycle))
    assert settings.research is not None
    for symbol in settings.symbols:
        quantity = settings.research.sizing[symbol].target_quantity
        assert quantity is not None
        result = bot.round_trip(symbol, quantity)
        print(exact_json(result))
        if result.state.value != "CLOSED":
            return 1
    print(exact_json({"status": "DEMO_COMPLETE", "directory": str(directory.resolve()),
                     "mock_orders": len(transport.orders), "network_requests": 0}))
    return 0
