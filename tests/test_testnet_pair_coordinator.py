from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from crypto_arbitrage_bot.execution.pair_coordinator import (
    ExecutionJournalError as JournalError,
)
from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairExecutionState as PairState,
)
from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairOrderPlan as PairPlan,
)
from crypto_arbitrage_bot.execution.pair_coordinator import (
    SQLitePairExecutionJournal as PairJournal,
)
from crypto_arbitrage_bot.execution.pair_coordinator import (
    TestnetPairCoordinator as PairCoordinator,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind as ErrorKind,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionError as ExecutionError,
)
from crypto_arbitrage_bot.execution.types import (
    ExecutionMarket,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.market import Side


@pytest.fixture
def tmp_path():
    """Use a test-owned workspace directory; Windows temp ACLs are restricted here."""
    path = Path(__file__).resolve().parents[1] / "data" / f"pair-test-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        database = path / "execution.sqlite"
        if database.exists():
            database.unlink()
        path.rmdir()


class Gateway:
    def __init__(self, market, outcomes=(), queries=()):
        self.market = market
        self.outcomes = list(outcomes)
        self.queries = list(queries)
        self.orders = []
        self.query_ids = []

    def place_ioc_limit_order(self, order):
        self.orders.append(order)
        result = self.outcomes.pop(0) if self.outcomes else order.quantity
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, OrderReceipt):
            return result
        executed = result
        return OrderReceipt(
            self.market, order.symbol, str(len(self.orders)), order.client_order_id,
            OrderState.FILLED if executed == order.quantity else OrderState.EXPIRED,
            order.quantity, executed, executed * order.limit_price, order.limit_price if executed else Decimal('0'),
            side=order.side, order_type='LIMIT', time_in_force='IOC', limit_price=order.limit_price,
            reduce_only=order.reduce_only if self.market is ExecutionMarket.USDTM else None,
            position_side='BOTH' if self.market is ExecutionMarket.USDTM else None,
        )

    def get_order(self, symbol, client_order_id):
        self.query_ids.append(client_order_id)
        result = self.queries.pop(0) if self.queries else None
        if isinstance(result, BaseException):
            raise result
        if result is None:
            raise ExecutionError(ErrorKind.NETWORK_ERROR, "query unavailable", market=self.market)
        return result


def receipt(market, cid, quantity, *, executed=None):
    filled = quantity if executed is None else executed
    return OrderReceipt(
        market, "BTCUSDT", f"{market.value}-existing", cid,
        OrderState.FILLED if filled == quantity else OrderState.EXPIRED,
        quantity, filled, None, None,
    )


def make_plan(attempt="attempt-1", quantity=Decimal("0.001")):
    return PairPlan(
        attempt,
        IocLimitOrder(ExecutionMarket.SPOT, "BTCUSDT", Side.BUY, quantity,
                      Decimal("50000"), f"{attempt}-se"),
        IocLimitOrder(ExecutionMarket.USDTM, "BTCUSDT", Side.SELL, quantity,
                      Decimal("49999"), f"{attempt}-pe"),
        IocLimitOrder(ExecutionMarket.SPOT, "BTCUSDT", Side.SELL, quantity,
                      Decimal("49990"), f"{attempt}-sx"),
        IocLimitOrder(ExecutionMarket.USDTM, "BTCUSDT", Side.BUY, quantity,
                      Decimal("50010"), f"{attempt}-px"),
    )


def coordinator(tmp_path: Path, spot, perp):
    journal = PairJournal(tmp_path / "execution.sqlite")
    journal.initialize()
    return PairCoordinator(spot, perp, journal), journal


def test_complete_pair_open_and_close_are_journaled_and_idempotent(tmp_path):
    spot = Gateway(ExecutionMarket.SPOT)
    perp = Gateway(ExecutionMarket.USDTM)
    runner, journal = coordinator(tmp_path, spot, perp)
    plan = make_plan()

    opened = runner.open_pair(plan)
    assert opened.state is PairState.HEDGED
    assert opened.spot_entry.executed_quantity == Decimal("0.001")
    assert opened.perp_entry.executed_quantity == Decimal("0.001")
    duplicate = runner.open_pair(plan)
    assert duplicate.state is PairState.DUPLICATE_BLOCKED
    assert duplicate.previous_state is PairState.HEDGED
    assert len(spot.orders) == len(perp.orders) == 1

    closed = runner.close_pair(plan)
    assert closed.state is PairState.CLOSED
    assert closed.spot_exit.executed_quantity == Decimal("0.001")
    assert closed.perp_exit.executed_quantity == Decimal("0.001")
    assert [order.side for order in spot.orders] == [Side.BUY, Side.SELL]
    assert [order.side for order in perp.orders] == [Side.SELL, Side.BUY]
    assert journal.get(plan.attempt_id) == closed
    assert runner.close_pair(plan).state is PairState.DUPLICATE_BLOCKED
    assert len(spot.orders) == len(perp.orders) == 2


def test_partial_spot_fill_is_flattened_without_submitting_perp(tmp_path):
    spot = Gateway(ExecutionMarket.SPOT, [Decimal("0.0005"), Decimal("0.0005")])
    perp = Gateway(ExecutionMarket.USDTM)
    runner, _ = coordinator(tmp_path, spot, perp)

    result = runner.open_pair(make_plan())

    assert result.state is PairState.FLATTENED
    assert result.failure_code == "SPOT_ENTRY_PARTIAL"
    assert len(spot.orders) == 2
    assert spot.orders[1].side is Side.SELL
    assert spot.orders[1].quantity == Decimal("0.0005")
    assert perp.orders == []


def test_partial_perp_fill_flattens_both_known_legs(tmp_path):
    spot = Gateway(ExecutionMarket.SPOT)
    perp = Gateway(ExecutionMarket.USDTM, [Decimal("0.0005"), Decimal("0.0005")])
    runner, _ = coordinator(tmp_path, spot, perp)

    result = runner.open_pair(make_plan())

    assert result.state is PairState.FLATTENED
    assert result.failure_code == "PERP_ENTRY_PARTIAL"
    assert [(order.side, order.quantity) for order in spot.orders] == [
        (Side.BUY, Decimal("0.001")), (Side.SELL, Decimal("0.001")),
    ]
    assert [(order.side, order.quantity) for order in perp.orders] == [
        (Side.SELL, Decimal("0.001")), (Side.BUY, Decimal("0.0005")),
    ]


def test_ambiguous_spot_submission_is_queried_once_then_never_retried(tmp_path):
    unknown = ExecutionError(
        ErrorKind.SUBMISSION_UNKNOWN, "timeout", market=ExecutionMarket.SPOT,
        submission_may_have_reached_exchange=True,
    )
    spot = Gateway(ExecutionMarket.SPOT, [unknown])
    perp = Gateway(ExecutionMarket.USDTM)
    runner, _ = coordinator(tmp_path, spot, perp)
    plan = make_plan()

    result = runner.open_pair(plan)
    duplicate = runner.open_pair(plan)

    assert result.state is PairState.RECOVERY_REQUIRED
    assert result.failure_code == "SPOT_SUBMISSION_UNKNOWN"
    assert spot.query_ids == [plan.spot_entry.client_order_id]
    assert len(spot.orders) == 1
    assert perp.orders == []
    assert duplicate.state is PairState.DUPLICATE_BLOCKED


def test_nonterminal_queried_order_stops_for_manual_recovery(tmp_path):
    unknown = ExecutionError(
        ErrorKind.SUBMISSION_UNKNOWN, "timeout", market=ExecutionMarket.SPOT,
        submission_may_have_reached_exchange=True,
    )
    still_open = OrderReceipt(
        ExecutionMarket.SPOT, "BTCUSDT", "1", "attempt-1-se", OrderState.NEW,
        Decimal("0.001"), Decimal("0"), None, None,
        side=Side.BUY, order_type='LIMIT', time_in_force='IOC', limit_price=Decimal('50000'),
    )
    spot = Gateway(ExecutionMarket.SPOT, [unknown], [still_open])
    perp = Gateway(ExecutionMarket.USDTM)
    runner, _ = coordinator(tmp_path, spot, perp)

    result = runner.open_pair(make_plan())

    assert result.state is PairState.RECOVERY_REQUIRED
    assert result.spot_entry.status is OrderState.NEW
    assert result.failure_code == "SPOT_SUBMISSION_UNKNOWN"
    assert perp.orders == []


def test_known_failed_second_leg_flattens_first_leg(tmp_path):
    spot = Gateway(ExecutionMarket.SPOT)
    perp = Gateway(ExecutionMarket.USDTM, [ExecutionError(
        ErrorKind.HTTP_REJECTED, "rejected", market=ExecutionMarket.USDTM,
    )])
    runner, _ = coordinator(tmp_path, spot, perp)

    result = runner.open_pair(make_plan())

    assert result.state is PairState.FLATTENED
    assert result.failure_code == "PERP_ENTRY_FAILED"
    assert len(spot.orders) == 2
    assert len(perp.orders) == 1


def test_journal_rejects_reusing_attempt_id_for_changed_plan(tmp_path):
    _, journal = coordinator(tmp_path, Gateway(ExecutionMarket.SPOT), Gateway(ExecutionMarket.USDTM))
    first = make_plan()
    assert journal.create(first)
    assert not journal.create(first)
    changed = make_plan(quantity=Decimal("0.002"))
    with pytest.raises(ValueError, match="different order plan"):
        journal.create(changed)


def test_journal_failure_happens_before_any_order_submission(tmp_path):
    spot = Gateway(ExecutionMarket.SPOT)
    perp = Gateway(ExecutionMarket.USDTM)
    runner = PairCoordinator(spot, perp, PairJournal(tmp_path / "not-initialized.sqlite"))

    with pytest.raises(JournalError):
        runner.open_pair(make_plan())

    assert spot.orders == []
    assert perp.orders == []


def test_incomplete_exit_is_recovery_required_and_never_retried(tmp_path):
    spot = Gateway(ExecutionMarket.SPOT)
    perp = Gateway(ExecutionMarket.USDTM)
    runner, _ = coordinator(tmp_path, spot, perp)
    plan = make_plan()
    assert runner.open_pair(plan).state is PairState.HEDGED
    spot.outcomes.append(Decimal("0.0005"))

    result = runner.close_pair(plan)
    duplicate = runner.close_pair(plan)

    assert result.state is PairState.RECOVERY_REQUIRED
    assert result.failure_code == "SPOT_EXIT_INCOMPLETE"
    assert duplicate.state is PairState.DUPLICATE_BLOCKED
    assert duplicate.previous_state is PairState.RECOVERY_REQUIRED
    assert len(spot.orders) == 2
    assert len(perp.orders) == 1
