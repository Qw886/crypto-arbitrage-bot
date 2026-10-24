"""Testnet-only execution orchestration. No financial formula is implemented here."""

import logging
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from pathlib import Path
from threading import Event, get_ident
from typing import Literal, cast
from uuid import uuid4

from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.arbitrage.funding_projection import projected_scenario_definitions
from crypto_arbitrage_bot.arbitrage.scenario import evaluate_scenarios
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.aborted_entry_recovery import recover_aborted_entry
from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.balance_frame_store import BalanceFrameStore
from crypto_arbitrage_bot.execution.balance_ledger import (
    balance_entry_blocker,
    reconcile_balance_ledger,
)
from crypto_arbitrage_bot.execution.exit_market_data import validate_exit_depth
from crypto_arbitrage_bot.execution.exit_recovery import recover_perp_exit
from crypto_arbitrage_bot.execution.funding_ledger import reconcile_funding_ledger
from crypto_arbitrage_bot.execution.market_data import ExitDepthObservation, TestnetCollector
from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairExecutionResult,
    PairExecutionState,
    PairOrderPlan,
    SQLitePairExecutionJournal,
    TestnetPairCoordinator,
)
from crypto_arbitrage_bot.execution.spot_exit_recovery import recover_spot_exit
from crypto_arbitrage_bot.execution.strategy_admission import (
    refresh_entry_admission,
    review_automatic_entry,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionClient,
    TestnetExecutionError,
    TestnetExecutionPolicy,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder, SymbolTradingRules
from crypto_arbitrage_bot.execution.unhedged_entry_recovery import (
    UNHEDGED_RECOVERY_CODES,
    recover_unhedged_entry,
)
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.calculation import FillStatus, TradeLeg
from crypto_arbitrage_bot.models.market import Market, OrderBook, Side
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.models.strategy_admission import StrategyAdmissionResult
from crypto_arbitrage_bot.runner import CycleStatus, ResearchRunner
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


class ExecutionGuardError(RuntimeError):
    """A bounded testnet request failed preflight; no entry should be submitted."""

    def __init__(self, reason: str, *, diagnostics: Mapping[str, object] | None = None) -> None:
        self.reason = reason[:240]
        self.diagnostics = dict(diagnostics or {})
        super().__init__(self.reason)


class ProcessLockLease:
    """Opaque access to a lock held by this process and thread; not trade permission."""

    __slots__ = ()

    def require_owned(self, path: Path) -> None:
        record = _LOCK_LEASES.get(id(self))
        resolved = path.resolve()
        if (record is None or record[0] is not self or record[1] != resolved
                or record[2] != os.getpid() or record[3] != get_ident()):
            raise ExecutionGuardError('LIVE_SAME_THREAD_JOURNAL_LOCK_REQUIRED')
        try:
            current = resolved.stat()
        except OSError:
            raise ExecutionGuardError('LIVE_SAME_THREAD_JOURNAL_LOCK_REQUIRED') from None
        if (current.st_dev, current.st_ino) != record[4:]:
            raise ExecutionGuardError('LIVE_SAME_THREAD_JOURNAL_LOCK_REQUIRED')


_LOCK_LEASES: dict[int, tuple[ProcessLockLease, Path, int, int, int, int]] = {}


@contextmanager
def lease_process_lock(path: Path) -> Iterator[ProcessLockLease]:
    """Hold the original OS lock and lend it explicitly without releasing it."""
    resolved = path.resolve()
    with process_lock(resolved):
        identity = resolved.stat()
        lease = ProcessLockLease()
        _LOCK_LEASES[id(lease)] = (lease, resolved, os.getpid(), get_ident(), identity.st_dev, identity.st_ino)
        try:
            yield lease
        finally:
            _LOCK_LEASES.pop(id(lease), None)


@dataclass(frozen=True)
class BotCycle:
    symbol: str
    status: str
    opportunity_id: str | None = None
    pair: PairExecutionResult | None = None
    reason: str | None = None
    estimated_net_profit: Decimal | None = None
    estimated_funding: Decimal | None = None
    next_funding_at: datetime | None = None
    planned_exit_at: datetime | None = None
    decision_reason_codes: tuple[str, ...] = ()
    admission: StrategyAdmissionResult | None = None


@contextmanager
def process_lock(path: Path) -> Iterator[None]:
    """OS-owned lock releases on process death, without stale PID-file guesses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ExecutionGuardError("another testnet runner owns this journal") from None
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ExecutionGuardError("another testnet runner owns this journal") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class TestnetBot:
    def __init__(self, settings: Settings, spot: TestnetExecutionClient, perp: TestnetExecutionClient,
                 *, clock: Callable[[], datetime] = utc_now, stop: Event | None = None) -> None:
        if settings.testnet is None or settings.research is None:
            raise ValueError("explicit research and testnet configuration required")
        if settings.testnet.journal_path.resolve() == settings.research.sqlite_path.resolve():
            raise ValueError("execution journal must be separate from research archive")
        self.settings = settings
        self.config = settings.testnet
        self.research = settings.research
        self.spot, self.perp, self.clock = spot, perp, clock
        self.stop = stop if stop is not None else Event()
        self.collector = TestnetCollector(spot, perp, clock=clock)
        self.journal = SQLitePairExecutionJournal(self.config.journal_path)
        self.journal.initialize()
        self.coordinator = TestnetPairCoordinator(spot, perp, self.journal)
        self.repository = SQLiteOpportunityRepository(SQLiteConfig(self.research.sqlite_path))
        self.repository.initialize()
        # No mainnet funding scheduler is attached to the isolated testnet runner.
        self.research_runner = ResearchRunner(settings, self.collector, self.repository, clock=clock,
                                              stop_signal=self.stop)
        self.logger = logging.getLogger("crypto_arbitrage_bot.testnet")
        self.status_callback: Callable[[BotCycle], None] | None = None
        self._control_thread: int | None = None
        self._control_lock_lease: ProcessLockLease | None = None
        self._control_manage_only = False
        self._control_entry_stop: Callable[[], bool] | None = None
        self.balance_directory = self.config.balance_evidence_directory or self.config.journal_path.parent / 'balance-evidence'
        if self.balance_directory.resolve() in (self.config.journal_path.resolve(), self.research.sqlite_path.resolve()):
            raise ValueError('balance evidence directory must be separate from database files')

    def check(self, symbol: str) -> SnapshotAssembly:
        assembly = self.collector.collect(symbol)
        self._require_quality(assembly, "INVALID_TESTNET_MARKET_DATA")
        for client in (self.spot, self.perp):
            client.get_symbol_rules(symbol)
            account = client.get_account_state(symbol)
            if not account.can_trade or account.open_order_count:
                raise ExecutionGuardError("account disabled or unmanaged orders present")
            if client.market is ExecutionMarket.USDTM and (
                not account.one_way or not account.single_asset or account.leverage != 1
            ):
                raise ExecutionGuardError("USDT-M must already use one-way, single-asset, 1x settings")
        return assembly

    def _require_quality(self, assembly: SnapshotAssembly, reason: str) -> None:
        """Keep bounded provenance diagnostics without serializing books or raw payloads."""
        quality = validate_market_data(assembly, self.research.data_quality, evaluated_at=self.clock())
        if quality.valid:
            return
        sources: dict[str, object] = {}
        for name in ("spot_depth", "perp_depth", "premium_index", "open_interest"):
            source = getattr(assembly.sources, name)
            if source is None:
                sources[name] = {"missing": True}
                continue
            sources[name] = {
                "endpoint": source.request.endpoint.value,
                "started_at": source.started_at, "finished_at": source.finished_at,
                "received_at": source.received_at, "elapsed_ms": source.elapsed_ms,
                "error_kind": source.error.kind.value if source.error else None,
                "api_code": source.error.api_code if source.error else None,
                "phase": source.error.phase if source.error else None,
                "http_status": source.status_code,
            }
        raise ExecutionGuardError(reason, diagnostics={
            "evaluated_at": quality.evaluated_at,
            "quality_reason_codes": tuple(code.value for code in quality.reason_codes),
            "sampling_skew": quality.sampling_skew, "sources": sources,
        })

    def prepare(self, symbol: str) -> None:
        """Explicit Demo account setup, without submitting any trading order."""
        with process_lock(self.config.journal_path.with_suffix(".lock")):
            self._assert_open_allowed(symbol)
            account = self.perp.ensure_one_x(symbol)
            self.logger.info(exact_json({"stage": "TESTNET_PREPARE", "symbol": symbol,
                                        "leverage": account.leverage, "orders_submitted": False}))

    def _limit(self, assembly: SnapshotAssembly, leg: TradeLeg, quantity: Decimal,
               rules: SymbolTradingRules) -> Decimal:
        snapshot = assembly.snapshot
        assert snapshot is not None
        book = snapshot.spot_book if leg.market.value == "SPOT" else snapshot.perp_book
        return self._book_limit(book, leg, quantity, rules)

    def _book_limit(self, book: OrderBook, leg: TradeLeg, quantity: Decimal,
                    rules: SymbolTradingRules) -> Decimal:
        """Original VWAP/IOC/tick logic, shared by entry and risk-reduction quotes."""
        fill = simulate_fill(book, leg, quantity)
        if fill.status is not FillStatus.FILLED:
            raise ExecutionGuardError("INSUFFICIENT_TESTNET_DEPTH")
        with localcontext() as ctx:
            ctx.prec = 50
            direction = Decimal("1") if leg.side is Side.BUY else Decimal("-1")
            price = fill.fills[-1].price * (1 + direction * self.config.max_price_deviation)
            rounding = ROUND_CEILING if leg.side is Side.BUY else ROUND_FLOOR
            return (price / rules.tick_size).to_integral_value(rounding=rounding) * rules.tick_size

    def _collect_exit_depth(self, symbol: str, *, markets: tuple[Market, ...]) -> ExitDepthObservation:
        observation = self.collector.collect_exit_depth(symbol, markets=markets)
        self._require_exit_quality(observation, "INVALID_TESTNET_MARKET_DATA", markets=markets)
        return observation

    def _require_exit_quality(
        self, observation: ExitDepthObservation, reason: str, *, markets: tuple[Market, ...],
    ) -> None:
        quality = validate_exit_depth(observation, self.research.data_quality,
                                      evaluated_at=self.clock(), markets=markets)
        if not quality.valid:
            sources: dict[str, object] = {}
            for name in ("spot_depth", "perp_depth"):
                source = getattr(observation, name)
                if source is not None:
                    sources[name] = {
                        "endpoint": source.request.url,
                        "started_at": source.started_at, "received_at": source.received_at,
                        "finished_at": source.finished_at, "http_status": source.status_code,
                        "error_kind": source.error.kind.value if source.error else None,
                        "api_code": source.error.api_code if source.error else None,
                        "phase": source.error.phase if source.error else None,
                    }
            raise ExecutionGuardError(reason, diagnostics={
                "scope": "TESTNET_EXIT_DEPTH_ONLY", "markets": markets,
                "evaluated_at": quality.evaluated_at, "sampling_skew": quality.sampling_skew,
                "quality_reason_codes": tuple(code.value for code in quality.reason_codes), "sources": sources,
            })

    def _exit_limit(self, observation: ExitDepthObservation, leg: TradeLeg, quantity: Decimal,
                    rules: SymbolTradingRules) -> Decimal:
        if leg not in (TradeLeg.SPOT_EXIT, TradeLeg.PERP_EXIT):
            raise ExecutionGuardError("EXIT_QUOTE_CANNOT_AUTHORIZE_ENTRY")
        book = observation.book(leg.market)
        if book.symbol != rules.symbol or rules.symbol != observation.symbol:
            raise ExecutionGuardError("EXIT_QUOTE_IDENTITY_MISMATCH")
        return self._book_limit(book, leg, quantity, rules)

    def _plan(self, symbol: str, quantity: Decimal, attempt_id: str,
              *, opportunity: OpportunityRecord | None = None) -> PairOrderPlan:
        self._assert_open_allowed(symbol)
        # Slow configuration/account calls must precede the executable quote.
        if self.config.balance_evidence_enabled:
            if self.journal.get(attempt_id) is not None:
                raise ExecutionGuardError('BALANCE_ATTEMPT_ALREADY_EXISTS')
            if blocker := balance_entry_blocker(self):
                raise ExecutionGuardError(blocker)
            before = BalanceFrameStore(self.balance_directory).capture_before(attempt_id, symbol, quantity,
                self.spot, self.perp, clock=self.clock)
            if not before.success or before.frame is None:
                raise ExecutionGuardError('BALANCE_PREENTRY_EVIDENCE_FAILED', diagnostics={
                    'error_code': before.error_code.value if before.error_code else None,
                    'error_kind': before.error_kind, 'source_error': before.diagnostics})
            if not BalanceCheckRepository(self.journal).register_before(before.frame):
                raise ExecutionGuardError('BALANCE_PREENTRY_MARKER_CONFLICT')
        sr, pr = self.spot.get_symbol_rules(symbol), self.perp.get_symbol_rules(symbol)
        spot, perp = self.spot.get_account_state(symbol), self.perp.get_account_state(symbol)
        if not spot.can_trade or not perp.can_trade or spot.open_order_count or perp.open_order_count:
            raise ExecutionGuardError("ACCOUNT_NOT_READY")
        if not perp.one_way or not perp.single_asset or perp.position_quantity != 0:
            raise ExecutionGuardError("UNMANAGED_POSITION_OR_UNSUPPORTED_ACCOUNT_MODE")
        if perp.leverage != 1:
            try:
                perp = self.perp.ensure_one_x(symbol)
            except TestnetExecutionError as error:
                raise ExecutionGuardError("TESTNET_1X_PREPARATION_FAILED: " + error.kind.value) from None
        assembly = self.collector.collect(symbol)
        self._require_quality(assembly, "INVALID_TESTNET_MARKET_DATA")
        orders = tuple(IocLimitOrder(market, symbol, leg.side, quantity, self._limit(assembly, leg, quantity, rules),
                                    f"{attempt_id[:29]}-{suffix}", reduce_only=leg is TradeLeg.PERP_EXIT)
                       for market, leg, rules, suffix in (
                           (ExecutionMarket.SPOT, TradeLeg.SPOT_ENTRY, sr, "se"),
                           (ExecutionMarket.USDTM, TradeLeg.PERP_ENTRY, pr, "pe"),
                           (ExecutionMarket.SPOT, TradeLeg.SPOT_EXIT, sr, "sx"),
                           (ExecutionMarket.USDTM, TradeLeg.PERP_EXIT, pr, "px")))
        if opportunity is not None:
            orders = self._cap_research_entry_prices(orders, assembly, opportunity, sr, pr)
        for order, rules, client in zip(orders, (sr, pr, sr, pr), (self.spot, self.perp, self.spot, self.perp), strict=True):
            client._validate_order(order, rules)
        spot_required = orders[0].quantity * orders[0].limit_price * (1 + self.settings.fees.spot_entry)
        perp_required = orders[1].quantity * orders[1].limit_price * (1 + self.config.collateral_buffer)
        if spot.quote_available < spot_required or perp.quote_available < perp_required:
            raise ExecutionGuardError("INSUFFICIENT_TESTNET_BALANCE")
        self._require_quality(assembly, "MARKET_EXPIRED_DURING_PREFLIGHT")
        plan = PairOrderPlan(attempt_id, *orders)
        assumptions = self.research.scenarios[cast(Literal["BTCUSDT", "ETHUSDT"], symbol)]
        if opportunity is not None and assumptions.funding_projection is not None:
            plan = self._funding_entry_plan(plan, assembly, opportunity)
        return plan

    def _cap_research_entry_prices(
        self, orders: tuple[IocLimitOrder, ...], assembly: SnapshotAssembly,
        opportunity: OpportunityRecord, spot_rules: SymbolTradingRules,
        perp_rules: SymbolTradingRules,
    ) -> tuple[IocLimitOrder, ...]:
        """Intersect IOC tolerance with the existing research price ceiling.

        A configured deviation is a transport price bound, not permission to
        spend the research profit buffer. Keep the original best-quote guard:
        buys may not exceed the recorded ask, shorts may not fall below the
        recorded bid. Prove complete *current* depth inside these tighter caps
        before either leg is submitted. Neither snapshot nor estimate is edited.
        Explicit round-trip/connectivity tests without an Opportunity retain
        their existing IOC tolerance and are not represented as strategy PASS.
        """
        if (opportunity.symbol != assembly.symbol or opportunity.matched_quantity != orders[0].quantity
                or opportunity.matched_quantity != orders[1].quantity):
            raise ExecutionGuardError("RESEARCH_ENTRY_IDENTITY_MISMATCH")
        if (opportunity.decision != "PASS" or opportunity.data_quality_status != "VALID"
                or opportunity.net_profit is None or opportunity.net_return is None
                or opportunity.net_profit <= 0 or opportunity.net_return <= 0):
            raise ExecutionGuardError("PERSISTED_RESEARCH_NOT_ELIGIBLE")
        snapshot = assembly.snapshot
        assert snapshot is not None
        capped = list(orders)
        for index, leg, reference, rules, book in (
            (0, TradeLeg.SPOT_ENTRY, opportunity.spot_ask, spot_rules, snapshot.spot_book),
            (1, TradeLeg.PERP_ENTRY, opportunity.perp_bid, perp_rules, snapshot.perp_book),
        ):
            if reference is None:
                raise ExecutionGuardError("MISSING_RESEARCH_PRICE_REFERENCE")
            require_decimal(reference, "research entry price", positive=True)
            with localcontext() as context:
                context.prec = 50
                rounding = ROUND_FLOOR if leg.side is Side.BUY else ROUND_CEILING
                boundary = (reference / rules.tick_size).to_integral_value(rounding=rounding) * rules.tick_size
            price = min(orders[index].limit_price, boundary) if leg.side is Side.BUY else max(
                orders[index].limit_price, boundary)
            capped[index] = replace(orders[index], limit_price=price)
            executable = replace(book, asks=tuple(level for level in book.asks if level.price <= price)) \
                if leg.side is Side.BUY else replace(
                    book, bids=tuple(level for level in book.bids if level.price >= price))
            if not simulate_fill(executable, leg, orders[index].quantity).complete:
                raise ExecutionGuardError("INSUFFICIENT_DEPTH_WITHIN_RESEARCH_PRICE_CAP",
                                          diagnostics={"leg": leg.value, "limit_price": price})
        return tuple(capped)

    def _funding_entry_plan(self, plan: PairOrderPlan, assembly: SnapshotAssembly,
                            opportunity: OpportunityRecord) -> PairOrderPlan:
        """Fresh scenario recheck and durable exit plan; uses existing engines."""
        symbol = cast(Literal["BTCUSDT", "ETHUSDT"], plan.symbol)
        policy = self.research.scenarios[symbol].funding_projection
        assert policy is not None and assembly.snapshot is not None
        now = self.clock()
        settlement = assembly.snapshot.next_funding_time
        lead = settlement - now
        if opportunity.symbol != plan.symbol or opportunity.matched_quantity != plan.spot_entry.quantity:
            raise ExecutionGuardError("FUNDING_OPPORTUNITY_MISMATCH")
        if settlement != opportunity.next_funding_time:
            raise ExecutionGuardError("FUNDING_SETTLEMENT_CHANGED")
        if not timedelta(seconds=policy.min_entry_lead_seconds) <= lead <= timedelta(seconds=policy.entry_window_seconds):
            raise ExecutionGuardError("OUTSIDE_FUNDING_ENTRY_WINDOW")
        definitions = projected_scenario_definitions(assembly.snapshot, now, self.research.scenarios[symbol])
        results = evaluate_scenarios(assembly, self.research.sizing[symbol].as_estimate(
            self.settings.fees.as_rates(), self.research.data_quality), definitions, evaluated_at=now)
        decision = decide_scenarios(results, self.research.decision)
        if (decision.status.value != "PASS" or any(not item.complete for item in results)
                or results[0].matched_quantity != plan.spot_entry.quantity):
            raise ExecutionGuardError("FRESH_FUNDING_SCENARIOS_NOT_PASS")
        if not validate_market_data(assembly, self.research.data_quality, evaluated_at=self.clock()).valid:
            raise ExecutionGuardError("FUNDING_PREFLIGHT_MARKET_EXPIRED")
        if settlement - self.clock() < timedelta(seconds=policy.min_entry_lead_seconds):
            raise ExecutionGuardError("FUNDING_ENTRY_DEADLINE_PASSED")
        deadline = settlement - timedelta(seconds=policy.min_entry_lead_seconds)
        return replace(plan, spot_entry=replace(plan.spot_entry, submit_before=deadline),
                       perp_entry=replace(plan.perp_entry, submit_before=deadline),
                       scheduled_exit_at=definitions[0].exit_scenario.closed_at,
                       funding_settlement_at=settlement, opportunity_id=opportunity.id,
                       projection_json=exact_json({"forecast_only": True, "entry_snapshot": assembly.snapshot,
                           "definitions": definitions, "config": results[0].estimate.config,
                           "decision": decision, "config_version": self.research.config_version}))

    def _exit_due(self, plan: PairOrderPlan, opened_at: datetime) -> bool:
        deadline = plan.scheduled_exit_at or opened_at + timedelta(seconds=self.config.holding_seconds)
        return self.clock() >= deadline

    def _assert_open_allowed(self, symbol: str) -> None:
        plans = self.journal.plans()
        active = []
        for plan, opened in plans:
            result = self.journal.get(plan.attempt_id)
            if (result is not None and result.state is PairExecutionState.HEDGED and
                self._exit_due(plan, opened)):
                raise ExecutionGuardError("OVERDUE_EXIT_BLOCKS_NEW_ENTRY")
            if result is not None and result.state not in (
                PairExecutionState.CLOSED, PairExecutionState.FLATTENED, PairExecutionState.FAILED
            ):
                active.append((plan, result))
        if any(r.state is not PairExecutionState.HEDGED for _, r in active):
            raise ExecutionGuardError("UNRESOLVED_EXECUTION_BLOCKS_NEW_ENTRY")
        if len(plans) >= self.config.max_total_attempts:
            raise ExecutionGuardError("SESSION_ATTEMPT_LIMIT")
        if len(active) >= self.config.max_active_pairs or any(p.symbol == symbol for p, _ in active):
            raise ExecutionGuardError("ACTIVE_PAIR_LIMIT")

    def _close(self, plan: PairOrderPlan) -> PairExecutionResult:
        current = self.journal.get(plan.attempt_id)
        assert current is not None and current.spot_entry is not None and current.perp_entry is not None
        spot = self.spot.get_account_state(plan.symbol)
        perp = self.perp.get_account_state(plan.symbol)
        if (not spot.can_trade or not perp.can_trade or spot.open_order_count or perp.open_order_count
                or not perp.one_way or not perp.single_asset or perp.leverage != 1):
            raise ExecutionGuardError("EXIT_ACCOUNT_NOT_READY")
        if current.spot_entry.base_commission is None:
            raise ExecutionGuardError("SPOT_COMMISSION_UNKNOWN")
        quantity = current.spot_entry.executed_quantity - current.spot_entry.base_commission
        if spot.base_available < quantity or perp.position_quantity != -current.perp_entry.executed_quantity:
            raise ExecutionGuardError("ACCOUNT_EXPOSURE_DIFFERS_FROM_JOURNAL")
        sr, pr = self.spot.get_symbol_rules(plan.symbol), self.perp.get_symbol_rules(plan.symbol)
        markets = (Market.SPOT, Market.USDTM)
        depths = self._collect_exit_depth(plan.symbol, markets=markets)
        sp = self._exit_limit(depths, TradeLeg.SPOT_EXIT, quantity, sr)
        pp = self._exit_limit(depths, TradeLeg.PERP_EXIT, current.perp_entry.executed_quantity, pr)
        self._require_exit_quality(depths, "EXIT_MARKET_EXPIRED_DURING_PREFLIGHT", markets=markets)
        # The gateways still validate each order against these freshly fetched
        # filters; avoid downloading the same bulk exchangeInfo after the quote.
        # Expired rules fall back to the ordinary fresh GET, never stale reuse.
        rule_age = self.research.data_quality.max_data_age
        with self.spot.preflight_rules(sr, max_age=rule_age), self.perp.preflight_rules(pr, max_age=rule_age):
            return self.coordinator.close_pair(plan, spot_price=sp, perp_price=pp)

    def round_trip(self, symbol: str, quantity: Decimal) -> PairExecutionResult:
        """Explicit virtual-money connection test; bypasses opportunity selection."""
        with process_lock(self.config.journal_path.with_suffix(".lock")):
            plan = self._plan(symbol, quantity, uuid4().hex)
            result = self.coordinator.open_pair(plan)
            if result.state is PairExecutionState.HEDGED:
                return self._close(plan)
            return result

    def _log_execution_failure(self, plan: PairOrderPlan, status: str, error: Exception, *, stage: str) -> None:
        """Keep diagnostic metadata without arbitrary exception text or account/order bodies."""
        payload: dict[str, object] = {"symbol": plan.symbol, "attempt_id": plan.attempt_id,
            "status": status, "stage": stage, "at": self.clock(), "error_type": type(error).__name__,
            "reason": type(error).__name__, "scheduled_exit_at": plan.scheduled_exit_at}
        if isinstance(error, TestnetExecutionError):
            payload.update(reason=error.kind.value, diagnostics=error.diagnostic_fields())
        elif isinstance(error, ExecutionGuardError):
            payload.update(reason=error.reason, diagnostics=error.diagnostics)
        self.logger.error(exact_json(payload))

    def _managed_exits(self) -> None:
        for plan, opened_at in self.journal.plans():
            try:
                prior = self.journal.get(plan.attempt_id)
            except Exception as error:
                self._log_execution_failure(plan, "EXIT_BLOCKED", error, stage="JOURNAL_READ")
                continue
            if prior is not None and prior.state is PairExecutionState.RECOVERY_REQUIRED:
                try:
                    # Resolve original durable IDs before considering a bounded
                    # repair. A lost ACK can mean already CLOSED or HEDGED;
                    # neither permits resending an entry/exit with that ID.
                    # An unexplained/manual recovery state is not permission
                    # to resume normal exposure management automatically.
                    query_codes = {"SPOT_SUBMISSION_UNKNOWN", "PERP_SUBMISSION_UNKNOWN",
                                   "SPOT_EXIT_INCOMPLETE", "PERP_EXIT_INCOMPLETE",
                                   "SPOT_FLATTEN_INCOMPLETE", "PERP_FLATTEN_INCOMPLETE",
                                   "SPOT_COMMISSION_UNKNOWN", "RECONCILIATION_UNRESOLVED"}
                    # This abort validates cached receipts BEFORE generic
                    # reconciliation could replace them with changed evidence.
                    # Never interpret a receipt-less Perp intent as no exposure.
                    if (self.config.balance_evidence_enabled and prior.failure_code in UNHEDGED_RECOVERY_CODES
                            and prior.perp_entry is None and prior.perp_exit is None
                            and not any(i.market is ExecutionMarket.USDTM
                                        for i in self.journal.intents(plan.attempt_id))):
                        result = recover_unhedged_entry(self, plan)
                        self.logger.info(exact_json(result))
                        continue
                    aborted = recover_aborted_entry(self, plan)
                    if aborted is not None:
                        self.logger.info(exact_json(aborted))
                        continue
                    result = (self.coordinator.reconcile(plan)
                              if prior.failure_code in query_codes else prior)
                    if result.state is PairExecutionState.RECOVERY_REQUIRED:
                        result = recover_spot_exit(self, plan)
                        if result.state is PairExecutionState.RECOVERY_REQUIRED:
                            result = recover_perp_exit(self, plan)
                    elif result.state is PairExecutionState.HEDGED and self._exit_due(plan, opened_at):
                        result = self._close(plan)
                    self.logger.info(exact_json(result))
                except Exception as error:
                    self._log_execution_failure(plan, "RECOVERY_BLOCKED", error, stage="RECOVERY")
                # A recovery/unknown order must never become a fresh entry.
                continue
            try:
                result = self.coordinator.reconcile(plan)
            except Exception as error:
                self._log_execution_failure(plan, "EXIT_BLOCKED", error, stage="ORDER_RECONCILIATION")
                continue
            if result.state is PairExecutionState.HEDGED and self._exit_due(plan, opened_at):
                try:
                    result = self._close(plan)
                except Exception as error:
                    self._log_execution_failure(plan, "EXIT_BLOCKED", error, stage="EXIT_PREFLIGHT")
                    continue
                self.logger.info(exact_json(result))

    def manage_existing_once(self) -> tuple[BotCycle, ...]:
        """Manage original journal attempts only; never scan or create an entry.

        Existing deadlines, unknown-order queries and bounded recovery remain
        owned by the original exit/recovery methods. HEDGED is not closed early.
        """
        with self._cycle_lock():
            self._managed_exits()
            pending = {cycle.pair.attempt_id: cycle for cycle in self._exit_priority_cycles()
                       if cycle.pair is not None}
            results = []
            for plan, _ in self.journal.plans():
                pair = self.journal.get(plan.attempt_id)
                results.append(pending.get(plan.attempt_id) or BotCycle(plan.symbol, "RECOVERY_REQUIRED" if pair is None else pair.state.value,
                    pair=pair, reason="JOURNAL_RESULT_MISSING" if pair is None else pair.failure_code,
                    planned_exit_at=plan.scheduled_exit_at))
            return self._publish_cycles(tuple(results))

    @contextmanager
    def controlled_session(self, *, manage_only: bool,
                           entry_stop: Callable[[], bool] | None = None,
                           lock_lease: ProcessLockLease | None = None) -> Iterator[None]:
        """Own the original OS journal lock for the entire bounded controller run.

        Only this bot and this thread may reuse the held lock. Other bots,
        processes and threads still take the original non-reentrant lock.
        """
        if type(manage_only) is not bool:
            raise ValueError("explicit control mode required")
        if self._control_thread is not None:
            raise ExecutionGuardError('CONTROLLED_SESSION_ALREADY_ACTIVE')
        lock_path = self.config.journal_path.with_suffix('.lock')
        if lock_lease is not None:
            if type(lock_lease) is not ProcessLockLease:
                raise ExecutionGuardError('LIVE_SAME_THREAD_JOURNAL_LOCK_REQUIRED')
            lock_lease.require_owned(lock_path)
        with (process_lock(lock_path) if lock_lease is None else nullcontext()):
            self._control_thread = get_ident()
            self._control_lock_lease = lock_lease
            self._control_manage_only = manage_only
            self._control_entry_stop = entry_stop
            try:
                yield
            finally:
                self._control_thread = None
                self._control_lock_lease = None
                self._control_manage_only = False
                self._control_entry_stop = None

    @contextmanager
    def _cycle_lock(self) -> Iterator[None]:
        if self._control_lock_lease is not None:
            self._control_lock_lease.require_owned(self.config.journal_path.with_suffix('.lock'))
        if self._control_thread == get_ident():
            yield
        else:
            with process_lock(self.config.journal_path.with_suffix(".lock")):
                yield

    def _new_entries_blocked(self) -> bool:
        if self._control_lock_lease is not None:
            self._control_lock_lease.require_owned(self.config.journal_path.with_suffix('.lock'))
        return self.stop.is_set() or (self._control_thread == get_ident() and (
            self._control_manage_only or self._control_entry_stop is not None and self._control_entry_stop()))

    def run_once(self) -> tuple[BotCycle, ...]:
        if self._new_entries_blocked():
            return self.manage_existing_once()
        with self._cycle_lock():
            self._managed_exits()
            if priority := self._exit_priority_cycles():
                return self._publish_cycles(priority)
            try:
                reconcile_funding_ledger(self)
            except Exception as error:
                self.logger.error(exact_json({"stage": "TESTNET_FUNDING_LEDGER", "error": type(error).__name__}))
            # Accounting can itself take long enough for another exit to become
            # due. Do not add a whole market scan before the next exit attempt.
            if priority := self._exit_priority_cycles():
                return self._publish_cycles(priority)
            try:
                reconcile_balance_ledger(self)
            except Exception as error:
                self.logger.error(exact_json({'stage': 'TESTNET_BALANCE_LEDGER', 'error': type(error).__name__}))
            if priority := self._exit_priority_cycles():
                return self._publish_cycles(priority)
            results = []
            research_cycles = self.research_runner.run_once()
            # A bounded but slow scan may cross a holding deadline. Manage it
            # before entries or the next poll, without retrying a failed exit twice.
            if self._exit_priority_cycles():
                self._managed_exits()
                if priority := self._exit_priority_cycles():
                    return self._publish_cycles(priority)
            for cycle in research_cycles:
                try:
                    record = None if cycle.opportunity_id is None else self.repository.get_opportunity_by_id(cycle.opportunity_id)
                except Exception as error:
                    results.append(BotCycle(cycle.symbol, "ENTRY_BLOCKED", cycle.opportunity_id,
                                            reason=type(error).__name__))
                    continue
                summary = BotCycle(cycle.symbol, "NO_ENTRY", cycle.opportunity_id)
                if record is not None:
                    summary = replace(summary, estimated_net_profit=record.net_profit,
                        estimated_funding=record.estimated_funding, next_funding_at=record.next_funding_time,
                        decision_reason_codes=record.decision_reason_codes,
                        planned_exit_at=next((s.closed_at for s in record.scenarios
                                             if s.name == record.primary_scenario_name), None))
                if cycle.status is not CycleStatus.SAVED or cycle.decision != "PASS":
                    results.append(replace(summary, status="NO_ENTRY" if cycle.status is CycleStatus.SAVED else
                                            "RESEARCH_FAILED", reason=cycle.failure_reason or cycle.decision))
                    continue
                try:
                    assert cycle.opportunity_id is not None
                    if self._new_entries_blocked():
                        raise ExecutionGuardError("CONTROL_STOP_BLOCKS_NEW_ENTRY")
                    if record is None or record.matched_quantity is None:
                        raise ExecutionGuardError("MISSING_PASS_QUANTITY")
                    if record.decision != "PASS" or record.data_quality_status != "VALID":
                        raise ExecutionGuardError("PERSISTED_RESEARCH_NOT_ELIGIBLE")
                    if (record.net_profit is None or record.net_return is None or
                        record.net_profit <= 0 or record.net_return <= 0):
                        raise ExecutionGuardError("PERSISTED_RESEARCH_NOT_POSITIVE")
                    admission = review_automatic_entry(self.settings, record, evaluated_at=self.clock())
                    summary = replace(summary, admission=admission)
                    if not admission.eligible:
                        raise ExecutionGuardError("STRATEGY_ADMISSION_DENIED: " + ",".join(admission.reason_codes))
                    # A fresh estimate is not silently substituted: cap quote drift
                    # against the stored executable entry references.
                    plan = self._plan(cycle.symbol, record.matched_quantity, uuid4().hex, opportunity=record)
                    if (record.spot_ask is None or record.perp_bid is None or
                        plan.spot_entry.limit_price > record.spot_ask or
                        plan.perp_entry.limit_price < record.perp_bid):
                        raise ExecutionGuardError("EXECUTION_PRICES_WORSE_THAN_RESEARCH_REFERENCE")
                    admission = refresh_entry_admission(admission, self.settings, record, evaluated_at=self.clock())
                    summary = replace(summary, admission=admission)
                    if not admission.eligible:
                        raise ExecutionGuardError("STRATEGY_ADMISSION_DENIED: " + ",".join(admission.reason_codes))
                    if self._new_entries_blocked():
                        raise ExecutionGuardError("CONTROL_STOP_BLOCKS_NEW_ENTRY")
                    pair = self.coordinator.open_pair(plan)
                    results.append(replace(summary, status=pair.state.value, pair=pair))
                except Exception as error:
                    results.append(replace(summary, status="ENTRY_BLOCKED",
                                            reason=str(error)[:240] if isinstance(error, ExecutionGuardError)
                                            else type(error).__name__))
            return self._publish_cycles(tuple(results))

    def _exit_priority_cycles(self) -> tuple[BotCycle, ...]:
        """Keep overdue/uncertain exits ahead of optional research and entry.

        Do not retry a POST, alter the persisted plan, or change the poll wait.
        The next normal loop reconciles/manages the same original attempt.
        """
        pending: dict[str, tuple[PairOrderPlan, PairExecutionResult, datetime]] = {}
        for plan, opened_at in self.journal.plans():
            pair = self.journal.get(plan.attempt_id)
            if pair is not None and (pair.state is PairExecutionState.RECOVERY_REQUIRED
                    or pair.state in (PairExecutionState.CLOSING_SPOT, PairExecutionState.CLOSING_PERP)
                    or pair.state is PairExecutionState.HEDGED and self._exit_due(plan, opened_at)):
                pending[plan.symbol] = (plan, pair, opened_at)
        if not pending:
            return ()
        cycles = []
        for symbol in self.settings.symbols:
            item = pending.get(symbol)
            cycles.append(BotCycle(symbol, "EXIT_PRIORITY", pair=item[1] if item else None,
                reason="MANAGED_EXIT_PENDING" if item else "OTHER_SYMBOL_EXIT_PENDING",
                planned_exit_at=(item[0].scheduled_exit_at or item[2] + timedelta(seconds=self.config.holding_seconds))
                                if item else None))
        return tuple(cycles)

    def _publish_cycles(self, results: tuple[BotCycle, ...]) -> tuple[BotCycle, ...]:
        for result in results:
            self.logger.info(exact_json(result))
            if self.status_callback is not None:
                self.status_callback(result)
        return results

    def run_forever(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    self.run_once()
                except Exception as error:
                    self.logger.error(exact_json({"stage": "TESTNET_CYCLE", "error": type(error).__name__}))
                if self.stop.wait(self.settings.poll_interval_seconds):
                    break
        except KeyboardInterrupt:
            self.stop.set()
        # Shutdown preserves managed positions; next start checks the journal.


def create_testnet_bot(settings: Settings, *, environ: Mapping[str, str] | None = None) -> TestnetBot:
    if settings.testnet is None:
        raise ValueError("[testnet] configuration required")
    policy = TestnetExecutionPolicy(settings.testnet.max_order_notional, settings.testnet.timeout_seconds)
    return TestnetBot(settings, TestnetExecutionClient.from_env(ExecutionMarket.SPOT, policy, environ=environ),
                      TestnetExecutionClient.from_env(ExecutionMarket.USDTM, policy, environ=environ))
