"""Bounded execution snapshots and original terminal recovery with requests blocked."""

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairExecutionResult,
    SQLitePairExecutionJournal,
    TestnetPairCoordinator,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder, OrderReceipt
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.native_identity import NativeIdentity, NativeState

MAX_ROWS = 10000
MAX_BYTES = 4 * 1024 * 1024
MAX_SECONDS = 5
TABLES = ("testnet_pair_attempts", "testnet_order_intents", "testnet_spot_exit_recoveries", "testnet_exit_recoveries")


@contextmanager
def _snapshot_source(path: Path, connection: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
    if connection is not None:
        try:
            yield connection
        finally:
            connection.set_progress_handler(None, 0)
    else:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0)) as db:
            db.execute("BEGIN")
            yield db


def execution_snapshot(path: Path, attempt_id: str, *, connection: sqlite3.Connection | None = None) -> dict[str, Any]:
    started = time.monotonic()
    with _snapshot_source(path, connection) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.set_progress_handler(lambda: int(time.monotonic() - started > MAX_SECONDS), 1000)
        attempt = db.execute("SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if attempt is None or attempt["state"] not in ("CLOSED", "FLATTENED"):
            raise ValueError("original terminal attempt required")
        tables = {}
        count = 0
        used = 0
        for table in TABLES:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                if table in TABLES[:2]:
                    raise ValueError("execution source table absent")
                continue
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
            if any(not name.replace('_', '').isalnum() or not name.isascii() for name in columns):
                raise ValueError("invalid execution schema")
            size = ' + '.join(f'COALESCE(LENGTH(CAST("{name}" AS BLOB)),0)' for name in columns)
            totals = db.execute(f"SELECT COUNT(*),COALESCE(SUM({size}),0) FROM {table}").fetchone()
            count += totals[0]
            used += totals[1]
            if count > MAX_ROWS or used > MAX_BYTES or time.monotonic() - started > MAX_SECONDS:
                raise ValueError("execution snapshot bounds exceeded")
            rows = [dict(row) for row in db.execute(f"SELECT * FROM {table}")]
            tables[table] = sorted(rows, key=exact_json)
        intents = sorted([row for row in tables["testnet_order_intents"] if row['attempt_id'] == attempt_id], key=lambda row: row['client_order_id'])
        if not intents or len({row['client_order_id'] for row in intents}) != len(intents):
            raise ValueError("unambiguous original terminal intents required")
        plan = json.loads(attempt['plan_json'])
        if plan['attempt_id'] != attempt_id or hashlib.sha256(attempt['plan_json'].encode()).hexdigest() != attempt['plan_sha256']:
            raise ValueError("original plan identity differs")
        return {"schema_version": 1, "attempt_id": attempt_id, "state": attempt['state'],
            "attempt_sha256": hashlib.sha256(exact_json(dict(attempt)).encode()).hexdigest(),
            "intents_sha256": hashlib.sha256(exact_json(intents).encode()).hexdigest(),
            "execution_sha256": hashlib.sha256(exact_json(tables).encode()).hexdigest(),
            "rows": count, "bytes": used}


class _ReadOnlyJournal(SQLitePairExecutionJournal):
    @contextmanager
    def _connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        if create:
            raise ValueError("restart capture cannot initialize a journal")
        with closing(sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            yield db


class _BlockedGateway:
    def __init__(self, market: ExecutionMarket) -> None:
        self.market = market
        self.calls = 0

    def place_ioc_limit_order(self, order: IocLimitOrder) -> OrderReceipt:
        self.calls += 1
        raise ValueError("restart capture gateway disabled")

    def get_order(self, symbol: str, client_order_id: str) -> OrderReceipt:
        self.calls += 1
        raise ValueError("restart capture gateway disabled")


def replay_terminal(path: Path, attempt_id: str) -> tuple[PairExecutionResult, int]:
    journal = _ReadOnlyJournal(path)
    # Snapshot checks the global row/byte limit before the original plan parser.
    execution_snapshot(path, attempt_id)
    plans = [plan for plan, _ in journal.plans() if plan.attempt_id == attempt_id]
    if len(plans) != 1:
        raise ValueError("one original terminal plan required")
    spot, perp = _BlockedGateway(ExecutionMarket.SPOT), _BlockedGateway(ExecutionMarket.USDTM)
    original = journal.get(attempt_id)
    result = TestnetPairCoordinator(spot, perp, journal).reconcile(plans[0])
    calls = spot.calls + perp.calls
    if calls or result != original:
        raise ValueError("original terminal recovery made calls or changed result")
    return result, calls


def derive_restart_event(events: tuple[dict[str, Any], ...], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the original session and one actual read-only process restart."""
    begins = [index for index, event in enumerate(events) if event['kind'] == 'SESSION_BEGIN']
    ends = [index for index, event in enumerate(events) if event['kind'] == 'SESSION_END']
    if len(begins) != 2 or len(ends) != 2 or begins[0] != 0 or ends[-1] != len(events)-1 or begins[1] != ends[0]+1:
        raise ValueError('one original session and one restart session required')
    original_begin, original_end = events[0], events[ends[0]]
    restarted = events[begins[1]:]
    if [event['kind'] for event in restarted] != ['SESSION_BEGIN', 'TERMINAL_REPLAY_BEGIN', 'TERMINAL_REPLAY_END', 'SESSION_END']:
        raise ValueError('restart must capture the bounded original replay without requests')
    start, before, after, end = restarted
    old, new = NativeIdentity.parse(original_begin['process_identity']), NativeIdentity.parse(start['process_identity'])
    def date(value: str) -> datetime:
        return utc_timestamp(datetime.fromisoformat(value))

    if (old == new or NativeIdentity.parse(original_end['process_identity']) != old
            or NativeIdentity.parse(end['process_identity']) != new
            or NativeIdentity.parse(before['previous_process']) != old
            or not old.created_at <= date(original_begin['at']) <= date(original_end['at']) <= new.created_at <= date(start['at'])
            or start.get('phase') != 'TERMINAL_RESTART'
            or before['previous_observation'] not in (NativeState.EXITED, NativeState.MISSING, NativeState.REPLACED)
            or original_end['capture_complete'] is not True or end['capture_complete'] is not True
            or before['before'] != after['after'] or before['before'] != snapshot
            or original_end['terminal_snapshot'] != snapshot or end['terminal_snapshot'] != snapshot
            or type(after['request_call_count']) is not int or after['request_call_count'] != 0
            or after['result_state'] != snapshot['state']):
        raise ValueError('native restart identity, original snapshot or no-call evidence differs')
    return {'kind': 'TERMINAL_RESTART_NO_NEW_INTENTS', 'before_intents_sha256': snapshot['intents_sha256'],
        'after_intents_sha256': snapshot['intents_sha256'], 'entry_post_count': 0,
        'source_begin_sequence': before['sequence'], 'source_end_sequence': after['sequence'],
        'before_execution_sha256': snapshot['execution_sha256'], 'after_execution_sha256': snapshot['execution_sha256'],
        'previous_process': old.payload(), 'restarted_process': new.payload()}
