"""Bounded independent research IO, claiming only free executor slots.

No caches of market/Funding, financial math, retry policy or task-state rewrite.
Only the controlling thread claims tasks, reports progress and handles failures.
"""

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from crypto_arbitrage_bot.models.validation_task import ValidationTask


def process_task_batch(
    pending: tuple[ValidationTask, ...], *,
    claim: Callable[[ValidationTask], ValidationTask | None],
    execute: Callable[[ValidationTask], None],
    failure: Callable[[ValidationTask, Exception], None],
    stopped: Callable[[], bool],
    progress: Callable[[], None],
    workers: int = 4,
) -> int:
    """Return claimed count, including failures; no success/qualification claims.

    A task is never leased while waiting in an executor queue. Already-running
    bounded IO finishes on stop; unclaimed tasks remain untouched for recovery.
    BaseException propagates to the controller, leaving interrupted task leases.
    Progress/storage failure stops feeding but does not replay completed work.
    """
    if type(workers) is not int or not 1 <= workers <= 4 or len(pending) > 1000:
        raise ValueError("BOUNDED_RESEARCH_EXECUTOR_REQUIRED")
    remaining = iter(pending)
    active: dict[Future[None], ValidationTask] = {}
    claimed = 0
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="public-exit") as pool:
        while active or not exhausted:
            while len(active) < workers and not exhausted and not stopped():
                item = next(remaining, None)
                if item is None:
                    exhausted = True
                    break
                task = claim(item)
                if task is not None:
                    claimed += 1
                    active[pool.submit(execute, task)] = task
            if not active:
                break
            done, _ = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
            progress()
            for future in done:
                task = active.pop(future)
                try:
                    future.result()
                except Exception as error:
                    failure(task, error)
    return claimed
