"""Publish an already durable temporary file; never retry its business work."""

import sys
from pathlib import Path
from time import sleep

MAX_REPLACE_ATTEMPTS = 10
REPLACE_WAIT_SECONDS = .01
_WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33})


def publish_state(source: Path, target: Path) -> None:
    """Retry only Windows sharing/access failures of the same atomic replace.

    Caller closes and fsyncs the temporary file, and owns the worker lock.
    The same bytes/claim are published at most once; no unlink fallback, rewrite,
    extra claim, market request or counter refund. Persistent failures propagate.
    At most ten replace calls and nine 10ms sleeps; this is not an OS call timeout
    or a guarantee against persistent ACL errors, slow disks or power loss.
    """
    for attempt in range(MAX_REPLACE_ATTEMPTS):
        try:
            source.replace(target)
            return
        except OSError as error:
            if (sys.platform != "win32" or getattr(error, "winerror", None) not in _WINDOWS_SHARING_ERRORS
                    or attempt == MAX_REPLACE_ATTEMPTS-1):
                raise
            sleep(REPLACE_WAIT_SECONDS)
