"""Local OS-owned nonblocking locks; no exchange or execution imports."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ControlError(RuntimeError):
    """Stable local lifecycle failure shared by CLI and imported controllers."""


class LockBusyError(RuntimeError):
    pass


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
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
            except OSError as error:
                raise LockBusyError("LOCAL_WORKER_LOCK_HELD") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise LockBusyError("LOCAL_WORKER_LOCK_HELD") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def lock_is_held(path: Path) -> bool:
    try:
        with file_lock(path):
            return False
    except LockBusyError:
        return True
