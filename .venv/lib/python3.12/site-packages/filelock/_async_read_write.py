"""Async wrapper around :class:`ReadWriteLock` for use with ``asyncio``."""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from ._read_write import ReadWriteLock

if TYPE_CHECKING:
    import os
    from collections.abc import AsyncGenerator, Callable
    from concurrent import futures
    from types import TracebackType


class AsyncAcquireReadWriteReturnProxy:
    """Context-aware object that releases the async read/write lock on exit."""

    def __init__(self, lock: AsyncReadWriteLock) -> None:
        self.lock = lock

    async def __aenter__(self) -> AsyncReadWriteLock:
        return self.lock

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.lock.release()


class AsyncReadWriteLock:
    """
    Async wrapper around :class:`ReadWriteLock` for use in ``asyncio`` applications.

    Because Python's :mod:`sqlite3` module has no async API, all blocking SQLite operations are dispatched to a thread
    pool via ``loop.run_in_executor()``. Reentrancy, upgrade/downgrade rules, and singleton behavior are delegated
    to the underlying :class:`ReadWriteLock`.

    :param lock_file: path to the SQLite database file used as the lock
    :param timeout: maximum wait time in seconds; ``-1`` means block indefinitely
    :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable
    :param is_singleton: if ``True``, reuse existing :class:`ReadWriteLock` instances for the same resolved path
    :param loop: event loop for ``run_in_executor``; ``None`` uses the running loop
    :param executor: executor for ``run_in_executor``; ``None`` uses the default executor

    .. versionadded:: 3.21.0

    """

    def __init__(  # noqa: PLR0913
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        *,
        blocking: bool = True,
        is_singleton: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: futures.Executor | None = None,
    ) -> None:
        self._lock = ReadWriteLock(lock_file, timeout, blocking=blocking, is_singleton=is_singleton)
        self._loop = loop
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1)

    @property
    def lock_file(self) -> str:
        """:returns: the path to the lock file."""
        return self._lock.lock_file

    @property
    def timeout(self) -> float:
        """:returns: the default timeout."""
        return self._lock.timeout

    @property
    def blocking(self) -> bool:
        """:returns: whether blocking is enabled by default."""
        return self._lock.blocking

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """:returns: the event loop (or ``None`` for the running loop)."""
        return self._loop

    @property
    def executor(self) -> futures.Executor | None:
        """:returns: the executor (or ``None`` for the default)."""
        return self._executor

    async def _run(self, func: Callable[..., object], *args: object, **kwargs: object) -> object:
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

    async def acquire_read(self, timeout: float = -1, *, blocking: bool = True) -> AsyncAcquireReadWriteReturnProxy:
        """
        Acquire a shared read lock.

        See :meth:`ReadWriteLock.acquire_read` for full semantics.

        :param timeout: maximum wait time in seconds; ``-1`` means block indefinitely
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable

        :returns: a proxy that can be used as an async context manager to release the lock

        :raises RuntimeError: if a write lock is already held on this instance
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds

        """
        await self._run(self._lock.acquire_read, timeout, blocking=blocking)
        return AsyncAcquireReadWriteReturnProxy(lock=self)

    async def acquire_write(self, timeout: float = -1, *, blocking: bool = True) -> AsyncAcquireReadWriteReturnProxy:
        """
        Acquire an exclusive write lock.

        See :meth:`ReadWriteLock.acquire_write` for full semantics.

        :param timeout: maximum wait time in seconds; ``-1`` means block indefinitely
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately when the lock is unavailable

        :returns: a proxy that can be used as an async context manager to release the lock

        :raises RuntimeError: if a read lock is already held, or a write lock is held by a different thread
        :raises Timeout: if the lock cannot be acquired within *timeout* seconds

        """
        await self._run(self._lock.acquire_write, timeout, blocking=blocking)
        return AsyncAcquireReadWriteReturnProxy(lock=self)

    async def release(self, *, force: bool = False) -> None:
        """
        Release one level of the current lock.

        See :meth:`ReadWriteLock.release` for full semantics.

        :param force: if ``True``, release the lock completely regardless of the current lock level

        :raises RuntimeError: if no lock is currently held and *force* is ``False``

        """
        await self._run(self._lock.release, force=force)

    @asynccontextmanager
    async def read_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> AsyncGenerator[None]:
        """
        Async context manager that acquires and releases a shared read lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        """
        if timeout is None:
            timeout = self._lock.timeout
        if blocking is None:
            blocking = self._lock.blocking
        await self.acquire_read(timeout, blocking=blocking)
        try:
            yield
        finally:
            await self.release()

    @asynccontextmanager
    async def write_lock(self, timeout: float | None = None, *, blocking: bool | None = None) -> AsyncGenerator[None]:
        """
        Async context manager that acquires and releases an exclusive write lock.

        Falls back to instance defaults for *timeout* and *blocking* when ``None``.

        :param timeout: maximum wait time in seconds, or ``None`` to use the instance default
        :param blocking: if ``False``, raise :class:`~filelock.Timeout` immediately; ``None`` uses the instance default

        """
        if timeout is None:
            timeout = self._lock.timeout
        if blocking is None:
            blocking = self._lock.blocking
        await self.acquire_write(timeout, blocking=blocking)
        try:
            yield
        finally:
            await self.release()

    async def close(self) -> None:
        """
        Release the lock (if held) and close the underlying SQLite connection.

        After calling this method, the lock instance is no longer usable.

        """
        await self._run(self._lock.close)
        if self._owns_executor:
            self._executor.shutdown(wait=False)


__all__ = [
    "AsyncAcquireReadWriteReturnProxy",
    "AsyncReadWriteLock",
]
