"""Parallelism utilities."""

import multiprocessing as mp
import os
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray


def _forkserver_pool(n_workers: int) -> Pool:
    """Return a :class:`~multiprocessing.pool.Pool` backed by ``forkserver``.

    Forks workers from a clean single-threaded helper instead of the parent.
    numpy/scipy/fftw leave background threads in the parent that hold locks
    only the calling thread can release; the default fork start method copies
    those locks into the child without their threads, risking deadlock.
    """
    return mp.get_context("forkserver").Pool(n_workers)


def _resolve_n_jobs(n_jobs: int) -> int:
    """Convert ``n_jobs`` parameter to actual worker count.

    Follows scikit-learn convention: 1 means sequential, -1 means all CPUs, and
    positive integers specify the exact worker count. ``-1`` resolves to the
    process's CPU affinity mask where available (respecting container/cgroup
    CPU limits and taskset pinning), falling back to the system-wide CPU count
    on platforms without :func:`os.sched_getaffinity` (e.g. macOS, Windows).

    Parameters
    ----------
    n_jobs : int
        Desired parallelism. 1 for sequential, -1 for all CPUs.

    Returns
    -------
    int
        Resolved worker count (always >= 1).

    Raises
    ------
    ValueError
        If ``n_jobs`` is 0 or less than -1.
    """
    if n_jobs == 1:
        return 1
    if n_jobs == -1:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return cpu_count()
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs}")
    return n_jobs


@contextmanager
def _shared_memory_view(array: NDArray) -> Iterator[SharedMemory]:
    """Yield a :class:`~multiprocessing.shared_memory.SharedMemory` block
    populated with the contents of ``array``.

    On exit, the block is closed and unlinked.

    Raises
    ------
    ValueError
        If ``array`` is empty; zero-length shared memory is not permitted and
        callers should not be passing empty inputs.
    """
    if array.nbytes == 0:
        raise ValueError("cannot create shared-memory view of an empty array")

    shm = SharedMemory(create=True, size=array.nbytes)
    try:
        buffer_view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        buffer_view[:] = array
        yield shm
    finally:
        shm.close()
        shm.unlink()
