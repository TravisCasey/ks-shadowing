"""Parallelism utilities."""

from multiprocessing import cpu_count


def _resolve_n_jobs(n_jobs: int) -> int:
    """Convert ``n_jobs`` parameter to actual worker count.

    Follows scikit-learn convention: 1 means sequential, -1 means all CPUs, and
    positive integers specify the exact worker count.

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
        return cpu_count()
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs}")
    return n_jobs
