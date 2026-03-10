"""Shared utility classes and functions."""

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


class _UnionFind:
    """Disjoint set data structure with path compression and union by rank.

    Efficiently tracks the inclusion of integers in disjoint sets. Supports
    two operations:

    - :meth:`find(x) <find>`: returns the representative element of the set
      containing ``x``.
    - :meth:`union(x, y) <union>`: merges the sets containing ``x`` and ``y``.

    Both operations run in amortized nearly constant time due to path
    compression and union by rank.

    Parameters
    ----------
    n : int
        Number of elements (indexed 0 to ``n - 1``), each initially in its
        own singleton set.
    """

    def __init__(self, n: int):
        """Initialize with ``n`` elements, each in its own disjoint set."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the representative of the set containing ``x``."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing ``x`` and ``y``."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
