"""Tests for utility classes."""

from ks_shadowing.core.util import UnionFind


class TestUnionFind:
    """Tests for the UnionFind disjoint set data structure."""

    def test_initially_separate(self):
        """Each element starts in its own singleton set."""
        uf = UnionFind(5)
        for element in range(5):
            assert uf.find(element) == element

    def test_union_merges_sets(self):
        """Union merges two sets into one."""
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_is_transitive(self):
        """Chained unions result in a single set."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_disjoint_sets_remain_separate(self):
        """Sets that are not unioned remain disjoint."""
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) != uf.find(2)
