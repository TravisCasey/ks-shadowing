/*
 * C interface for Python ctypes binding to union-find (disjoint set).
 *
 * Provides a batch API: takes all edge pairs at once, performs union-find
 * with path compression and union by rank, and returns component labels.
 * This avoids per-call ctypes overhead on millions of union/find operations.
 */

#include <cstdint>
#include <numeric>
#include <vector>

namespace {

class UnionFind {
public:
  explicit UnionFind(int32_t n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int32_t find(int32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path splitting
      x = parent[x];
    }
    return x;
  }

  void unite(int32_t x, int32_t y) {
    int32_t rx = find(x);
    int32_t ry = find(y);
    if (rx == ry)
      return;
    if (rank[rx] < rank[ry]) {
      int32_t tmp = rx;
      rx = ry;
      ry = tmp;
    }
    parent[ry] = rx;
    if (rank[rx] == rank[ry])
      ++rank[rx];
  }

private:
  std::vector<int32_t> parent;
  std::vector<int32_t> rank;
};

} // namespace

extern "C" {

/**
 * Perform union-find on a set of edges and write component labels.
 *
 * Creates a disjoint set of `n` elements (0 to n-1), unions all given
 * edge pairs, then writes the component root for each element to `out`.
 *
 * Parameters:
 *   n         - Number of elements
 *   edges_a   - First element of each edge pair
 *   edges_b   - Second element of each edge pair
 *   num_edges - Number of edge pairs
 *   out       - Output array of size n, filled with component root per element
 */
void connected_components_c(int32_t n, const int32_t *edges_a,
                            const int32_t *edges_b, int64_t num_edges,
                            int32_t *out) {
  UnionFind uf(n);

  for (int64_t i = 0; i < num_edges; ++i) {
    uf.unite(edges_a[i], edges_b[i]);
  }

  for (int32_t i = 0; i < n; ++i) {
    out[i] = uf.find(i);
  }
}

} // extern "C"
