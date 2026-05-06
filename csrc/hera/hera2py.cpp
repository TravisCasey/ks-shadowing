/*
 * C interface for Python binding to Hera for Wasserstein distance computations.
 *
 * Exposes batched W_2 distance computations between persistence diagrams.
 * Diagrams are passed as flat arrays of (birth, death) pairs in row-major
 * order.
 */

#include <cstdint>
#include <hera/wasserstein.h>
#include <utility>
#include <vector>

extern "C" {

/**
 * Compute Wasserstein-2 distance matrix from a set of persistence diagrams to
 * another diagram.
 *
 * The set of diagrams are flattened into a single array, with an offset array
 * indicating where each diagram starts.
 *
 * Parameters:
 *   dgms_a    - Flattened (birth, death) pairs for all A diagrams
 *   offsets_a - Start index of each A diagram (num_a + 1,)
 *   num_a     - Number of A diagrams
 *   dgm_b     - Flattened (birth, death) pairs for the B diagram
 *   length_b  - Number of (birth, death) pairs on dgm_b
 *   delta     - Relative error tolerance for (1+delta)-approximation
 *   out       - Output distances; out[i] = W_2(dgm_a_i, dgm_b)
 */
void wasserstein_column_c(const double *dgms_a, const int64_t *offsets_a,
                          int64_t num_a, const double *dgm_b, int64_t length_b,
                          double delta, double *out) {
  // Configure for W_2 distance with given tolerance
  hera::AuctionParams<double> params;
  params.wasserstein_power = 2.0;
  params.delta = delta;
  params.internal_p = hera::get_infinity<double>();

  // Pre-convert all A diagrams
  std::vector<std::vector<std::pair<double, double>>> pair_dgms_a(num_a);
  for (int64_t i = 0; i < num_a; ++i) {
    int64_t start = offsets_a[i];
    int64_t end = offsets_a[i + 1];
    pair_dgms_a[i].reserve(end - start);
    for (int64_t k = start; k < end; ++k) {
      pair_dgms_a[i].emplace_back(dgms_a[2 * k], dgms_a[2 * k + 1]);
    }
  }

  // Convert B diagram
  std::vector<std::pair<double, double>> pair_dgm_b;
  pair_dgm_b.reserve(length_b);
  for (int64_t k = 0; k < length_b; ++k) {
    pair_dgm_b.emplace_back(dgm_b[2 * k], dgm_b[2 * k + 1]);
  }

  // Compute distances
  for (int64_t i = 0; i < num_a; ++i) {
    out[i] = hera::wasserstein_dist(pair_dgms_a[i], pair_dgm_b, params);
  }
}

} // extern "C"
