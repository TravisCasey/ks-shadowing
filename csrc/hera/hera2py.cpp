/*
 * C interface for Python binding to Hera for Wasserstein distance computations.
 *
 * Exposes batched W_2 distance computations between persistence diagrams.
 * Diagrams are passed as flat arrays of (birth, death) pairs in row-major
 * order.
 */

#include <cstdint>
#include <hera/wasserstein.h>
#include <span>
#include <utility>
#include <vector>

namespace {

std::vector<std::pair<double, double>>
pair_dgm_from_flat(std::span<const double> flat) {
  std::vector<std::pair<double, double>> out;
  out.reserve(flat.size() / 2);
  for (std::size_t k = 0; k + 1 < flat.size(); k += 2) {
    out.emplace_back(flat[k], flat[k + 1]);
  }
  return out;
}

} // namespace

extern "C" {

/*
 * Compute W_2 distances from each diagram in dgms_a to dgm_b.
 *
 * Returns 0 on success, 1 on any internal failure. Python-side validates
 * inputs before calling.
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
int wasserstein_column_c(const double *dgms_a, const int64_t *offsets_a,
                         int64_t num_a, const double *dgm_b, int64_t length_b,
                         double delta, double *out) {
  try {
    // Configure for W_2 distance with given tolerance
    hera::AuctionParams<double> params;
    params.wasserstein_power = 2.0;
    params.delta = delta;
    params.internal_p = hera::get_infinity<double>();

    std::vector<std::vector<std::pair<double, double>>> pair_dgms_a;
    pair_dgms_a.reserve(num_a);
    for (int64_t i = 0; i < num_a; ++i) {
      int64_t start = offsets_a[i];
      int64_t end = offsets_a[i + 1];
      pair_dgms_a.push_back(pair_dgm_from_flat(std::span<const double>{
          dgms_a + (2 * start), static_cast<std::size_t>(2 * (end - start))}));
    }

    std::vector<std::pair<double, double>> pair_dgm_b = pair_dgm_from_flat(
        std::span<const double>{dgm_b, static_cast<std::size_t>(2 * length_b)});

    for (int64_t i = 0; i < num_a; ++i) {
      out[i] = hera::wasserstein_dist(pair_dgms_a[i], pair_dgm_b, params);
    }
    return 0;
  } catch (...) {
    return 1;
  }
}

} // extern "C"
