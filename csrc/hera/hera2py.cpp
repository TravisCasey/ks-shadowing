/*
 * C interface for Python ctypes binding to Hera Wasserstein distance.
 *
 * Exposes a single function for computing W_2 distance between persistence
 * diagrams. Diagrams are passed as flat arrays of (birth, death) pairs in
 * row-major order.
 */

#include <hera/wasserstein.h>
#include <utility>
#include <vector>

extern "C" {

/**
 * Compute Wasserstein-2 distance between two persistence diagrams.
 *
 * Parameters:
 *   dgm_a  - First diagram, flat array of (birth, death) pairs, shape (n_a * 2)
 *   n_a    - Number of points in first diagram
 *   dgm_b  - diagram two, flat array of (birth, death) pairs, shape (n_b * 2)
 *   n_b    - Number of points in second diagram delta  - Relative error
 *     tolerance for (1+delta)-approximation
 *
 * Returns:
 *   The Wasserstein-2 distance between the diagrams.
 */
double wasserstein_dist_c(const double *dgm_a, int n_a, const double *dgm_b,
                          int n_b, double delta) {
  // Convert flat arrays to vector of pairs
  std::vector<std::pair<double, double>> A;
  std::vector<std::pair<double, double>> B;
  A.reserve(n_a);
  B.reserve(n_b);

  for (int i = 0; i < n_a; ++i) {
    A.emplace_back(dgm_a[2 * i], dgm_a[2 * i + 1]);
  }
  for (int i = 0; i < n_b; ++i) {
    B.emplace_back(dgm_b[2 * i], dgm_b[2 * i + 1]);
  }

  // Configure for W_2 distance with given tolerance
  hera::AuctionParams<double> params;
  params.wasserstein_power = 2.0;
  params.delta = delta;
  params.internal_p = hera::get_infinity<double>();

  return hera::wasserstein_dist(A, B, params);
}

} // extern "C"
