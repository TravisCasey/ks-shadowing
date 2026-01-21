/*
 * C interface for Python ctypes binding to Hera Wasserstein distance.
 *
 * Exposes functions for computing W_2 distance between persistence diagrams.
 * Diagrams are passed as flat arrays of (birth, death) pairs in row-major
 * order.
 */

#include <cstdint>
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

/**
 * Compute Wasserstein-2 distance matrix between two sets of persistence
 * diagrams.
 *
 * Diagrams are flattened into a single array, with an offset array indicating
 * where each diagram starts.
 *
 * Parameters:
 *   traj_points  - Flattened (birth, death) pairs for all trajectory diagrams
 *   traj_offsets - Start index of each diagram (length num_traj + 1)
 *   num_traj     - Number of trajectory diagrams (I)
 *   rpo_points   - Flattened (birth, death) pairs for all RPO diagrams
 *   rpo_offsets  - Start index of each diagram (length num_rpo + 1)
 *   num_rpo      - Number of RPO diagrams (J)
 *   delta        - Relative error tolerance for (1+delta)-approximation
 *   out          - Output array of size I * J (row-major: out[i*J + j] =
 * dist(i,j))
 */
void wasserstein_matrix_c(const double *traj_points,
                          const int64_t *traj_offsets, int64_t num_traj,
                          const double *rpo_points, const int64_t *rpo_offsets,
                          int64_t num_rpo, double delta, double *out) {
  // Configure for W_2 distance with given tolerance
  hera::AuctionParams<double> params;
  params.wasserstein_power = 2.0;
  params.delta = delta;
  params.internal_p = hera::get_infinity<double>();

  // Pre-convert all trajectory diagrams (reused for every RPO column)
  std::vector<std::vector<std::pair<double, double>>> traj_dgms(num_traj);
  for (int64_t i = 0; i < num_traj; ++i) {
    int64_t start = traj_offsets[i];
    int64_t end = traj_offsets[i + 1];
    traj_dgms[i].reserve(end - start);
    for (int64_t k = start; k < end; ++k) {
      traj_dgms[i].emplace_back(traj_points[2 * k], traj_points[2 * k + 1]);
    }
  }

  // Reusable RPO diagram vector
  std::vector<std::pair<double, double>> rpo_dgm;

  for (int64_t j = 0; j < num_rpo; ++j) {
    int64_t start = rpo_offsets[j];
    int64_t end = rpo_offsets[j + 1];

    rpo_dgm.clear();
    rpo_dgm.reserve(end - start);
    for (int64_t k = start; k < end; ++k) {
      rpo_dgm.emplace_back(rpo_points[2 * k], rpo_points[2 * k + 1]);
    }

    // Compute distances for this RPO against all trajectory diagrams
    for (int64_t i = 0; i < num_traj; ++i) {
      out[i * num_rpo + j] =
          hera::wasserstein_dist(traj_dgms[i], rpo_dgm, params);
    }
  }
}

} // extern "C"
