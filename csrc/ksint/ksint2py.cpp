/*
 * C interface for Python ctypes binding to the KS integrator.
 * Exports ksf() for full state-space integration; N=32 is fixed at compile
 * time.
 */

#include "ksint.hpp"
#include <Eigen/Dense>
#include <cstring>

constexpr int N = 32;

extern "C" {

/*
 * Integrate KS equation in full state space. Returns 0 on success, 1 on
 * any internal failure (e.g., FFTW allocation failure). Python-side
 * validates inputs before calling.
 *
 * Parameters:
 *   out_trajectory - Pre-allocated output array, shape (nstp/np + 1, 30)
 *   initial_state  - Initial condition, shape (30,)
 *   domain_size    - Spatial domain size (typically 22.0)
 *   time_step      - Integration time step
 *   num_steps      - Total number of integration steps
 *   save_interval  - Save state every save_interval steps
 */
int ksf(double *out_trajectory, const double *initial_state, double domain_size,
        double time_step, std::size_t num_steps, std::size_t save_interval) {
  try {
    ksint::KS integrator(N, time_step, domain_size);

    Eigen::Map<const Eigen::ArrayXd> a0(initial_state, N - 2);
    Eigen::ArrayXXd result = integrator.intg(a0, num_steps, save_interval);

    std::memcpy(out_trajectory, result.data(), result.size() * sizeof(double));
    return 0;
  } catch (...) {
    return 1;
  }
}

} // extern "C"
