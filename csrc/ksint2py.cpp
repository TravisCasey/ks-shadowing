/*
 * C interface for Python ctypes binding to the KS integrator.
 *
 * Exports a single function:
 *   ksf - Full state space integration
 *
 * All arrays use row-major layout with dimensions:
 *   - State vectors: (N-2,) = (30,) for N=32
 *   - Trajectories: (n_points, N-2)
 *
 * The integrator uses N=32 Fourier modes, giving 30 real coefficients
 * (modes 1 through 15, each with real and imaginary parts).
 */

#include "ksint.hpp"
#include <cstring>

constexpr int N = 32;

extern "C" {

/*
 * Integrate KS equation in full state space.
 *
 * Parameters:
 *   out_trajectory - Pre-allocated output array, shape (nstp/np + 1, 30)
 *   initial_state  - Initial condition, shape (30,)
 *   domain_size    - Spatial domain size (typically 22.0)
 *   time_step      - Integration time step
 *   num_steps      - Total number of integration steps
 *   save_interval  - Save state every save_interval steps
 *
 * The output is written directly to out_trajectory in row-major order.
 */
void ksf(double *out_trajectory, double *initial_state, double domain_size,
         double time_step, int num_steps, int save_interval) {

  KS integrator(N, time_step, domain_size);

  Map<ArrayXd> a0(initial_state, N - 2);
  ArrayXXd result = integrator.intg(a0, num_steps, save_interval);

  std::memcpy(out_trajectory, result.data(), result.size() * sizeof(double));
}

} // extern "C"
