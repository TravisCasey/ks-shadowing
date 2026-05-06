/*
 * ETDRK4 integrator for the Kuramoto-Sivashinsky equation in Fourier space.
 * The state is represented using N/2-1 complex modes, stored as N-2 real
 * coefficients in interleaved real/imaginary format.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>
#include <memory>

namespace ksint {

using Complex = std::complex<double>;

struct FFTWBufferDeleter {
  void operator()(void *p) const noexcept { fftw_free(p); }
};

struct FFTWPlanDeleter {
  void operator()(fftw_plan p) const noexcept {
    if (p) {
      fftw_destroy_plan(p);
    }
  }
};

template <typename T>
using FFTWBuffer = std::unique_ptr<T[], FFTWBufferDeleter>;
using FFTWPlan = std::unique_ptr<fftw_plan_s, FFTWPlanDeleter>;

/*
 * ETDRK4 integrator for the Kuramoto-Sivashinsky equation.
 *
 * The integrator operates in Fourier space. The state vector contains N-2
 * real values representing modes 1 to N/2-1 in interleaved format:
 * [Re(a_1), Im(a_1), Re(a_2), Im(a_2), ...].
 *
 * Mode 0 (the mean) is always zero. Mode N/2 (the Nyquist mode) is excluded
 * from the state vector and its wavenumber is set to zero internally.
 *
 * Usage:
 *   ksint::KS ks(32, 0.25, 22.0);   // N=32 modes, dt=0.25, domain=22
 *   Eigen::ArrayXd a0 = ...;        // Initial condition (30 coefficients)
 *   Eigen::ArrayXXd traj = ks.intg(a0, 1000, 10);  // 1000 steps, save every 10
 */
class KS {
public:
  // N: number of Fourier modes (must be even, typically 32)
  // h: time step
  // d: spatial domain size (default 22)
  explicit KS(int N = 32, double h = 0.25, double d = 22.0);

  KS(const KS &) = delete;
  KS &operator=(const KS &) = delete;
  KS(KS &&) = delete;
  KS &operator=(KS &&) = delete;

  // Integrate forward in time.
  // a0:   initial condition (N-2,), interleaved Fourier coefficients
  // nstp: total integration steps
  // np:   save interval (store state every np steps)
  // Returns: trajectory (N-2, nstp/np + 1), column 0 is initial condition
  Eigen::ArrayXXd intg(const Eigen::ArrayXd &a0, std::size_t nstp,
                       std::size_t np = 1);

  const int N;
  const double d;
  const double h;

private:
  // ETDRK4 coefficients
  Eigen::ArrayXd K;          // Wavenumbers
  Eigen::ArrayXd L;          // Linear operator: k^2 - k^4
  Eigen::ArrayXd E, E2;      // Exponential factors
  Eigen::ArrayXd Q;          // Nonlinear coefficient
  Eigen::ArrayXd f1, f2, f3; // ETDRK4 coefficients
  Eigen::ArrayXcd G;         // Nonlinear prefactor: i*k*N/2

  // FFTW workspace: owns its buffers and plans via RAII. Eigen Map views
  // are constructed lazily on each access; constructing a Map is a pointer
  // + dim copy with no allocation.
  struct FFTWorkspace {
    FFTWBuffer<double> real_buffer;
    FFTWBuffer<fftw_complex> complex_buffer;
    FFTWBuffer<fftw_complex> nonlinear_buffer;
    FFTWPlan forward_plan;
    FFTWPlan inverse_plan;
    int N = 0;

    [[nodiscard]] Eigen::Map<Eigen::ArrayXXd> real_view() const {
      return Eigen::Map<Eigen::ArrayXXd>(real_buffer.get(), N, 1);
    }
    [[nodiscard]] Eigen::Map<Eigen::ArrayXXcd> complex_view() const {
      return Eigen::Map<Eigen::ArrayXXcd>(
          reinterpret_cast<Complex *>(complex_buffer.get()), (N / 2) + 1, 1);
    }
    [[nodiscard]] Eigen::Map<Eigen::ArrayXXcd> nonlinear_view() const {
      return Eigen::Map<Eigen::ArrayXXcd>(
          reinterpret_cast<Complex *>(nonlinear_buffer.get()), (N / 2) + 1, 1);
    }
  };

  FFTWorkspace fft_v_, fft_a_, fft_b_, fft_c_;

  void initializeCoefficients();
  static FFTWorkspace makeWorkspace(int N);
  void evaluateNonlinear(FFTWorkspace &ws);
  static void fft(FFTWorkspace &ws);
  void ifft(FFTWorkspace &ws) const;
  static Eigen::ArrayXXd complexToReal(const Eigen::ArrayXXcd &v);
  static Eigen::ArrayXXcd realToComplex(const Eigen::ArrayXXd &v);

  static constexpr int CONTOUR_POINTS = 16;
};

} // namespace ksint
