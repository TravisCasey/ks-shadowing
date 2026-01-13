/*
 * ETDRK4 integrator for the Kuramoto-Sivashinsky equation in Fourier space.
 * The state is represented using N/2-1 complex modes, stored as N-2 real
 * coefficients in interleaved real/imaginary format.
 *
 * Build requirements: C++17, Eigen >= 3.3, FFTW3
 */

#ifndef KSINT_HPP
#define KSINT_HPP

#include <Eigen/Dense>
#include <complex>
#include <fftw3.h>

using Complex = std::complex<double>;

using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::ArrayXXd;
using Eigen::Map;

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
 *   KS ks(32, 0.25, 22.0);  // N=32 modes, dt=0.25, domain=22
 *   ArrayXd a0 = ...;        // Initial condition (30 coefficients)
 *   ArrayXXd trajectory = ks.intg(a0, 1000, 10);  // 1000 steps, save every 10
 */
class KS {
public:
  // N: number of Fourier modes (must be even, typically 32)
  // h: time step
  // d: spatial domain size (default 22)
  explicit KS(int N = 32, double h = 0.25, double d = 22.0);
  ~KS();

  KS(const KS &) = delete;
  KS &operator=(const KS &) = delete;
  KS(KS &&) = delete;
  KS &operator=(KS &&) = delete;

  // Integrate forward in time.
  // a0: initial condition (N-2,), interleaved Fourier coefficients
  // nstp: total integration steps
  // np: save interval (store state every np steps)
  // Returns: trajectory (N-2, nstp/np + 1), column 0 is initial condition
  ArrayXXd intg(const ArrayXd &a0, size_t nstp, size_t np = 1);

  const int N;
  const double d;
  const double h;

private:
  // ETDRK4 coefficients
  ArrayXd K;          // Wavenumbers
  ArrayXd L;          // Linear operator: k^2 - k^4
  ArrayXd E, E2;      // Exponential factors
  ArrayXd Q;          // Nonlinear coefficient
  ArrayXd f1, f2, f3; // ETDRK4 coefficients
  ArrayXcd G;         // Nonlinear prefactor: i*k*N/2

  // FFTW workspace
  struct FFTWorkspace {
    fftw_plan forward_plan = nullptr;
    fftw_plan inverse_plan = nullptr;
    double *real_buffer = nullptr;
    fftw_complex *complex_buffer = nullptr;
    fftw_complex *nonlinear_buffer = nullptr;
    Map<ArrayXXd> real_view{nullptr, 0, 0};
    Map<ArrayXXcd> complex_view{nullptr, 0, 0};
    Map<ArrayXXcd> nonlinear_view{nullptr, 0, 0};
  };

  FFTWorkspace fft_v_, fft_a_, fft_b_, fft_c_;

  void initializeCoefficients();
  void evaluateNonlinear(FFTWorkspace &ws);
  void initializeFFT(FFTWorkspace &ws);
  void freeFFT(FFTWorkspace &ws);
  void fft(FFTWorkspace &ws);
  void ifft(FFTWorkspace &ws);
  ArrayXXd complexToReal(const ArrayXXcd &v);
  ArrayXXcd realToComplex(const ArrayXXd &v);

  static constexpr int CONTOUR_POINTS = 16;
};

#endif // KSINT_HPP
