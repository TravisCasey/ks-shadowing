/*
 * Implementation of the ETDRK4 integrator for the Kuramoto-Sivashinsky
 * equation.
 */

#include "ksint.hpp"
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Construction / Destruction
// ============================================================================

KS::KS(int N, double h, double d) : N(N), h(h), d(d) {
  initializeCoefficients();

  initializeFFT(fft_v_);
  initializeFFT(fft_a_);
  initializeFFT(fft_b_);
  initializeFFT(fft_c_);
}

KS::~KS() {
  freeFFT(fft_v_);
  freeFFT(fft_a_);
  freeFFT(fft_b_);
  freeFFT(fft_c_);
}

// ============================================================================
// ETDRK4 Coefficient Initialization
// ============================================================================

void KS::initializeCoefficients() {
  // Wavenumbers: k = 2*pi/d * [0, 1, 2, ..., N/2]
  K = ArrayXd::LinSpaced(N / 2 + 1, 0.0, N / 2.0) * 2 * M_PI / d;
  K(N / 2) = 0; // Zero out Nyquist mode

  L = K * K - K * K * K * K;
  E = (h * L).exp();
  E2 = (h / 2 * L).exp();

  // ETDRK4 coefficients via contour integral approximation
  // Uses M points on a circle in the complex plane to avoid numerical
  // instability at small L values.
  const int M = CONTOUR_POINTS;
  ArrayXd tmp = ArrayXd::LinSpaced(M, 1, M);
  ArrayXXcd r = ((tmp - 0.5) / M * Complex(0, M_PI)).exp().transpose();

  ArrayXXcd Lc = ArrayXXcd::Zero(N / 2 + 1, 1);
  Lc.real() = L;

  ArrayXXcd LR = h * Lc.replicate(1, M) + r.replicate(N / 2 + 1, 1);
  ArrayXXcd LR2 = LR.square();
  ArrayXXcd LR3 = LR.cube();
  ArrayXXcd LRe = LR.exp();

  Q = h * (((LR / 2.0).exp() - 1) / LR).rowwise().mean().real();
  f1 = h * ((-4.0 - LR + LRe * (4.0 - 3.0 * LR + LR2)) / LR3)
               .rowwise()
               .mean()
               .real();
  f2 = h * ((2.0 + LR + LRe * (-2.0 + LR)) / LR3).rowwise().mean().real();
  f3 = h * ((-4.0 - 3.0 * LR - LR2 + LRe * (4.0 - LR)) / LR3)
               .rowwise()
               .mean()
               .real();

  G = 0.5 * Complex(0, 1) * K * N;
}

// ============================================================================
// Integration
// ============================================================================

ArrayXXd KS::intg(const ArrayXd &a0, size_t nstp, size_t np) {
  if (N - 2 != a0.rows()) {
    std::fprintf(stderr,
                 "KS::intg: initial condition has wrong dimension "
                 "(expected %d, got %ld)\n",
                 N - 2, a0.rows());
    std::exit(1);
  }

  fft_v_.complex_view = realToComplex(a0);

  ArrayXXd trajectory(N - 2, nstp / np + 1);
  trajectory.col(0) = a0;

  for (size_t i = 1; i <= nstp; i++) {
    // ETDRK4 stages
    evaluateNonlinear(fft_v_);
    fft_a_.complex_view = E2 * fft_v_.complex_view + Q * fft_v_.nonlinear_view;

    evaluateNonlinear(fft_a_);
    fft_b_.complex_view = E2 * fft_v_.complex_view + Q * fft_a_.nonlinear_view;

    evaluateNonlinear(fft_b_);
    fft_c_.complex_view =
        E2 * fft_a_.complex_view +
        Q * (2.0 * fft_b_.nonlinear_view - fft_v_.nonlinear_view);

    evaluateNonlinear(fft_c_);

    // Combine stages
    fft_v_.complex_view =
        E * fft_v_.complex_view + f1 * fft_v_.nonlinear_view +
        2.0 * f2 * (fft_a_.nonlinear_view + fft_b_.nonlinear_view) +
        f3 * fft_c_.nonlinear_view;

    if (i % np == 0) {
      trajectory.col(i / np) = complexToReal(fft_v_.complex_view);
    }
  }

  return trajectory;
}

// ============================================================================
// Nonlinear Term Evaluation
// ============================================================================

void KS::evaluateNonlinear(FFTWorkspace &ws) {
  // Transform to physical space
  ifft(ws);

  // Compute u^2 in physical space
  ws.real_view = ws.real_view * ws.real_view;

  // Transform back and apply derivative operator
  fft(ws);
  ws.nonlinear_view *= G; // Multiply by i*k (derivative) and normalization
}

// ============================================================================
// FFT Management
// ============================================================================

void KS::initializeFFT(FFTWorkspace &ws) {
  ws.real_buffer = (double *)fftw_malloc(sizeof(double) * N);
  ws.complex_buffer =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
  ws.nonlinear_buffer =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

  // Create Eigen Map views of the FFTW buffers
  new (&ws.real_view) Map<ArrayXXd>(ws.real_buffer, N, 1);
  new (&ws.complex_view)
      Map<ArrayXXcd>((Complex *)ws.complex_buffer, N / 2 + 1, 1);
  new (&ws.nonlinear_view)
      Map<ArrayXXcd>((Complex *)ws.nonlinear_buffer, N / 2 + 1, 1);

  ws.forward_plan = fftw_plan_dft_r2c_1d(N, ws.real_buffer, ws.nonlinear_buffer,
                                         FFTW_MEASURE);
  ws.inverse_plan = fftw_plan_dft_c2r_1d(N, ws.complex_buffer, ws.real_buffer,
                                         FFTW_MEASURE | FFTW_PRESERVE_INPUT);
}

void KS::freeFFT(FFTWorkspace &ws) {
  if (ws.forward_plan)
    fftw_destroy_plan(ws.forward_plan);
  if (ws.inverse_plan)
    fftw_destroy_plan(ws.inverse_plan);
  if (ws.complex_buffer)
    fftw_free(ws.complex_buffer);
  if (ws.real_buffer)
    fftw_free(ws.real_buffer);
  if (ws.nonlinear_buffer)
    fftw_free(ws.nonlinear_buffer);

  // Release Eigen maps
  new (&ws.complex_view) Map<ArrayXXcd>(nullptr, 0, 0);
  new (&ws.real_view) Map<ArrayXXd>(nullptr, 0, 0);
  new (&ws.nonlinear_view) Map<ArrayXXcd>(nullptr, 0, 0);
}

void KS::fft(FFTWorkspace &ws) { fftw_execute(ws.forward_plan); }

void KS::ifft(FFTWorkspace &ws) {
  fftw_execute(ws.inverse_plan);
  ws.real_view /= N; // FFTW doesn't normalize
}

// ============================================================================
// Format Conversion
// ============================================================================

// Complex (N/2+1, M) -> Real interleaved (N-2, M)
// Extracts modes 1 to N/2-1, reinterprets as real pairs
ArrayXXd KS::complexToReal(const ArrayXXcd &v) {
  int rows = v.rows();
  int cols = v.cols();

  // Extract middle rows (skip mode 0 and Nyquist)
  ArrayXXcd middle = v.middleRows(1, rows - 2);

  // Reinterpret complex as interleaved real
  return Map<ArrayXXd>((double *)middle.data(), 2 * (rows - 2), cols);
}

// Real interleaved (N-2, M) -> Complex (N/2+1, M)
// Adds zero padding for mode 0 and Nyquist
ArrayXXcd KS::realToComplex(const ArrayXXd &v) {
  int rows = v.rows();
  int cols = v.cols();

  if (rows % 2 != 0) {
    std::fprintf(stderr,
                 "KS::realToComplex: input must have even number of rows\n");
    std::exit(1);
  }

  ArrayXXcd result = ArrayXXcd::Zero(rows / 2 + 2, cols);
  result.middleRows(1, rows / 2) =
      Map<const ArrayXXcd>((const Complex *)v.data(), rows / 2, cols);

  return result;
}
