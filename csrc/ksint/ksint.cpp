/*
 * Implementation of the ETDRK4 integrator for the Kuramoto-Sivashinsky
 * equation.
 */

#include "ksint.hpp"
#include <numbers>

namespace ksint {

using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::ArrayXXd;
using Eigen::Map;

KS::KS(int N, double h, double d) : N(N), d(d), h(h) {
  initializeCoefficients();

  fft_v_ = makeWorkspace(N);
  fft_a_ = makeWorkspace(N);
  fft_b_ = makeWorkspace(N);
  fft_c_ = makeWorkspace(N);
}

void KS::initializeCoefficients() {
  constexpr double pi = std::numbers::pi_v<double>;

  // Wavenumbers: k = 2*pi/d * [0, 1, 2, ..., N/2]
  K = ArrayXd::LinSpaced((N / 2) + 1, 0.0, N / 2.0) * 2 * pi / d;
  K(N / 2) = 0; // Zero out Nyquist mode

  L = K * K - K * K * K * K;
  E = (h * L).exp();
  E2 = (h / 2 * L).exp();

  // ETDRK4 coefficients via contour integral approximation
  // Uses M points on a circle in the complex plane to avoid numerical
  // instability at small L values.
  const int M = CONTOUR_POINTS;
  ArrayXd tmp = ArrayXd::LinSpaced(M, 1, M);
  ArrayXXcd r = ((tmp - 0.5) / M * Complex(0, pi)).exp().transpose();

  ArrayXXcd Lc = ArrayXXcd::Zero((N / 2) + 1, 1);
  Lc.real() = L;

  ArrayXXcd LR = h * Lc.replicate(1, M) + r.replicate((N / 2) + 1, 1);
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

ArrayXXd KS::intg(const ArrayXd &a0, std::size_t nstp, std::size_t np) {
  fft_v_.complex_view() = realToComplex(a0);

  ArrayXXd trajectory(N - 2, (nstp / np) + 1);
  trajectory.col(0) = a0;

  for (std::size_t i = 1; i <= nstp; i++) {
    // ETDRK4 stages
    evaluateNonlinear(fft_v_);
    fft_a_.complex_view() =
        E2 * fft_v_.complex_view() + Q * fft_v_.nonlinear_view();

    evaluateNonlinear(fft_a_);
    fft_b_.complex_view() =
        E2 * fft_v_.complex_view() + Q * fft_a_.nonlinear_view();

    evaluateNonlinear(fft_b_);
    fft_c_.complex_view() =
        E2 * fft_a_.complex_view() +
        Q * (2.0 * fft_b_.nonlinear_view() - fft_v_.nonlinear_view());

    evaluateNonlinear(fft_c_);

    // Combine stages
    fft_v_.complex_view() =
        E * fft_v_.complex_view() + f1 * fft_v_.nonlinear_view() +
        2.0 * f2 * (fft_a_.nonlinear_view() + fft_b_.nonlinear_view()) +
        f3 * fft_c_.nonlinear_view();

    if (i % np == 0) {
      trajectory.col(static_cast<Eigen::Index>(i / np)) =
          complexToReal(fft_v_.complex_view());
    }
  }

  return trajectory;
}

void KS::evaluateNonlinear(FFTWorkspace &ws) {
  // Transform to physical space
  ifft(ws);

  // Compute u^2 in physical space
  auto rv = ws.real_view();
  rv = rv * rv;

  // Transform back and apply derivative operator
  fft(ws);
  ws.nonlinear_view() *= G; // i*k (derivative) and normalization factor
}

KS::FFTWorkspace KS::makeWorkspace(int N) {
  FFTWorkspace ws;
  ws.N = N;

  ws.real_buffer = FFTWBuffer<double>{
      static_cast<double *>(fftw_malloc(sizeof(double) * N))};
  ws.complex_buffer = FFTWBuffer<fftw_complex>{static_cast<fftw_complex *>(
      fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)))};
  ws.nonlinear_buffer = FFTWBuffer<fftw_complex>{static_cast<fftw_complex *>(
      fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1)))};

  ws.forward_plan = FFTWPlan{fftw_plan_dft_r2c_1d(
      N, ws.real_buffer.get(), ws.nonlinear_buffer.get(), FFTW_MEASURE)};
  ws.inverse_plan = FFTWPlan{
      fftw_plan_dft_c2r_1d(N, ws.complex_buffer.get(), ws.real_buffer.get(),
                           FFTW_MEASURE | FFTW_PRESERVE_INPUT)};

  return ws;
}

void KS::fft(FFTWorkspace &ws) {
  fftw_execute(ws.forward_plan.get());
}

void KS::ifft(FFTWorkspace &ws) const {
  fftw_execute(ws.inverse_plan.get());
  auto rv = ws.real_view();
  rv /= N; // FFTW doesn't normalize
}

// Complex (N/2+1, M) -> Real interleaved (N-2, M)
// Extracts modes 1 to N/2-1, reinterprets as real pairs
ArrayXXd KS::complexToReal(const ArrayXXcd &v) {
  Eigen::Index rows = v.rows();
  Eigen::Index cols = v.cols();

  // Extract middle rows (skip mode 0 and Nyquist)
  ArrayXXcd middle = v.middleRows(1, rows - 2);

  // Reinterpret complex as interleaved real
  return Map<ArrayXXd>(reinterpret_cast<double *>(middle.data()),
                       2 * (rows - 2), cols);
}

// Real interleaved (N-2, M) -> Complex (N/2+1, M)
// Adds zero padding for mode 0 and Nyquist
ArrayXXcd KS::realToComplex(const ArrayXXd &v) {
  Eigen::Index rows = v.rows();
  Eigen::Index cols = v.cols();

  ArrayXXcd result = ArrayXXcd::Zero((rows / 2) + 2, cols);
  result.middleRows(1, rows / 2) = Map<const ArrayXXcd>(
      reinterpret_cast<const Complex *>(v.data()), rows / 2, cols);

  return result;
}

} // namespace ksint
