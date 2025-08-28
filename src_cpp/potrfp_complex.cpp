
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <numeric>

template <device::Precision prec>
inline void complex_imax_dispatcher(cudaStream_t stream, int32_t N, void* X, void* C, int32_t ldc, int32_t* piv, void* diag) {
  if constexpr(prec == device::Precision::FP64)
    internal::Cholesky::imax_cf64(stream, N, (double*)X, (std::complex<double>*)C, ldc, piv, (double*)diag);
  else if constexpr(prec == device::Precision::FP32)
    internal::Cholesky::imax_cf32(stream, N, (float*)X, (std::complex<float>*)C, ldc, piv, (float*)diag);
  else if constexpr(prec == device::Precision::FP128_DD)
    internal::Cholesky::imax_cf128_dd(stream, N, (double2*)X, (complex_double2*)C, ldc, piv, (double2*)diag);
  else if constexpr(prec == device::Precision::FP128_QF)
    internal::Cholesky::imax_cf128_qf(stream, N, (float4*)X, (complex_float4*)C, ldc, piv, (float4*)diag);
}

template <device::Precision prec>
inline void complex_swap_cols_dispatcher(cudaStream_t stream, int32_t i, int32_t j, int32_t N, void* A, int32_t lda) {
  if constexpr(prec == device::Precision::FP64)
    internal::Cholesky::swap_cols_cf64(stream, i, j, N, (std::complex<double>*)A, lda);
  else if constexpr(prec == device::Precision::FP32)
    internal::Cholesky::swap_cols_cf32(stream, i, j, N, (std::complex<float>*)A, lda);
  else if constexpr(prec == device::Precision::FP128_DD)
    internal::Cholesky::swap_cols_cf128_dd(stream, i, j, N, (complex_double2*)A, lda);
  else if constexpr(prec == device::Precision::FP128_QF)
    internal::Cholesky::swap_cols_cf128_qf(stream, i, j, N, (complex_float4*)A, lda);
}

template <device::Precision prec, class real_t>
inline void rsqrt_real(real_t& f, real_t& rsq) {
  if constexpr(prec == device::Precision::FP64)
  { f = std::sqrt(f); rsq = 1. / f; }
  else if constexpr(prec == device::Precision::FP32)
  { f = std::sqrt(f); rsq = 1.f / f; }
  else if constexpr(prec == device::Precision::FP128_DD)
  { rsq = device::dd::frsqrt(f); f = device::dd::mul(rsq, f); }
  else if constexpr(prec == device::Precision::FP128_QF)
  { rsq = device::qf::frsqrt(f); f = device::qf::mul(rsq, f); }
}

template <device::Precision prec, class real_t>
inline void complex_gemv_dispatcher(cudaStream_t stream, real_t* scale, int32_t M, int32_t N, const void* A, int32_t lda, void* B, real_t* D) {
  if constexpr(prec == device::Precision::FP64)
    internal::Cholesky::gemv_scal_cf64(stream, scale, M, N, (const std::complex<double>*)A, lda, (std::complex<double>*)B, D);
  else if constexpr(prec == device::Precision::FP32)
    internal::Cholesky::gemv_scal_cf32(stream, scale, M, N, (const std::complex<float>*)A, lda, (std::complex<float>*)B, D);
  else if constexpr(prec == device::Precision::FP128_DD)
    internal::Cholesky::gemv_scal_cf128_dd(stream, scale, M, N, (const complex_double2*)A, lda, (complex_double2*)B, D);
  else if constexpr(prec == device::Precision::FP128_QF)
    internal::Cholesky::gemv_scal_cf128_qf(stream, scale, M, N, (const complex_float4*)A, lda, (complex_float4*)B, D);
}

template <device::Precision prec, class real_t>
inline double conv_f64(real_t r) {
  if constexpr(prec == device::Precision::FP64) return r;
  else if constexpr(prec == device::Precision::FP32) return double(r);
  else if constexpr(prec == device::Precision::FP128_DD) return r.x;
  else if constexpr(prec == device::Precision::FP128_QF) return double(r.x) + double(r.y);
}

template <device::Precision prec, class real_t, class complex_t>
inline int32_t complex_potrfp(cudaStream_t stream, double epi, int32_t start, int32_t end, int32_t N, complex_t* A, int32_t lda, int32_t* jpiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &jpiv[algnN + 8];
  real_t* scale = (real_t*)(&jpiv[algnN]), *diag = (real_t*)(&A[uint64_t(N) * uint64_t(lda)]);

  if (0 < start) { // restarts
    cudaMemcpyAsync(scale, A, sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    double diag_f64 = conv_f64<prec>(scale[0]);
    epi = epi * diag_f64;
    if (!std::isnormal(diag_f64))
      return start;
  }
  else { // initialize
    std::iota(jpiv, &jpiv[N], 1);
    cudaMemcpy2DAsync(diag, sizeof(real_t), A, sizeof(complex_t) * uint64_t(lda + 1), sizeof(real_t), N, cudaMemcpyDeviceToDevice, stream);
    complex_imax_dispatcher<prec>(stream, N, diag, A, lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    int32_t j = *pivot_i;
    if (0 < j) {
      std::iter_swap(&jpiv[0], &jpiv[j]);
      complex_swap_cols_dispatcher<prec>(stream, 0, j, N, A, lda);
    }

    rsqrt_real<prec>(scale[0], scale[1]);
    double diag_f64 = conv_f64<prec>(scale[0]);
    complex_gemv_dispatcher<prec>(stream, scale, N, 0, A, lda, A, diag);

    epi = epi * diag_f64;
    if (!(std::isnormal(diag_f64) && epi <= diag_f64 && 0 <= j))
      return 0;
    start = 1;
  }

  for (int32_t i = start; i < end; ++i) {
    uint64_t A_diag = uint64_t(i) * uint64_t(lda + 1);
    uint64_t A_col = uint64_t(i) * uint64_t(lda);
    complex_imax_dispatcher<prec>(stream, N - i, &diag[i], &A[A_diag], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&jpiv[i], &jpiv[j]);
      complex_swap_cols_dispatcher<prec>(stream, i, j, N, A, lda);
    }

    rsqrt_real<prec>(scale[0], scale[1]);
    double diag_f64 = conv_f64<prec>(scale[0]);
    complex_gemv_dispatcher<prec>(stream, scale, N - i, i, &A[A_col], lda, &A[A_diag], &diag[i]);

    if (!(std::isnormal(diag_f64) && epi <= diag_f64 && 0 <= *pivot_i))
      end = i;
  }
  return end;
}

int32_t device::Cholesky::cpotrfp(cudaStream_t stream, double epi, int32_t start, int32_t end, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv) {
  epi = std::min(1., std::max(0., epi));
  start = std::min(N, std::max(0, start));
  end = std::min(N, std::max(start, end));

  if (start < end)
    switch (precA) {
      case Precision::FP64:
        return complex_potrfp<Precision::FP64, double, std::complex<double>>(stream, epi, start, end, N, (std::complex<double>*)A, lda, jpiv);
      case Precision::FP32:
        return complex_potrfp<Precision::FP32, float, std::complex<float>>(stream, epi, start, end, N, (std::complex<float>*)A, lda, jpiv);
      case Precision::FP128_DD:
        return complex_potrfp<Precision::FP128_DD, double2, complex_double2>(stream, epi, start, end, N, (complex_double2*)A, lda, jpiv);
      case Precision::FP128_QF:
        return complex_potrfp<Precision::FP128_QF, float4, complex_float4>(stream, epi, start, end, N, (complex_float4*)A, lda, jpiv);
    }
  return -1;
}
