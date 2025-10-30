
#include <hyacin.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

template <device::Precision prec, class real_t>
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const real_t* X, int32_t incx, real_t* D, real_t* diag_piv) {
  if constexpr(prec == device::Precision::FP64)
    internal::Cholesky::imax_f64(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == device::Precision::FP32)
    internal::Cholesky::imax_f32(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == device::Precision::FP128_DD)
    internal::Cholesky::imax_f128_dd(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == device::Precision::FP128_QF)
    internal::Cholesky::imax_f128_qf(stream, N, X, incx, D, diag_piv);
}

template <device::Precision prec, class real_t, class matrix_t>
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, real_t* scale, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda, real_t* D) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));

  if constexpr(COMPLEX && prec == device::Precision::FP64)
    internal::Cholesky::gemv_cublas_cf64(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(COMPLEX && prec == device::Precision::FP32)
    internal::Cholesky::gemv_cublas_cf32(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(!COMPLEX && prec == device::Precision::FP64)
    internal::Cholesky::gemv_cublas_f64(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(!COMPLEX && prec == device::Precision::FP32)
    internal::Cholesky::gemv_cublas_f32(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(COMPLEX && prec == device::Precision::FP128_DD)
    internal::Cholesky::gemv_scal_cf128_dd(stream, scale, j, M, N, A, lda, D);
  else if constexpr(COMPLEX && prec == device::Precision::FP128_QF)
    internal::Cholesky::gemv_scal_cf128_qf(stream, scale, j, M, N, A, lda, D);
  else if constexpr(!COMPLEX && prec == device::Precision::FP128_DD)
    internal::Cholesky::gemv_scal_f128_dd(stream, scale, j, M, N, A, lda, D);
  else if constexpr(!COMPLEX && prec == device::Precision::FP128_QF)
    internal::Cholesky::gemv_scal_f128_qf(stream, scale, j, M, N, A, lda, D);
}

template <device::Precision prec, class real_t>
inline double conv_f64(real_t r) {
  if constexpr(prec == device::Precision::FP64) return r;
  else if constexpr(prec == device::Precision::FP32) return double(r);
  else if constexpr(prec == device::Precision::FP128_DD) return r.x;
  else if constexpr(prec == device::Precision::FP128_QF) return double(r.x) + double(r.y);
}

template <device::Precision prec, class real_t, class matrix_t>
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t iters, int32_t N, matrix_t* A, int32_t lda, int32_t* jpiv, real_t* scale) {
  real_t* diag = (real_t*)(&A[int64_t(N) * int64_t(lda)]);
  imax_dispatcher<prec>(stream, N, (real_t*)A, sizeof(real_t) == sizeof(matrix_t) ? (lda + 1) : (2 * lda + 2), diag, scale);
  int32_t* pivot_i = (int32_t*)&scale[2];

  for (int32_t i = 0; i < iters; ++i) {
    int64_t A_col = int64_t(i) * int64_t(lda);
    int32_t j = (*pivot_i) - 1;
    double diag_f64 = conv_f64<prec>(scale[0]);
    epi = 0 < i ? epi : (epi * diag_f64);

    if (!(std::isnormal(diag_f64) && epi <= diag_f64 && 0 <= j)) return i;
      else if (0 < j) std::iter_swap(&jpiv[i], &jpiv[i + j]);
    gemv_dispatcher<prec>(stream, handle, scale, j, N - i, i, &A[A_col], lda, &diag[i]);
  }
  return iters;
}

void device::Cholesky::rpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work) {
  epi = std::min(1., std::max(0., std::abs(epi)));
  int32_t rank = std::min(N, std::max(0, *iters));
  
  switch (precA) {
    case Precision::FP64:
      *iters = potrfp<Precision::FP64>(stream, handle, epi, rank, N, (double*)A, lda, jpiv, (double*)pinned_work); break;
    case Precision::FP32:
      *iters = potrfp<Precision::FP32>(stream, handle, epi, rank, N, (float*)A, lda, jpiv, (float*)pinned_work); break;
    case Precision::FP128_DD:
      *iters = potrfp<Precision::FP128_DD>(stream, handle, epi, rank, N, (double2*)A, lda, jpiv, (double2*)pinned_work); break;
    case Precision::FP128_QF:
      *iters = potrfp<Precision::FP128_QF>(stream, handle, epi, rank, N, (float4*)A, lda, jpiv, (float4*)pinned_work); break;
    default:
      *iters = 0; break;
  }
}

void device::Cholesky::cpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work) {
  epi = std::min(1., std::max(0., std::abs(epi)));
  int32_t rank = std::min(N, std::max(0, *iters));

  switch (precA) {
    case Precision::FP64:
      *iters = potrfp<Precision::FP64>(stream, handle, epi, rank, N, (std::complex<double>*)A, lda, jpiv, (double*)pinned_work); break;
    case Precision::FP32:
      *iters = potrfp<Precision::FP32>(stream, handle, epi, rank, N, (std::complex<float>*)A, lda, jpiv, (float*)pinned_work); break;
    case Precision::FP128_DD:
      *iters = potrfp<Precision::FP128_DD>(stream, handle, epi, rank, N, (complex_double2*)A, lda, jpiv, (double2*)pinned_work); break;
    case Precision::FP128_QF:
      *iters = potrfp<Precision::FP128_QF>(stream, handle, epi, rank, N, (complex_float4*)A, lda, jpiv, (float4*)pinned_work); break;
    default:
      *iters = 0; break;
  }
}

