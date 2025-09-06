
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <numeric>

template <device::Precision prec, class real_t>
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const real_t* X, real_t* diag_piv) {
  using namespace internal::Cholesky;

  if constexpr(prec == device::Precision::FP64)
    imax_f64(stream, N, X, diag_piv);
  else if constexpr(prec == device::Precision::FP32)
    imax_f32(stream, N, X, diag_piv);
  else if constexpr(prec == device::Precision::FP128_DD)
    imax_f128_dd(stream, N, X, diag_piv);
  else if constexpr(prec == device::Precision::FP128_QF)
    imax_f128_qf(stream, N, X, diag_piv);
}

template <int32_t COMPLEX, device::Precision prec, class real_t, class matrix_t>
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, real_t* scale, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda, real_t* D, real_t* diag_piv) {
  using namespace internal::Cholesky;

  if constexpr(COMPLEX && prec == device::Precision::FP64)
    gemv_cublas_cf64(stream, handle, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(COMPLEX && prec == device::Precision::FP32)
    gemv_cublas_cf32(stream, handle, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(prec == device::Precision::FP64)
    gemv_cublas_f64(stream, handle, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(prec == device::Precision::FP32)
    gemv_cublas_f32(stream, handle, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(COMPLEX && prec == device::Precision::FP128_DD)
    gemv_scal_cf128_dd(stream, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(COMPLEX && prec == device::Precision::FP128_QF)
    gemv_scal_cf128_qf(stream, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(prec == device::Precision::FP128_DD)
    gemv_scal_f128_dd(stream, scale, j, M, N, A, lda, D, diag_piv);
  else if constexpr(prec == device::Precision::FP128_QF)
    gemv_scal_f128_qf(stream, scale, j, M, N, A, lda, D, diag_piv);
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
inline double conv_f64(real_t r) {
  if constexpr(prec == device::Precision::FP64) return r;
  else if constexpr(prec == device::Precision::FP32) return double(r);
  else if constexpr(prec == device::Precision::FP128_DD) return r.x;
  else if constexpr(prec == device::Precision::FP128_QF) return double(r.x) + double(r.y);
}

template <device::Precision prec, class real_t, class matrix_t>
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t iters, int32_t N, matrix_t* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
  real_t* scale = (real_t*)(pinned_work), *diag = (real_t*)(&A[int64_t(N) * int64_t(lda)]);
  int32_t* pivot_i = (int32_t*)&scale[1];

  std::iota(jpiv, &jpiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(real_t), A, sizeof(matrix_t) * int64_t(lda + 1), sizeof(real_t), N, cudaMemcpyDeviceToDevice, stream);
  imax_dispatcher<prec>(stream, N, diag, scale);

  int32_t j = *pivot_i;
  if (0 < j)
    std::iter_swap(&jpiv[0], &jpiv[j]);

  rsqrt_real<prec>(scale[0], scale[1]);
  double diag_f64 = conv_f64<prec>(scale[0]);
  gemv_dispatcher<COMPLEX, prec>(stream, handle, scale, j, N, 0, A, lda, diag, scale);

  epi = epi * diag_f64;
  if (!(std::isnormal(diag_f64) && epi <= diag_f64 && 0 <= j))
    return 1;

  for (int32_t i = 1; i < iters; ++i) {
    int64_t A_col = int64_t(i) * int64_t(lda);
    int32_t j = *pivot_i;
    if (0 < j)
      std::iter_swap(&jpiv[i], &jpiv[i + j]);

    rsqrt_real<prec>(scale[0], scale[1]);
    double diag_f64 = conv_f64<prec>(scale[0]);
    gemv_dispatcher<COMPLEX, prec>(stream, handle, scale, j, N - i, i, &A[A_col], lda, &diag[i], scale);

    if (!(std::isnormal(diag_f64) && epi <= diag_f64 && 0 <= j))
      return i + 1;
  }
  return iters;
}

void device::Cholesky::rpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work) {
  epi = std::min(1., std::max(0., epi));
  int32_t rank = std::min(N, std::max(0, *iters));
  
  switch (precA) {
    case Precision::FP64:
      *iters = potrfp<Precision::FP64, double, double>(stream, handle, epi, rank, N, (double*)A, lda, jpiv, pinned_work); break;
    case Precision::FP32:
      *iters = potrfp<Precision::FP32, float, float>(stream, handle, epi, rank, N, (float*)A, lda, jpiv, pinned_work); break;
    case Precision::FP128_DD:
      *iters = potrfp<Precision::FP128_DD, double2, double2>(stream, handle, epi, rank, N, (double2*)A, lda, jpiv, pinned_work); break;
    case Precision::FP128_QF:
      *iters = potrfp<Precision::FP128_QF, float4, float4>(stream, handle, epi, rank, N, (float4*)A, lda, jpiv, pinned_work); break;
    default:
      *iters = -1; break;
  }
}

void device::Cholesky::cpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work) {
  epi = std::min(1., std::max(0., epi));
  int32_t rank = std::min(N, std::max(0, *iters));

  switch (precA) {
    case Precision::FP64:
      *iters = potrfp<Precision::FP64, double, std::complex<double>>(stream, handle, epi, rank, N, (std::complex<double>*)A, lda, jpiv, pinned_work); break;
    case Precision::FP32:
      *iters = potrfp<Precision::FP32, float, std::complex<float>>(stream, handle, epi, rank, N, (std::complex<float>*)A, lda, jpiv, pinned_work); break;
    case Precision::FP128_DD:
      *iters = potrfp<Precision::FP128_DD, double2, complex_double2>(stream, handle, epi, rank, N, (complex_double2*)A, lda, jpiv, pinned_work); break;
    case Precision::FP128_QF:
      *iters = potrfp<Precision::FP128_QF, float4, complex_float4>(stream, handle, epi, rank, N, (complex_float4*)A, lda, jpiv, pinned_work); break;
    default:
      *iters = -1; break;
  }
}

