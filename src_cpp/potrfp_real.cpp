
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <numeric>

template <Precision prec>
inline void imax_dispatcher(cudaStream_t stream, int32_t N, void* X, void* C, int32_t ldc, int32_t* piv, void* rsq) {
  if constexpr(prec == Precision::FP64)
    internal::Cholesky::imax_f64(stream, N, (double*)X, (double*)C, ldc, piv, (double*)rsq);
  else if constexpr(prec == Precision::FP32)
    internal::Cholesky::imax_f32(stream, N, (float*)X, (float*)C, ldc, piv, (float*)rsq);
  else if constexpr(prec == Precision::FP128_DD)
    internal::Cholesky::imax_f128_dd(stream, N, (double2*)X, (double2*)C, ldc, piv, (double2*)rsq);
  else if constexpr(prec == Precision::FP128_QF)
    internal::Cholesky::imax_f128_qf(stream, N, (float4*)X, (float4*)C, ldc, piv, (float4*)rsq);
}

template <Precision prec>
inline void swap_cols_dispatcher(cudaStream_t stream, int32_t i, int32_t j, int32_t N, void* A, int32_t lda) {
  if constexpr(prec == Precision::FP64)
    internal::Cholesky::swap_cols_f64(stream, i, j, N, (double*)A, lda);
  else if constexpr(prec == Precision::FP32)
    internal::Cholesky::swap_cols_f32(stream, i, j, N, (float*)A, lda);
  else if constexpr(prec == Precision::FP128_DD)
    internal::Cholesky::swap_cols_f128_dd(stream, i, j, N, (double2*)A, lda);
  else if constexpr(prec == Precision::FP128_QF)
    internal::Cholesky::swap_cols_f128_qf(stream, i, j, N, (float4*)A, lda);
}

template <Precision prec>
inline void gemv_dispatcher(cudaStream_t stream, void* scale, int32_t M, int32_t N, const void* A, int32_t lda, void* B, void* D) {
  if constexpr(prec == Precision::FP64)
    internal::Cholesky::gemv_scal_f64(stream, *((double*)scale), M, N, (const double*)A, lda, (double*)B, (double*)D);
  else if constexpr(prec == Precision::FP32)
    internal::Cholesky::gemv_scal_f32(stream, *((float*)scale), M, N, (const float*)A, lda, (float*)B, (float*)D);
  else if constexpr(prec == Precision::FP128_DD)
    internal::Cholesky::gemv_scal_f128_dd(stream, *((double2*)scale), M, N, (const double2*)A, lda, (double2*)B, (double2*)D);
  else if constexpr(prec == Precision::FP128_QF)
    internal::Cholesky::gemv_scal_f128_qf(stream, *((float4*)scale), M, N, (const float4*)A, lda, (float4*)B, (float4*)D);
}

template <Precision prec, class real_t>
inline double set_s0(real_t diag, double epi) {
  if constexpr(prec == Precision::FP64)
    return epi * diag;
  else if constexpr(prec == Precision::FP32)
    return epi * double(diag);
  else if constexpr(prec == Precision::FP128_DD)
    return epi * diag.x;
  else if constexpr(prec == Precision::FP128_QF)
    return epi * double(diag.x);
}

template <Precision prec, class real_t>
inline int32_t diag_pred(real_t diag, int32_t piv, double s0) {
  double diag_f64 = 0.;
  if constexpr(prec == Precision::FP64)
    diag_f64 = diag;
  else if constexpr(prec == Precision::FP32)
    diag_f64 = double(diag);
  else if constexpr(prec == Precision::FP128_DD)
    diag_f64 = diag.x;
  else if constexpr(prec == Precision::FP128_QF)
    diag_f64 = double(diag.x);
  return !(std::isnormal(diag_f64) && diag_f64 <= s0 && 0 <= piv);
}

template <Precision prec, class real_t>
inline int32_t real_potrfp(cudaStream_t stream, double epi, int32_t iters, int32_t N, real_t* A, int32_t lda, int32_t* jpiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &jpiv[algnN + 4];
  real_t* scale = (real_t*)(&jpiv[algnN]), *diag = (real_t*)(&A[uint64_t(N) * uint64_t(lda)]);
  double s0 = std::numeric_limits<double>::infinity();
  std::iota(jpiv, &jpiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(real_t), A, (lda + 1) * sizeof(real_t), sizeof(real_t), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < iters; ++i) {
    uint64_t A_diag = uint64_t(i) * uint64_t(lda + 1);
    uint64_t A_col = uint64_t(i) * uint64_t(lda);

    imax_dispatcher<prec>(stream, N - i, &diag[i], &A[A_diag], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (i == 0)
      s0 = set_s0<prec>(*scale, 1. / epi);
    if (diag_pred<prec>(*scale, *pivot_i, s0))
      return i;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&jpiv[i], &jpiv[j]);
      swap_cols_dispatcher<prec>(stream, i, j, N, A, lda);
    }
    gemv_dispatcher<prec>(stream, scale, N - i, i, &A[A_col], lda, &A[A_diag], &diag[i]);
  }
  return iters;
}

int32_t device::Cholesky::rpotrfp(cudaStream_t stream, double epi, int32_t iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv) {
  epi = std::min(1., std::max(epi, 0.));
  iters = std::min(N, std::max(0, iters));
  switch (precA) {
    case Precision::FP64:
      return real_potrfp<Precision::FP64, double>(stream, epi, iters, N, (double*)A, lda, jpiv);
    case Precision::FP32:
      return real_potrfp<Precision::FP32, float>(stream, epi, iters, N, (float*)A, lda, jpiv);
    case Precision::FP128_DD:
      return real_potrfp<Precision::FP128_DD, double2>(stream, epi, iters, N, (double2*)A, lda, jpiv);
    case Precision::FP128_QF:
      return real_potrfp<Precision::FP128_QF, float4>(stream, epi, iters, N, (float4*)A, lda, jpiv);
    default:
      return -1;
  }
}
