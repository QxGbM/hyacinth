
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

template <int32_t prec, class real_t, class matrix_t>
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const  matrix_t* X, int32_t incx, real_t* D, real_t* diag_piv) {
  if constexpr(prec == 0)
    internal::Cholesky::imax_f64(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 1)
    internal::Cholesky::imax_f32(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 2)
    internal::Cholesky::imax_f128_dd(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 3)
    internal::Cholesky::imax_f128_qf(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 8)
    internal::Cholesky::imax_cf64(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 9)
    internal::Cholesky::imax_cf32(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 10)
    internal::Cholesky::imax_cf128_dd(stream, N, X, incx, D, diag_piv);
  else if constexpr(prec == 11)
    internal::Cholesky::imax_cf128_qf(stream, N, X, incx, D, diag_piv);
}

template <int32_t prec, class real_t, class matrix_t>
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, real_t* scale, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda, real_t* D) {
  if constexpr(prec == 0)
    internal::Cholesky::gemv_cublas_f64(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 1)
    internal::Cholesky::gemv_cublas_f32(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 2)
    internal::Cholesky::gemv_scal_f128_dd(stream, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 3)
    internal::Cholesky::gemv_scal_f128_qf(stream, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 8)
    internal::Cholesky::gemv_cublas_cf64(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 9)
    internal::Cholesky::gemv_cublas_cf32(stream, handle, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 10)
    internal::Cholesky::gemv_scal_cf128_dd(stream, scale, j, M, N, A, lda, D);
  else if constexpr(prec == 11)
    internal::Cholesky::gemv_scal_cf128_qf(stream, scale, j, M, N, A, lda, D);
}

template <int32_t prec, class real_t>
inline double conv_f64(real_t r) {
  if constexpr((prec & 7) == 0) return r;
  else if constexpr((prec & 7) == 1) return double(r);
  else if constexpr((prec & 7) == 2) return r.x;
  else if constexpr((prec & 7) == 3) return double(r.x) + double(r.y) + double(r.z);
}

template <int32_t prec, class real_t, class matrix_t>
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, matrix_t* A, int32_t lda, int32_t* jpiv, real_t* scale) {
  real_t* diag = (real_t*)(&A[int64_t(N) * int64_t(lda)]);
  imax_dispatcher<prec>(stream, N, A, lda + 1, diag, scale);
  int32_t* pivot_i = (int32_t*)&scale[2], iters = std::min(N, p + (k ?: N));
  epi = conv_f64<prec>(scale[0]) * std::min(1., std::max(0., std::abs(epi)));

  for (int32_t i = 0, s = 0; i < iters; ++i) {
    int32_t j = (*pivot_i) - 1;
    double diag_f64 = conv_f64<prec>(scale[0]);

    if ((!std::isnormal(diag_f64)) || (p < (s += int32_t(diag_f64 < epi))) || (j < 0)) return i;
      else if (0 < j) std::iter_swap(&jpiv[i], &jpiv[i + j]);
    gemv_dispatcher<prec>(stream, handle, scale, j, N - i, i, &A[int64_t(i) * int64_t(lda)], lda, &diag[i]);
  }
  return iters;
}

int32_t internal::Cholesky::potrfp_f64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<0>(stream, handle, epi, k, p, N, A, lda, jpiv, (double*)pinned_work);
}

int32_t internal::Cholesky::potrfp_f32(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<1>(stream, handle, epi, k, p, N, A, lda, jpiv, (float*)pinned_work);
}

int32_t internal::Cholesky::potrfp_f128_dd(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<2>(stream, handle, epi, k, p, N, A, lda, jpiv, (double2*)pinned_work);
}

int32_t internal::Cholesky::potrfp_f128_qf(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<3>(stream, handle, epi, k, p, N, A, lda, jpiv, (float4*)pinned_work);
}

int32_t internal::Cholesky::potrfp_cf64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<8>(stream, handle, epi, k, p, N, A, lda, jpiv, (double*)pinned_work);
}

int32_t internal::Cholesky::potrfp_cf32(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<9>(stream, handle, epi, k, p, N, A, lda, jpiv, (float*)pinned_work);
}

int32_t internal::Cholesky::potrfp_cf128_dd(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<10>(stream, handle, epi, k, p, N, A, lda, jpiv, (double2*)pinned_work);
}

int32_t internal::Cholesky::potrfp_cf128_qf(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, void* pinned_work) {
  return potrfp<11>(stream, handle, epi, k, p, N, A, lda, jpiv, (float4*)pinned_work);
}
