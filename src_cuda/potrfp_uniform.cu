
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

template <class matrix_t> __device__ __forceinline__ matrix_t conj(matrix_t a) { return a; }
template <> __device__ __forceinline__ cuDoubleComplex conj(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
template <> __device__ __forceinline__ cuComplex conj(cuComplex a) { return make_cuComplex(a.x, -a.y); }
template <> __device__ __forceinline__ complex_double2 conj(complex_double2 a) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); }
template <> __device__ __forceinline__ complex_float4 conj(complex_float4 a) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); }

template <class matrix_t>
__global__ void matrix_fill_upper_to_full(matrix_t* __restrict__ A, int64_t lda) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < x) A[x + y * lda] = conj(A[y + x * lda]);
}

inline void imax_dispatcher(cudaStream_t stream, int32_t N, const double* X, int32_t incx, int32_t* jpiv, double* D, double* diag_piv)
{ internal::Cholesky::imax_f64(stream, N, X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const float* X, int32_t incx, int32_t* jpiv, float* D, float* diag_piv)
{ internal::Cholesky::imax_f32(stream, N, X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const double2* X, int32_t incx, int32_t* jpiv, double2* D, double2* diag_piv)
{ internal::Cholesky::imax_f128_dd(stream, N, X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const float4* X, int32_t incx, int32_t* jpiv, float4* D, float4* diag_piv)
{ internal::Cholesky::imax_f128_qf(stream, N, X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const cuDoubleComplex* X, int32_t incx, int32_t* jpiv, double* D, double* diag_piv)
{ internal::Cholesky::imax_cf64(stream, N, (const std::complex<double>*)X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const cuComplex* X, int32_t incx, int32_t* jpiv, float* D, float* diag_piv)
{ internal::Cholesky::imax_cf32(stream, N, (const std::complex<float>*)X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const complex_double2* X, int32_t incx, int32_t* jpiv, double2* D, double2* diag_piv)
{ internal::Cholesky::imax_cf128_dd(stream, N, X, incx, jpiv, D, diag_piv); }
inline void imax_dispatcher(cudaStream_t stream, int32_t N, const complex_float4* X, int32_t incx, int32_t* jpiv, float4* D, float4* diag_piv)
{ internal::Cholesky::imax_cf128_qf(stream, N, X, incx, jpiv, D, diag_piv); }

inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, double* D)
{ internal::Cholesky::gemv_cublas_f64(stream, handle, scale, j, M, N, A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, float* D)
{ internal::Cholesky::gemv_cublas_f32(stream, handle, scale, j, M, N, A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t, double2* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, double2* D)
{ internal::Cholesky::gemv_scal_f128_dd(stream, scale, j, M, N, A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t, float4* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, float4* D)
{ internal::Cholesky::gemv_scal_f128_qf(stream, scale, j, M, N, A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t j, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, double* D)
{ internal::Cholesky::gemv_cublas_cf64(stream, handle, scale, j, M, N, (std::complex<double>*)A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t j, int32_t M, int32_t N, cuComplex* A, int32_t lda, float* D)
{ internal::Cholesky::gemv_cublas_cf32(stream, handle, scale, j, M, N, (std::complex<float>*)A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t, double2* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D)
{ internal::Cholesky::gemv_scal_cf128_dd(stream, scale, j, M, N, A, lda, D); }
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t, float4* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D)
{ internal::Cholesky::gemv_scal_cf128_qf(stream, scale, j, M, N, A, lda, D); }

inline double conv_f64(double r) { return r; }
inline double conv_f64(float r) { return double(r); }
inline double conv_f64(double2 r) { return r.x; }
inline double conv_f64(float4 r) { return double(r.x) + double(r.y) + double(r.z); }

template <class real_t, class matrix_t>
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, matrix_t* A, int32_t lda, int32_t* jpiv, real_t* hvec, real_t* dvec) {
  if (fillmode == 'U' || fillmode == 'u')
    matrix_fill_upper_to_full<matrix_t> <<< dim3(uint32_t((N + 511) >> 9), uint32_t(N)), 512, 0, stream >>> (A, int64_t(lda));
  imax_dispatcher(stream, N, A, lda + 1, jpiv, dvec, hvec);
  int32_t* pivot_i = (int32_t*)&hvec[2], iters = std::min(N, std::max(0, k)); iters = iters ? iters : N;
  epi = conv_f64(hvec[0]) * std::min(1., std::max(0., std::abs(epi)));

  for (int32_t i = 0, s = 0; i < iters; ++i) {
    int32_t j = (*pivot_i) - 1;
    double diag_f64 = conv_f64(hvec[0]);

    if ((!std::isnormal(diag_f64)) || (p < (s += int32_t(diag_f64 < epi))) || (j < 0)) return i;
      else if (0 < j) std::iter_swap(&jpiv[i], &jpiv[i + j]);
    gemv_dispatcher(stream, handle, hvec, j, N - i, i, &A[int64_t(i) * int64_t(lda)], lda, &dvec[i]);
  }
  return iters;
}

int32_t internal::Cholesky::potrfp_f64(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, (double*)pinned_work, (double*)dev_work);
}

int32_t internal::Cholesky::potrfp_f32(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, (float*)pinned_work, (float*)dev_work);
}

int32_t internal::Cholesky::potrfp_f128_dd(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, (double2*)pinned_work, (double2*)dev_work);
}

int32_t internal::Cholesky::potrfp_f128_qf(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, (float4*)pinned_work, (float4*)dev_work);
}

int32_t internal::Cholesky::potrfp_cf64(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, (cuDoubleComplex*)A, lda, jpiv, (double*)pinned_work, (double*)dev_work);
}

int32_t internal::Cholesky::potrfp_cf32(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, (cuComplex*)A, lda, jpiv, (float*)pinned_work, (float*)dev_work);
}

int32_t internal::Cholesky::potrfp_cf128_dd(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, (double2*)pinned_work, (double2*)dev_work);
}

int32_t internal::Cholesky::potrfp_cf128_qf(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work) {
  return potrfp(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, (float4*)pinned_work, (float4*)dev_work);
}
