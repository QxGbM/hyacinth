
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <float_max.hpp>

template <class matrix_t> __device__ __forceinline__ matrix_t conj(matrix_t a) { return a; }
template <> __device__ __forceinline__ cuDoubleComplex conj(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
template <> __device__ __forceinline__ cuComplex conj(cuComplex a) { return make_cuComplex(a.x, -a.y); }
template <> __device__ __forceinline__ complex_double2 conj(complex_double2 a) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); }
template <> __device__ __forceinline__ complex_float4 conj(complex_float4 a) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); }

template <char mode, class matrix_t>
__global__ void matrix_fill_upper_to_full(matrix_t* __restrict__ A, int64_t lda) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred; if constexpr(mode == 'U') { pred = y < x; } else if constexpr(mode == 'L') { pred = x < y; } else { pred = 0; }
  if (pred) A[x + y * lda] = conj(A[y + x * lda]);
}

template <class real_t, class matrix_t, class idx_t>
inline int32_t potrfp_dispatcher(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, matrix_t* A, int32_t lda, int32_t* jpiv, real_t* dvec, idx_t* hvec) {
  if (fillmode == 'U' || fillmode == 'u')
    matrix_fill_upper_to_full<'U', matrix_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, int64_t(lda));
  else if (fillmode == 'L' || fillmode == 'l')
    matrix_fill_upper_to_full<'L', matrix_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, int64_t(lda));
  internal::Cholesky::imax_initializer(stream, epi, N, A, lda + 1, jpiv, dvec, hvec);
  int32_t iters = std::min(N, std::max(0, k)); iters = iters ? iters : N; p = std::max(0, p);

  for (int32_t i = 0, s = 0; i < iters; ++i) {
    int32_t j = hvec[0].idx;
    if ((p < (s += hvec[1].idx)) || (j < 0)) { return i; }
    internal::Cholesky::gemv_scal(stream, handle, hvec, j, N - i, i, &A[int64_t(i) * int64_t(lda)], lda, jpiv, dvec);
  }
  return iters;
}

namespace internal::Cholesky {

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (double_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (float_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (double2_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (float4_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (double_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (float_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (double2_idx*)pinned_work);
  }

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, void* pinned_work) {
    return potrfp_dispatcher(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, (float4_idx*)pinned_work);
  }

};