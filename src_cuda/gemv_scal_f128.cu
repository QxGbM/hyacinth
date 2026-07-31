
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct add_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
};

template <class T> __device__ __forceinline__ T load_const(const T* a) { return __ldg(a); }
template <> __device__ __forceinline__ complex_double2 load_const<complex_double2>(const complex_double2* a) { return device::dd::make_complex_double2(__ldg(&a->real), __ldg(&a->imag)); }
template <> __device__ __forceinline__ complex_float4 load_const<complex_float4>(const complex_float4* a) { return device::qf::make_complex_float4(__ldg(&a->real), __ldg(&a->imag)); }

__device__ __forceinline__ double2 fma_func(const double2 a, const double2 b, double2 c) { return device::dd::add(c, device::dd::mul(a, b)); }
__device__ __forceinline__ float4 fma_func(const float4 a, const float4 b, float4 c) { return device::qf::add(c, device::qf::mul(a, b)); }
__device__ __forceinline__ complex_double2 fma_func(const complex_double2 a, const complex_double2 b, complex_double2 c) {
  using device::dd::add, device::dd::mul, device::dd::negate, device::dd::make_complex_double2;
  return make_complex_double2(add(mul(a.real, b.real), add(mul(a.imag, b.imag), c.real)), add(mul(a.real, b.imag), add(mul(negate(a.imag), b.real), c.imag)));
}
__device__ __forceinline__ complex_float4 fma_func(const complex_float4 a, const complex_float4 b, complex_float4 c) {
  using device::qf::add, device::qf::mul, device::qf::negate, device::qf::make_complex_float4;
  return make_complex_float4(add(mul(a.real, b.real), add(mul(a.imag, b.imag), c.real)), add(mul(a.real, b.imag), add(mul(negate(a.imag), b.real), c.imag)));
}

__device__ __forceinline__ double2 neg_func(double2 a) { return device::dd::negate(a); }
__device__ __forceinline__ float4 neg_func(float4 a) { return device::qf::negate(a); }
__device__ __forceinline__ complex_double2 neg_func(complex_double2 a) { return device::dd::make_complex_double2(device::dd::negate(a.real), device::dd::negate(a.imag)); }
__device__ __forceinline__ complex_float4 neg_func(complex_float4 a) { return device::qf::make_complex_float4(device::qf::negate(a.real), device::qf::negate(a.imag)); }

template <int32_t WARP_THREADS, int32_t BLOCK_WARPS, class matrix_t>
__global__ void gemv_kernel(int32_t M, int32_t N, matrix_t* __restrict__ A, int64_t ldj, int64_t lda) {
  __shared__ typename cub::BlockReduce<matrix_t, WARP_THREADS>::TempStorage temp_reduce[BLOCK_WARPS];

  int32_t i = BLOCK_WARPS * blockIdx.x + threadIdx.y;
  const matrix_t* A_i = &A[int64_t(i) * lda], *A_j = &A[ldj];

  if (i < M && A_i != A_j) {
    matrix_t threadB = matrix_t();
    for (int32_t k = threadIdx.x; k < N; k += WARP_THREADS)
      threadB = fma_func(load_const(&A_i[k]), load_const(&A_j[k]), threadB);

    add_f128 add_func;
    threadB = cub::BlockReduce<matrix_t, WARP_THREADS>(temp_reduce[threadIdx.y]).Reduce(threadB, add_func);
    A = &A[ldj + int64_t(i + N)];

    if (threadIdx.x == 0)
      *A = add_func(neg_func(threadB), *A);
  }
}

template <class matrix_t>
inline void gemv_dispatcher(cudaStream_t stream, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda) {
  constexpr int32_t wthreads[]{ 32, 64, 128, 256, 512 };
  constexpr int32_t bwarps[]{ 16, 8, 4, 2, 1 };
  constexpr int32_t target_blocks = 512;
  int32_t grid[]{ (M + 15) >> 4, (M + 7) >> 3, (M + 3) >> 2, (M + 1) >> 1 };
  int64_t lda64 = int64_t(lda), ldj = int64_t(j) * lda64;

  if (target_blocks <= grid[0] || N <= wthreads[0]) { gemv_kernel<wthreads[0], bwarps[0]> <<< grid[0], dim3(wthreads[0], bwarps[0], 1), 0, stream >>> (M, N, A, ldj, lda64); } else
  if (target_blocks <= grid[1] || N <= wthreads[1]) { gemv_kernel<wthreads[1], bwarps[1]> <<< grid[1], dim3(wthreads[1], bwarps[1], 1), 0, stream >>> (M, N, A, ldj, lda64); } else
  if (target_blocks <= grid[2] || N <= wthreads[2]) { gemv_kernel<wthreads[2], bwarps[2]> <<< grid[2], dim3(wthreads[2], bwarps[2], 1), 0, stream >>> (M, N, A, ldj, lda64); } else
  if (target_blocks <= grid[3] || N <= wthreads[3]) { gemv_kernel<wthreads[3], bwarps[3]> <<< grid[3], dim3(wthreads[3], bwarps[3], 1), 0, stream >>> (M, N, A, ldj, lda64); } else
  { gemv_kernel<wthreads[4], bwarps[4]> <<< M, dim3(wthreads[4], bwarps[4], 1), 0, stream >>> (M, N, A, ldj, lda64); }
}

namespace internal::Cholesky {

void gemv_scal(cudaStream_t stream, cublasHandle_t, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher(stream, j, M, N, A, lda);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(double2), cudaMemcpyHostToDevice, stream);
}

void gemv_scal(cudaStream_t stream, cublasHandle_t, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher(stream, j, M, N, A, lda);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(float4), cudaMemcpyHostToDevice, stream);
}

void gemv_scal(cudaStream_t stream, cublasHandle_t, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher(stream, j, M, N, A, lda);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M) {
    cudaMemsetAsync(&((double2*)scale)[1], 0, sizeof(double), stream);
    cudaMemcpyAsync(&A[N], scale, sizeof(complex_double2), cudaMemcpyHostToDevice, stream);
  }
}

void gemv_scal(cudaStream_t stream, cublasHandle_t, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher(stream, j, M, N, A, lda);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M) {
    cudaMemsetAsync(&((float4*)scale)[1], 0, sizeof(double), stream);
    cudaMemcpyAsync(&A[N], scale, sizeof(complex_float4), cudaMemcpyHostToDevice, stream);
  }
}

}
