
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct fma_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 c) {
    return device::dd::add(c, device::dd::mul(a, b)); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) {
    return device::qf::add(c, device::qf::mul(a, b)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, complex_double2 c) {
    return device::dd::make_complex_double2(operator()(a.real, b.real, operator()(a.imag, b.imag, c.real)), 
      operator()(a.real, b.imag, operator()(device::dd::negate(a.imag), b.real, c.imag))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) {
    return device::qf::make_complex_float4(operator()(a.real, b.real, operator()(a.imag, b.imag, c.real)), 
      operator()(a.real, b.imag, operator()(device::qf::negate(a.imag), b.real, c.imag))); }
};

struct add_neg_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }

  __device__ __forceinline__ double2 operator()(double2 a) { return device::dd::negate(a); }
  __device__ __forceinline__ float4 operator()(float4 a) { return device::qf::negate(a); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a) {
    return device::dd::make_complex_double2(device::dd::negate(a.real), device::dd::negate(a.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a) {
    return device::qf::make_complex_float4(device::qf::negate(a.real), device::qf::negate(a.imag)); }
};

template <class matrix_t, class matrix_ptr, int32_t WARP_THREADS, int32_t BLOCK_THREADS>
__global__ void gemv_kernel(int32_t M, int32_t N, matrix_ptr A, int64_t ldj, int64_t lda) {
  constexpr int32_t block_warps = BLOCK_THREADS / WARP_THREADS;
  __shared__ typename cub::BlockReduce<matrix_t, WARP_THREADS>::TempStorage temp_reduce[block_warps];
  cub::BlockReduce<matrix_t, WARP_THREADS> block_reduce(temp_reduce[threadIdx.y]);

  int32_t i = block_warps * blockIdx.x + threadIdx.y;
  const matrix_ptr A_i = &A[int64_t(i) * lda], A_j = &A[ldj];

  if (i < M && A_i != A_j) {
    matrix_t threadB = matrix_t();
    fma_f128 fma_func;
    
    for (int32_t k = threadIdx.x; k < N; k += WARP_THREADS)
      threadB = fma_func(A_i[k], A_j[k], threadB);

    add_neg_f128 an_func;
    threadB = block_reduce.Reduce(threadB, an_func);
    A = &A[ldj + int64_t(i + N)];

    if (threadIdx.x == 0)
      *A = an_func(an_func(threadB), *A);
  }
}

constexpr int32_t target_blocks = 512;

template <class matrix_t, class matrix_ptr>
inline void gemv_dispatcher(cudaStream_t stream, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda) {
  constexpr int32_t warp_threads[]{ 1, 2, 4, 8, 16, 32, 64, 128, 256 };
  constexpr int32_t block_threads = 512;
  int32_t grid[]{ (M + 511) >> 9, (M + 255) >> 8, (M + 127) >> 7, (M + 63) >> 6, (M + 31) >> 5, (M + 15) >> 4, (M + 7) >> 3, (M + 3) >> 2, (M + 1) >> 1 };

  if (target_blocks <= grid[0] || N <= warp_threads[0])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[0], block_threads>
      <<< grid[0], dim3(warp_threads[0], block_threads / warp_threads[0], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[1] || N <= warp_threads[1])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[1], block_threads>
      <<< grid[1], dim3(warp_threads[1], block_threads / warp_threads[1], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[2] || N <= warp_threads[2])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[2], block_threads>
      <<< grid[2], dim3(warp_threads[2], block_threads / warp_threads[2], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[3] || N <= warp_threads[3])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[3], block_threads>
      <<< grid[3], dim3(warp_threads[3], block_threads / warp_threads[3], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[4] || N <= warp_threads[4])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[4], block_threads>
      <<< grid[4], dim3(warp_threads[4], block_threads / warp_threads[4], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[5] || N <= warp_threads[5])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[5], block_threads>
      <<< grid[5], dim3(warp_threads[5], block_threads / warp_threads[5], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[6] || N <= warp_threads[6])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[6], block_threads>
      <<< grid[6], dim3(warp_threads[6], block_threads / warp_threads[6], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[7] || N <= warp_threads[7])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[7], block_threads>
      <<< grid[7], dim3(warp_threads[7], block_threads / warp_threads[7], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else if (target_blocks <= grid[8] || N <= warp_threads[8])
    gemv_kernel <matrix_t, matrix_ptr, warp_threads[8], block_threads>
      <<< grid[8], dim3(warp_threads[8], block_threads / warp_threads[8], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
  else
    gemv_kernel <matrix_t, matrix_ptr, block_threads, block_threads>
      <<< M, block_threads, 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), lda);
}

void internal::Cholesky::gemv_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, double2* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher<double2, double2* __restrict__>(stream, j, M, N, A, lda);
    if (j)
      gemv_pp_f128_dd(stream, j, N, M, scale, A, lda, D);
    else
      gemv_pp_nopiv_f128_dd(stream, N, M, scale, A, lda, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(double2), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, float4* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher<float4, float4* __restrict__>(stream, j, M, N, A, lda);
    if (j)
      gemv_pp_f128_qf(stream, j, N, M, scale, A, lda, D);
    else
      gemv_pp_nopiv_f128_qf(stream, N, M, scale, A, lda, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(float4), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher<complex_double2, complex_double2* __restrict__>(stream, j, M, N, A, lda);
    if (j)
      gemv_pp_cf128_dd(stream, j, N, M, scale, A, lda, D);
    else
      gemv_pp_nopiv_cf128_dd(stream, N, M, scale, A, lda, D);
  }
  else if (1 == M) {
    scale[1] = make_double2(0., 0.);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(double2), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::gemv_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D) {
  if (2 <= M) {
    if (1 <= N)
      gemv_dispatcher<complex_float4, complex_float4* __restrict__>(stream, j, M, N, A, lda);
    if (j)
      gemv_pp_cf128_qf(stream, j, N, M, scale, A, lda, D);
    else
      gemv_pp_nopiv_cf128_qf(stream, N, M, scale, A, lda, D);
  }
  else if (1 == M) {
    scale[1] = make_float4(0.f, 0.f, 0.f, 0.f);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(float4), cudaMemcpyHostToDevice, stream);
  }
}
