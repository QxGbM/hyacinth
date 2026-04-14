
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct add_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
};

__device__ __forceinline__ double2 fma_func(const double2* a, const double2* b, double2 c) {
  return device::dd::add(c, device::dd::mul(__ldg(a), __ldg(b))); }
__device__ __forceinline__ float4 fma_func(const float4* a, const float4* b, float4 c) {
  return device::qf::add(c, device::qf::mul(__ldg(a), __ldg(b))); }
__device__ __forceinline__ complex_double2 fma_func(const complex_double2* a, const complex_double2* b, complex_double2 c) {
  double2 a_rl = __ldg(&a->real), a_im = __ldg(&a->imag), b_rl = __ldg(&b->real), b_im = __ldg(&b->imag);
  using device::dd::add, device::dd::mul, device::dd::negate, device::dd::make_complex_double2;
  return make_complex_double2(add(mul(a_rl, b_rl), add(mul(a_im, b_im), c.real)), add(mul(a_rl, b_im), add(mul(negate(a_im), b_rl), c.imag)));
}
__device__ __forceinline__ complex_float4 fma_func(const complex_float4* a, const complex_float4* b, complex_float4 c) {
  float4 a_rl = __ldg(&a->real), a_im = __ldg(&a->imag), b_rl = __ldg(&b->real), b_im = __ldg(&b->imag);
  using device::qf::add, device::qf::mul, device::qf::negate, device::qf::make_complex_float4;
  return make_complex_float4(add(mul(a_rl, b_rl), add(mul(a_im, b_im), c.real)), add(mul(a_rl, b_im), add(mul(negate(a_im), b_rl), c.imag)));
}

__device__ __forceinline__ double2 neg_func(double2 a) { return device::dd::negate(a); }
__device__ __forceinline__ float4 neg_func(float4 a) { return device::qf::negate(a); }
__device__ __forceinline__ complex_double2 neg_func(complex_double2 a) {
  return device::dd::make_complex_double2(device::dd::negate(a.real), device::dd::negate(a.imag)); }
__device__ __forceinline__ complex_float4 neg_func(complex_float4 a) {
  return device::qf::make_complex_float4(device::qf::negate(a.real), device::qf::negate(a.imag)); }

template <class matrix_t, int32_t WARP_THREADS, int32_t BLOCK_WARPS>
__global__ void gemv_kernel(int32_t M, int32_t N, matrix_t* __restrict__ A, int64_t ldj, int64_t lda) {
  __shared__ typename cub::BlockReduce<matrix_t, WARP_THREADS>::TempStorage temp_reduce[BLOCK_WARPS];

  int32_t i = BLOCK_WARPS * blockIdx.x + threadIdx.y;
  const matrix_t* A_i = &A[int64_t(i) * lda], *A_j = &A[ldj];

  if (i < M && A_i != A_j) {
    matrix_t threadB = matrix_t();
    
    for (int32_t k = threadIdx.x; k < N; k += WARP_THREADS)
      threadB = fma_func(&A_i[k], &A_j[k], threadB);

    add_f128 add_func;
    threadB = cub::BlockReduce<matrix_t, WARP_THREADS>(temp_reduce[threadIdx.y]).Reduce(threadB, add_func);
    A = &A[ldj + int64_t(i + N)];

    if (threadIdx.x == 0)
      *A = add_func(neg_func(threadB), *A);
  }
}

template <class matrix_t, class matrix_ptr>
inline void gemv_dispatcher(cudaStream_t stream, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda) {
  constexpr int32_t warp_threads[]{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
  constexpr int32_t block_warps[]{ 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 };
  constexpr int32_t target_blocks = 512;
  int32_t grid[]{ (M + 511) >> 9, (M + 255) >> 8, (M + 127) >> 7, (M + 63) >> 6, (M + 31) >> 5, (M + 15) >> 4, (M + 7) >> 3, (M + 3) >> 2, (M + 1) >> 1 };

  if (target_blocks <= grid[0] || N <= warp_threads[0])
    gemv_kernel<matrix_t, warp_threads[0], block_warps[0]>
      <<< grid[0], dim3(warp_threads[0], block_warps[0], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[1] || N <= warp_threads[1])
    gemv_kernel<matrix_t, warp_threads[1], block_warps[1]>
      <<< grid[1], dim3(warp_threads[1], block_warps[1], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[2] || N <= warp_threads[2])
    gemv_kernel<matrix_t, warp_threads[2], block_warps[2]>
      <<< grid[2], dim3(warp_threads[2], block_warps[2], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[3] || N <= warp_threads[3])
    gemv_kernel<matrix_t, warp_threads[3], block_warps[3]>
      <<< grid[3], dim3(warp_threads[3], block_warps[3], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[4] || N <= warp_threads[4])
    gemv_kernel<matrix_t, warp_threads[4], block_warps[4]>
      <<< grid[4], dim3(warp_threads[4], block_warps[4], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[5] || N <= warp_threads[5])
    gemv_kernel<matrix_t, warp_threads[5], block_warps[5]>
      <<< grid[5], dim3(warp_threads[5], block_warps[5], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[6] || N <= warp_threads[6])
    gemv_kernel<matrix_t, warp_threads[6], block_warps[6]>
      <<< grid[6], dim3(warp_threads[6], block_warps[6], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[7] || N <= warp_threads[7])
    gemv_kernel<matrix_t, warp_threads[7], block_warps[7]>
      <<< grid[7], dim3(warp_threads[7], block_warps[7], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else if (target_blocks <= grid[8] || N <= warp_threads[8])
    gemv_kernel<matrix_t, warp_threads[8], block_warps[8]>
      <<< grid[8], dim3(warp_threads[8], block_warps[8], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
  else
    gemv_kernel<matrix_t, warp_threads[9], block_warps[9]>
      <<< M, dim3(warp_threads[9], block_warps[9], 1), 0, stream >>> (M, N, A, int64_t(j) * int64_t(lda), int64_t(lda));
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
    cudaMemcpyAsync(&A[N], scale, sizeof(complex_double2), cudaMemcpyHostToDevice, stream);
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
    cudaMemcpyAsync(&A[N], scale, sizeof(complex_float4), cudaMemcpyHostToDevice, stream);
  }
}
