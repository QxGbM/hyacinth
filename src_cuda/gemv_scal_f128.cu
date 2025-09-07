
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

struct fma_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::mul(device::dd::negate(a), b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::mul(device::qf::negate(a), b); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 c) { return device::dd::add(c, operator()(a, b)); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) { return device::qf::add(c, operator()(a, b)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real, operator()(a.imag, b.imag)), 
      operator()(a.real, b.imag, operator()(device::dd::negate(a.imag), b.real))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real, operator()(a.imag, b.imag)), 
      operator()(a.real, b.imag, operator()(device::qf::negate(a.imag), b.real))); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, complex_double2 c) { 
    return device::dd::make_complex_double2(operator()(a.real, b.real, operator()(a.imag, b.imag, c.real)), 
      operator()(a.real, b.imag, operator()(device::dd::negate(a.imag), b.real, c.imag))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) { 
    return device::qf::make_complex_float4(operator()(a.real, b.real, operator()(a.imag, b.imag, c.real)), 
      operator()(a.real, b.imag, operator()(device::qf::negate(a.imag), b.real, c.imag))); }
};

template <int32_t ALG, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ void array_fma(matrix_t const (&a)[ITEMS_PER_THREAD], matrix_t const (&b)[ITEMS_PER_THREAD], matrix_t (&c)[ITEMS_PER_THREAD]) {
  fma_f128 fma_func;
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    if constexpr(ALG == 0)
      c[i] = fma_func(a[i], b[i]);
    else
      c[i] = fma_func(a[i], b[i], c[i]);
}

struct add_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
};

template <class real_t, class matrix_t> struct scal_f128 {
  real_t scale;
  scal_f128(real_t scale) : scale(scale) {}
  __device__ __forceinline__ double2 scal_a(double2 s, double2 a) { return device::dd::mul(s, a); }
  __device__ __forceinline__ float4 scal_a(float4 s, float4 a) { return device::qf::mul(s, a); }
  __device__ __forceinline__ complex_double2 scal_a(double2 s, complex_double2 a) {
    return device::dd::make_complex_double2(scal_a(s, a.real), scal_a(s, a.imag)); }
  __device__ __forceinline__ complex_float4 scal_a(float4 s, complex_float4 a) {
    return device::qf::make_complex_float4(scal_a(s, a.real), scal_a(s, a.imag)); }
  __device__ __forceinline__ matrix_t operator()(matrix_t a) { return scal_a(scale, a); }
};

template <class matrix_ptr, class matrix_const_ptr, int32_t WARP_THREADS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, class real_t, class matrix_t>
__global__ void gemv_kernel(int32_t j, int32_t M, int32_t N, matrix_const_ptr A, int32_t lda, matrix_ptr B, scal_f128<real_t, matrix_t> scal_func) {
  constexpr int32_t block_warps = BLOCK_THREADS / WARP_THREADS;
  constexpr int32_t elements = ITEMS_PER_THREAD * WARP_THREADS;
  int32_t N2 = N & (elements - 1), N1 = N - N2;
  int32_t inc_row = block_warps * gridDim.x;

  __shared__ typename cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load[block_warps];
  __shared__ typename cub::BlockReduce<matrix_t, WARP_THREADS>::TempStorage temp_reduce[block_warps];
  cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load[threadIdx.y]);
  cub::BlockReduce<matrix_t, WARP_THREADS> block_reduce(temp_reduce[threadIdx.y]);
  matrix_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];
  matrix_const_ptr A_j = &A[int64_t(j) * int64_t(lda)];

  for (int32_t i = (block_warps * blockIdx.x + threadIdx.y); i < M; i += inc_row) {
    matrix_const_ptr A_i = &A[int64_t(i) * int64_t(lda)];
    
    for (int32_t k = 0; k < N1; k += elements) {
      block_load.Load(&A_i[k], threadA);
      block_load.Load(&A_j[k], threadX);
      if (k == 0)
        array_fma<0>(threadA, threadX, threadB);
      else
        array_fma<1>(threadA, threadX, threadB);
    }

    if (0 < N2) {
      block_load.Load(&A_i[N1], threadA, N2, matrix_t());
      block_load.Load(&A_j[N1], threadX, N2, matrix_t());
      if (N1 == 0)
        array_fma<0>(threadA, threadX, threadB);
      else
        array_fma<1>(threadA, threadX, threadB);
    }

    add_f128 add_func;
    matrix_t block_res = block_reduce.Reduce(threadB, add_func);
    if (threadIdx.x == 0)
      B[i] = scal_func(add_func(block_res, B[i]));
  }
}

constexpr int32_t thread_bytes = 32;
constexpr int32_t target_blocks = 512;

template <class matrix_ptr, class matrix_const_ptr, class real_t, class matrix_t>
inline void gemv_dispatcher(cudaStream_t stream, real_t scale, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);
  constexpr int32_t warp_threads[4] { 32, 64, 128, 256 };
  constexpr int32_t warp_reduces[4] { 64 * items_per_thread, 128 * items_per_thread, 256 * items_per_thread, 512 * items_per_thread };
  constexpr int32_t block_threads = 512;
  int32_t grid[4] { (M + 15) >> 4, (M + 7) >> 3, (M + 3) >> 2, (M + 1) >> 1 };
  scal_f128<real_t, matrix_t> scal_func(scale);
  matrix_t* B = &A[int64_t(N) + int64_t(j) * int64_t(lda)];

  if (N <= 0) {
    thrust::device_ptr<matrix_t> Bptr(B);
    thrust::transform(thrust::cuda::par_nosync.on(stream), Bptr, &Bptr[M], Bptr, scal_func);
  }
  else if (target_blocks <= grid[0] || N < warp_reduces[0])
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[0], block_threads, items_per_thread>
      <<< grid[0], dim3(warp_threads[0], block_threads / warp_threads[0], 1), 0, stream >>> (j, M, N, A, lda, B, scal_func);
  else if (target_blocks <= grid[1] || N < warp_reduces[1])
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[1], block_threads, items_per_thread>
      <<< grid[1], dim3(warp_threads[1], block_threads / warp_threads[1], 1), 0, stream >>> (j, M, N, A, lda, B, scal_func);
  else if (target_blocks <= grid[2] || N < warp_reduces[2])
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[2], block_threads, items_per_thread>
      <<< grid[2], dim3(warp_threads[2], block_threads / warp_threads[2], 1), 0, stream >>> (j, M, N, A, lda, B, scal_func);
  else if (target_blocks <= grid[3] || N < warp_reduces[3])
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[3], block_threads, items_per_thread>
      <<< grid[3], dim3(warp_threads[3], block_threads / warp_threads[3], 1), 0, stream >>> (j, M, N, A, lda, B, scal_func);
  else
    gemv_kernel <matrix_ptr, matrix_const_ptr, block_threads, block_threads, items_per_thread>
      <<< M, block_threads, 0, stream >>> (j, M, N, A, lda, B, scal_func);
}

void internal::Cholesky::gemv_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, double2* D) {
  if (2 <= M) {
    gemv_dispatcher<double2* __restrict__, const double2* __restrict__>(stream, scale[1], j, M, N, A, lda);
    gemv_pp_f128_dd(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(double2), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, float4* D) {
  if (2 <= M) {
    gemv_dispatcher<float4* __restrict__, const float4* __restrict__>(stream, scale[1], j, M, N, A, lda);
    gemv_pp_f128_qf(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(float4), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D) {
  if (2 <= M) {
    gemv_dispatcher<complex_double2* __restrict__, const complex_double2* __restrict__>(stream, scale[1], j, M, N, A, lda);
    gemv_pp_cf128_dd(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M) {
    scale[1] = make_double2(0., 0.);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(double2), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::gemv_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D) {
  if (2 <= M) {
    gemv_dispatcher<complex_float4* __restrict__, const complex_float4* __restrict__>(stream, scale[1], j, M, N, A, lda);
    gemv_pp_cf128_qf(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M) {
    scale[1] = make_float4(0.f, 0.f, 0.f, 0.f);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(float4), cudaMemcpyHostToDevice, stream);
  }
}
