
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>

struct add_real {
  __device__ __forceinline__ double operator()(double a, double b) { return a + b; }
  __device__ __forceinline__ float operator()(float a, float b) { return a + b; }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
};

struct minus_a_mul_real {
  __device__ __forceinline__ double operator()(double a, double b) { return -a * b; }
  __device__ __forceinline__ float operator()(float a, float b) { return -a * b; }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { 
    return device::dd::mul(device::dd::negate(a), b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { 
    return device::qf::mul(device::qf::negate(a), b); }
};

struct minus_a_fma_real {
  __device__ __forceinline__ double operator()(double a, double b, double c) { return fma(-a, b, c); }
  __device__ __forceinline__ float operator()(float a, float b, float c) { return fmaf(-a, b, c); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 c) { return device::dd::fma(device::dd::negate(a), b, c); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) { return device::qf::fma(device::qf::negate(a), b, c); }
};

struct scal_add_real {
  __device__ __forceinline__ double operator()(double a, double b, double s) { return s * (a + b); }
  __device__ __forceinline__ float operator()(float a, float b, float s) { return s * (a + b); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 s) { 
    return device::dd::mul(s, device::dd::add(a, b)); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 s) { 
    return device::qf::mul(s, device::qf::add(a, b)); }
};

template <class real_t, class real_ptr, class real_const_ptr, int32_t GRID_WARPS, int32_t BLOCK_WARPS, int32_t ITEMS_PER_THREAD>
__global__ void minus_transAx_plusB_scale_real(real_const_ptr scale, int32_t M, int32_t N, real_const_ptr A, int32_t lda, real_ptr B) {
  constexpr int32_t BLOCK_THREADS = BLOCK_WARPS * 32;
  constexpr int32_t GRID_BLOCKS = GRID_WARPS / BLOCK_WARPS;
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;
  int32_t rem = N & (elements - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<real_t, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<real_t, BLOCK_THREADS> block_reduce(temp_reduce);
  add_real add_func;
  minus_a_mul_real mul_func;
  minus_a_fma_real fma_func;

  for (int32_t row = blockIdx.x; row < M; row += GRID_BLOCKS) {
    real_const_ptr A_i = &A[row * lda];

    block_load.Load(A_i, threadA, N1, real_t());
    block_load.Load(A, threadX, N1, real_t());

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      threadB[i] = mul_func(threadA[i], threadX[i]);

    for (int32_t i = elements; i < N1; i += elements) {
      block_load.Load(&A_i[i], threadA);
      block_load.Load(&A[i], threadX);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        threadB[j] = fma_func(threadA[j], threadX[j], threadB[j]);
    }

    if (0 < N2) {
      block_load.Load(&A_i[N1], threadA, N2, real_t());
      block_load.Load(&A[N1], threadX, N2, real_t());

      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        threadB[i] = fma_func(threadA[i], threadX[i], threadB[i]);
    }

    real_t block_res = block_reduce.Reduce(threadB, add_func);
    __syncthreads();

    if (threadIdx.x == 0) {
      scal_add_real scal_func;

      real_t res = scal_func(block_res, B[row], *scale);
      B[row] = res;
      B[row * lda] = res;
    }
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t grid_size = 2048;
constexpr int32_t grid_warps = grid_size * block_warps;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::minus_transAx_plusB_scale_double(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_transAx_plusB_scale_real <double, double* __restrict__, const double* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_transAx_plusB_scale_real <double, double* __restrict__, const double* __restrict__, grid_warps, 1, items_per_thread>
      <<< grid_size, 32, 0, stream >>> (scale, M, N, A, lda, B);
}

void internal::Cholesky::minus_transAx_plusB_scale_float(cudaStream_t stream, const float* scale, int32_t M, int32_t N, const float* A, int32_t lda, float* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_transAx_plusB_scale_real <float, float* __restrict__, const float* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_transAx_plusB_scale_real <float, float* __restrict__, const float* __restrict__, grid_warps, 1, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, A, lda, B);
}

void internal::Cholesky::minus_transAx_plusB_scale_double2(cudaStream_t stream, const double2* scale, int32_t M, int32_t N, const double2* A, int32_t lda, double2* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_transAx_plusB_scale_real <double2, double2* __restrict__, const double2* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_transAx_plusB_scale_real <double2, double2* __restrict__, const double2* __restrict__, grid_warps, 1, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, A, lda, B);
}

void internal::Cholesky::minus_transAx_plusB_scale_float4(cudaStream_t stream, const float4* scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_transAx_plusB_scale_real <float4, float4* __restrict__, const float4* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_transAx_plusB_scale_real <float4, float4* __restrict__, const float4* __restrict__, grid_warps, 1, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, A, lda, B);
}

