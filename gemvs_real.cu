
#include <hyacinth.hpp>

#include <cub/cub.cuh>
#include <float4.hpp>

struct add_real {
  __device__ __forceinline__ double operator()(double a, double b) { return a + b; }
  __device__ __forceinline__ float operator()(float a, float b) { return a + b; }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::f4::add(a, b); }
};

struct minus_a_fma_real {
  __device__ __forceinline__ double operator()(double a, double b, double c) { return fma(-a, b, c); }
  __device__ __forceinline__ float operator()(float a, float b, float c) { return fmaf(-a, b, c); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) { return device::f4::fma(device::f4::negate(a), b, c); }
};

struct init_real {
  __device__ __forceinline__ operator double() { return 0.; }
  __device__ __forceinline__ operator float() { return 0.f; }
  __device__ __forceinline__ operator float4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
};

struct scal_real {
  __device__ __forceinline__ double operator()(double a, double b) { return a * b; }
  __device__ __forceinline__ float operator()(float a, float b) { return a * b; }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::f4::fma(a, b, make_float4(0.f, 0.f, 0.f, 0.f)); }
};

template <class real_t, class real_ptr, class real_const_ptr, int32_t BLOCK_WARPS, int32_t ITEMS_PER_THREAD>
__global__ void minus_adjAx_plusB_scale_real(real_const_ptr scale, int32_t M, int32_t N, real_const_ptr A, int32_t lda, real_ptr B, real_ptr C) {
  using WarpLoad = cub::WarpLoad<real_t, ITEMS_PER_THREAD>;
  using WarpReduce = cub::WarpReduce<real_t>;
  constexpr int32_t elements = ITEMS_PER_THREAD * 32;

  __shared__ typename WarpLoad::TempStorage temp_loadA[BLOCK_WARPS], temp_loadX[BLOCK_WARPS];
  __shared__ typename WarpReduce::TempStorage temp_reduce[BLOCK_WARPS];

  add_real add_func;
  minus_a_fma_real fma_func;

  real_t thread_A[ITEMS_PER_THREAD], thread_X[ITEMS_PER_THREAD], thread_B[ITEMS_PER_THREAD], init = init_real();
  int32_t row = blockIdx.x * BLOCK_WARPS + threadIdx.y;

  if (row < M) {
    real_const_ptr A_i = &A[row * lda];

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      thread_B[i] = init;

    for (int32_t i = 0; i < N; i += elements) {
      int32_t num_items = min(elements, N - i);
      WarpLoad(temp_loadA[threadIdx.y]).Load(&A_i[i], thread_A, num_items, init);
      WarpLoad(temp_loadX[threadIdx.y]).Load(&A[i], thread_X, num_items, init);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        thread_B[j] = fma_func(thread_A[j], thread_X[j], thread_B[j]);
    }

    real_t thread_res = thread_B[0];
    #pragma unroll
    for (int32_t i = 1; i < ITEMS_PER_THREAD; ++i)
      thread_res = add_func(thread_res, thread_B[i]);

    real_t warp_res = WarpReduce(temp_reduce[threadIdx.y]).Reduce(thread_res, add_func);

    if (threadIdx.x == 0) {
      scal_real scal_func;

      real_t res = scal_func(add_func(warp_res, B[row]), *scale);
      B[row] = res;
      B[row * lda] = res;
      C[row] = fma_func(res, res, C[row]);
    }
  }
}

void minus_transAx_plusB_scale_double(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B, double* C) {
  constexpr int32_t block_warps = 4;
  constexpr int32_t items_per_thread = 8;

  int32_t grid_size = (M + block_warps - 1) / block_warps;
  minus_adjAx_plusB_scale_real <double, double* __restrict__, const double* __restrict__, block_warps, items_per_thread>
    <<< grid_size, dim3(32, block_warps, 1), 0, stream >>> (scale, M, N, A, lda, B, C);
}

