
#include <hyacinth.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <float4.hpp>

struct add_real {
  __device__ double operator()(double a, double b) { return a + b; }
  __device__ float operator()(float a, float b) { return a + b; }
  __device__ float4 operator()(float4 a, float4 b) { return device::f4::add(a, b); }
};

struct minus_a_fma_real {
  __device__ double operator()(double a, double b, double c) { return fma(-a, b, c); }
  __device__ float operator()(float a, float b, float c) { return fmaf(-a, b, c); }
  __device__ float4 operator()(float4 a, float4 b, float4 c) { return device::f4::fma(device::f4::negate(a), b, c); }
};

struct init_real {
  __device__ operator double() { return 0.; }
  __device__ operator float() { return 0.f; }
  __device__ operator float4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
};

struct scal_real {
  __device__ double operator()(double a, double b) { return a * b; }
  __device__ float operator()(float a, float b) { return a * b; }
  __device__ float4 operator()(float4 a, float4 b) { return device::f4::fma(a, b, make_float4(0.f, 0.f, 0.f, 0.f)); }
};

template <class real_t, class real_ptr, class real_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void minus_transAx_plusB_scale_real(double scale, int32_t N, real_const_ptr A, int32_t lda, real_const_ptr X, real_ptr B) {
  using BlockLoad = cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>;
  using BlockReduce = cub::BlockReduce<real_t, BLOCK_THREADS>;

  __shared__ typename BlockLoad::TempStorage temp_load;
  __shared__ typename BlockReduce::TempStorage temp_reduce;

  add_real add_func;
  minus_a_fma_real fma_func;
  init_real init_func;
  scal_real scal_func;

  real_t thread_A[ITEMS_PER_THREAD], thread_X[ITEMS_PER_THREAD], thread_B[ITEMS_PER_THREAD];
  int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;
  A += blockIdx.x * lda;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_B[i] = init_func;

  for (int32_t i = 0; i < N; i += elements) {
    int32_t num_items = min(elements, N - i);
    BlockLoad(temp_load).Load(&A[i], thread_A, num_items, init_func);
    BlockLoad(temp_load).Load(&X[i], thread_X, num_items, init_func);

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_B[j] = fma_func(thread_A[j], thread_X[j], thread_B[j]);
  }

  real_t res = BlockReduce(temp_reduce).Reduce(thread_B, add_func);
  if (threadIdx.x == 0)
    B[blockIdx.x] = scal_func(add_func(B[blockIdx.x], res), scale);
}

void minus_transAx_plusB_scale_double(cudaStream_t stream, double scale, int32_t M, int32_t N, const double* A, int32_t lda, const double* X, double* B) {
  minus_transAx_plusB_scale_real <double, double* __restrict__, const double* __restrict__, 32, 4>
    <<< M, 32, 0, stream >>> (scale, N, A, lda, X, B);
}

