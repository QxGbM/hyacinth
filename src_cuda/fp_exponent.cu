
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <cub/cub.cuh>

struct abs_max {
  __device__ __forceinline__ double operator()(double a) { return fabs(a); }
  __device__ __forceinline__ float operator()(float a) { return fabsf(a); }
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
};

__device__ __forceinline__ int32_t fp_expon(double a) { int32_t expon; frexp(a, &expon); return expon - 1; }
__device__ __forceinline__ int32_t fp_expon(float a) { int32_t expon; frexpf(a, &expon); return expon - 1; }

template <class real_t, class real_const_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void vector_exponent_kernel(int32_t order, int32_t M, int32_t N, real_const_ptr A, int32_t lda, int32_t* __restrict__ vec_expon) {
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;
  int32_t M2 = M & (elements - 1), M1 = M - M2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<real_t, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t threadA[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<real_t, BLOCK_THREADS> block_reduce(temp_reduce);
  abs_max amax_func;

  for (int32_t col = blockIdx.x; col < N; col += GRID_BLOCKS) {
    real_const_ptr A_i = &A[int64_t(col) * int64_t(lda)];

    for (int32_t i = 0; i < M1; i += elements) {
      block_load.Load(&A_i[i], threadA);

      if (i == 0) {
        #pragma unroll
        for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
          threadB[j] = amax_func(threadA[j]);
      }
      else {
        #pragma unroll
        for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
          threadB[j] = amax_func(amax_func(threadA[j]), threadB[j]);
      }
    }

    if (0 < M2) {
      block_load.Load(&A_i[M1], threadA, M2, real_t());

      if (M1 == 0) {
        #pragma unroll
        for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
          threadB[j] = amax_func(threadA[j]);
      }
      else {
        #pragma unroll
        for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
          threadB[j] = amax_func(amax_func(threadA[j]), threadB[j]);
      }
    }

    real_t block_res = real_t();
    if (0 < M)
      block_res = block_reduce.Reduce(threadB, amax_func);

    if (threadIdx.x == 0) {
      int32_t expon = fp_expon(block_res), vec_e;
      device::int8::fast_division_i32<device::Config::exp_base>(expon, vec_e, expon);
      vec_expon[col] = vec_e - order + 1;
    }
  }
}

constexpr int32_t block_threads = 512;
constexpr int32_t grid_blocks = 512;
constexpr int32_t thread_bytes = 32;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  vector_exponent_kernel <double, const double* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M, N, A, lda, vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  vector_exponent_kernel <float, const float* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M, N, A, lda, vec_expon);
}

void internal::int8::vexp_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  vector_exponent_kernel <double, const double* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M * 2, N, (const double*)A, lda * 2, vec_expon);
}

void internal::int8::vexp_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  vector_exponent_kernel <float, const float* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M * 2, N, (const float*)A, lda * 2, vec_expon);
}

