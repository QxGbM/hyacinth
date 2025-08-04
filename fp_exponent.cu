
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <cub/cub.cuh>

struct expon_imax {
  __device__ __forceinline__ int32_t operator()(double f) { return device::int8::get_double_top_exp(f); }
  __device__ __forceinline__ int32_t operator()(float f) { return device::int8::get_float_top_exp(f); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return max(a, b); }
};

template <class real_t, class real_const_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE>
__global__ void vector_exponent(int32_t order, int32_t M, int32_t N, real_const_ptr A, int32_t lda, int32_t* __restrict__ vec_expon) {
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;
  int32_t rem = M & (elements - 1), div = M - rem;
  int32_t M1 = max(div, rem), M2 = min(div, rem);

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t threadA[ITEMS_PER_THREAD];
  int32_t threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<int32_t, BLOCK_THREADS> block_reduce(temp_reduce);
  expon_imax expon_f;

  for (int32_t col = blockIdx.x; col < N; col += GRID_BLOCKS) {
    real_const_ptr A_i = &A[uint64_t(col) * uint64_t(lda)];
    block_load.Load(A_i, threadA, M1, real_t());

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      threadB[i] = expon_f(threadA[i]);

    for (int32_t i = elements; i < M1; i += elements) {
      block_load.Load(&A_i[i], threadA);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        threadB[j] = max(expon_f(threadA[j]), threadB[j]);
    }

    if (0 < M2) {
      block_load.Load(&A_i[M1], threadA, M2, real_t());

      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        threadB[i] = max(expon_f(threadA[i]), threadB[i]);
    }

    int32_t block_res = block_reduce.Reduce(threadB, expon_f);
    __syncthreads();

    if (threadIdx.x == 0) {
      int32_t vec_e;
      device::int8::fast_division_i32<BASE>(block_res, vec_e, block_res);
      vec_expon[col] = vec_e - order + 1;
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t thread_bytes = 32;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  vector_exponent <double, const double* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M, N, A, lda, vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  vector_exponent <float, const float* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M, N, A, lda, vec_expon);
}

void internal::int8::vexp_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  vector_exponent <double, const double* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M * 2, N, (const double*)A, lda * 2, vec_expon);
}

void internal::int8::vexp_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  vector_exponent <float, const float* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order, M * 2, N, (const float*)A, lda * 2, vec_expon);
}

