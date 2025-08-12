
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cub/cub.cuh>

struct acc_i32 {
  __device__ __forceinline__ void operator()(double& f, int32_t i, int32_t expon) {
    f += scalbn(double(i), expon);
  }

  __device__ __forceinline__ void operator()(float& f, int32_t i, int32_t expon) {
    f += scalbnf(float(i), expon);
  }

  __device__ __forceinline__ void operator()(double2& f, int32_t i, int32_t expon) {
    f = device::dd::add_sd(f, scalbn(double(i), expon));
  }

  __device__ __forceinline__ void operator()(float4& f, int32_t i, int32_t expon) {
    float c1 = float(i);
    float c2 = float(i - int32_t(c1));
    f = device::qf::add_df(f, make_float2(scalbnf(c1, expon), scalbnf(c2, expon)));
  }
};

template <class real_t, int32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void decode_i32_array(real_t (&val)[ITEMS_PER_THREAD], int32_t const (&c)[ITEMS_PER_THREAD], int32_t const (&expon)[ITEMS_PER_THREAD], int32_t expon_k) {
  acc_i32 acc_f;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    acc_f(val[i], c[i], expon[i] + expon_k);
}

template <class real_t, class real_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE>
__global__ void decode_kernel(int32_t k_begin, int32_t k_len, int32_t N, const int32_t* __restrict__ vec_expon, const int32_t* __restrict__ A, int32_t lda, uint64_t strideA, int64_t imAoffset, real_ptr C, int32_t ldc) {
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load_real;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load_real(temp_load_real);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  int32_t row_expon[ITEMS_PER_THREAD], c[ITEMS_PER_THREAD];
  real_t val[ITEMS_PER_THREAD];

  for (int32_t col = blockIdx.x; col < N; col += GRID_BLOCKS) {
    uint64_t A_col = uint64_t(col) * uint64_t(lda);
    real_ptr C_col = &C[uint64_t(col) * uint64_t(ldc)];
    int32_t col_expon = vec_expon[col] + k_begin;

    for (int32_t i = 0; i < N; i += elements) {
      uint64_t A_ij = uint64_t(i) + A_col;
      int32_t num_items = min(elements, N - i);
      block_load.Load(&vec_expon[i], row_expon, num_items);
      block_load_real.Load(&C_col[i], val, num_items);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        row_expon[j] = BASE * (row_expon[j] + col_expon);

      for (int32_t k = 0; k < k_len; ++k) {
        uint64_t A_k = A_ij + uint64_t(k) * strideA;

        block_load.Load(&A[A_k], *(int32_t(*)[ITEMS_PER_THREAD])(&c[0]), num_items);
        decode_i32_array(val, c, row_expon, BASE * k);
      }

      block_store.Store(&C_col[i], val, num_items);
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t items_per_thread = 4;

void internal::int8::decode_f64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, double* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <double, double* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, lda, strideA, 0, C, ldc);
}

void internal::int8::decode_f32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, float* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <float, float* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, lda, strideA, 0, C, ldc);
}

void internal::int8::decode_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, double2* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <double2, double2* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, lda, strideA, 0, C, ldc);
}

void internal::int8::decode_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, float4* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <float4, float4* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, lda, strideA, 0, C, ldc);
}

