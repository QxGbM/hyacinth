
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>

template <class real_t, class real_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_cols_real(int32_t i, int32_t j, int32_t N, real_ptr A, int32_t lda) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;
  int32_t thread_offset = threadIdx.x * ITEMS_PER_THREAD;
  int32_t N2 = N & (elements - 1), N1 = N - N2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  real_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  real_ptr A_col_i = &A[i * lda], A_col_j = &A[j * lda], A_row_j = &A[j];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A_col_i[k], thread_i);
    block_load.Load(&A_col_j[k], thread_j);

    block_store.Store(&A_col_j[k], thread_i);
    block_store.Store(&A_col_i[k], thread_j);

    int32_t thread_loc = thread_offset + k;
    int32_t row_begin = max(thread_loc, i + 1);
    int32_t row_end = min(thread_loc + ITEMS_PER_THREAD, N);

    for (int32_t l = row_begin; l < row_end; ++l)
      A_row_j[l * lda] = thread_i[l - thread_loc];
  }

  if (0 < N2 && blockIdx.x == 0) {
    block_load.Load(&A_col_i[N1], thread_i, N2);
    block_load.Load(&A_col_j[N1], thread_j, N2);

    block_store.Store(&A_col_j[N1], thread_i, N2);
    block_store.Store(&A_col_i[N1], thread_j, N2);

    int32_t thread_loc = thread_offset + N1;
    int32_t row_begin = max(thread_loc, i + 1);
    int32_t row_end = min(thread_loc + ITEMS_PER_THREAD, N);

    for (int32_t l = row_begin; l < row_end; ++l)
      A_row_j[l * lda] = thread_i[l - thread_loc];
  }
}

constexpr int32_t grid_blocks = 64;
constexpr int32_t block_threads = 8 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  swap_cols_real <double, double* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_float(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  swap_cols_real <float, float* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_double2(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double2* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  swap_cols_real <double2, double2* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_float4(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float4* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  swap_cols_real <float4, float4* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, A, lda);
}
