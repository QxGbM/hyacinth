
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>

template <class real_t, class real_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_cols_real(int32_t i, int32_t j, int32_t N, real_ptr A, int32_t lda) {
  using BlockLoad = cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockStore = cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename BlockLoad::TempStorage temp_load_i, temp_load_j;
  __shared__ typename BlockStore::TempStorage temp_store_i, temp_store_j;

  real_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  real_ptr A_col_i = &A[i * lda], A_col_j = &A[j * lda], A_row_j = &A[j];

  for (int32_t k = 0; k < N; k += elements) {
    int32_t num_items = min(elements, N - k);
    BlockLoad(temp_load_i).Load(&A_col_i[k], thread_i, num_items);
    BlockLoad(temp_load_j).Load(&A_col_j[k], thread_j, num_items);

    int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD + k;
    int32_t row_begin = max(thread_loc, i + 1);
    int32_t row_end = min(thread_loc + ITEMS_PER_THREAD, N);

    BlockStore(temp_store_j).Store(&A_col_j[k], thread_i, num_items);
    BlockStore(temp_store_i).Store(&A_col_i[k], thread_j, num_items);

    for (int32_t l = row_begin; l < row_end; ++l)
      A_row_j[l * lda] = thread_i[l - thread_loc];
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    real_t A_ii = A_col_i[i];
    real_t A_ji = A_col_i[j]; 
    real_t A_ij = A_col_j[i];

    A_col_i[i] = A_ji;
    A_col_i[j] = A_ii;
    A_col_j[i] = A_ii;
    A_col_j[j] = A_ij;
  }
}

constexpr int32_t block_threads = 8 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  swap_cols_real <double, double* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_float(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  swap_cols_real <float, float* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_double2(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double2* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  swap_cols_real <double2, double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_float4(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float4* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  swap_cols_real <float4, float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, A, lda);
}
