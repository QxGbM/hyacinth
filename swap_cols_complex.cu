
#include <hyacinth.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>
#include <float4.hpp>

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::f4::conj(f); }
};

template <class complex_t, class complex_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_cols_complex(int32_t i, int32_t j, int32_t N, complex_ptr A, int32_t lda) {
  using BlockLoad = cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockStore = cub::BlockStore<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename BlockLoad::TempStorage temp_load_i, temp_load_j;
  __shared__ typename BlockStore::TempStorage temp_store_i, temp_store_j;

  complex_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  complex_ptr A_col_i = &A[i * lda], A_col_j = &A[j * lda], A_row_j = &A[j];

  conj conj_func;

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
      A_row_j[l * lda] = conj_func(thread_i[l - thread_loc]);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    complex_t A_ii = A_col_i[i];
    complex_t A_ji = A_col_i[j]; 
    complex_t A_ij = A_col_j[i];
    complex_t A_jj = A_col_j[j];

    A_col_i[j] = A_ii;
    A_col_i[i] = A_ji;
    A_col_j[j] = A_ij;
    A_col_j[i] = A_jj;
  }
}

void swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t block_threads = 8 * 32;
  constexpr int32_t items_per_thread = 4;
  swap_cols_complex <cuDoubleComplex, cuDoubleComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, (cuDoubleComplex*)A, lda);
}
