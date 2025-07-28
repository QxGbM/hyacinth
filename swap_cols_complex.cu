
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::conj(f); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::conj(f); }
};

template <class complex_t, class complex_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_cols_complex(int32_t i, int32_t j, int32_t N, complex_ptr A, int32_t lda) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;
  int32_t thread_offset = threadIdx.x * ITEMS_PER_THREAD;
  int32_t rem = N & (elements_block - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  complex_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  complex_ptr A_col_i = &A[i * lda], A_col_j = &A[j * lda], A_row_j = &A[j];

  cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  conj conj_func;

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A_col_i[k], thread_i);
    block_load.Load(&A_col_j[k], thread_j);

    block_store.Store(&A_col_j[k], thread_i);
    block_store.Store(&A_col_i[k], thread_j);

    int32_t thread_loc = thread_offset + k;
    int32_t row_begin = max(thread_loc, i + 1);
    int32_t row_end = min(thread_loc + ITEMS_PER_THREAD, N);

    #pragma unroll
    for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
      thread_j[l] = conj_func(thread_i[l]);

    for (int32_t l = row_begin; l < row_end; ++l)
      A_row_j[l * lda] = thread_j[l - thread_loc];
  }

  if (0 < N2 && blockIdx.x == 0) {
    block_load.Load(&A_col_i[N1], thread_i, N2);
    block_load.Load(&A_col_j[N1], thread_j, N2);

    block_store.Store(&A_col_j[N1], thread_i, N2);
    block_store.Store(&A_col_i[N1], thread_j, N2);

    int32_t thread_loc = thread_offset + N1;
    int32_t row_begin = max(thread_loc, i + 1);
    int32_t row_end = min(thread_loc + ITEMS_PER_THREAD, N);

    #pragma unroll
    for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
      thread_j[l] = conj_func(thread_i[l]);

    for (int32_t l = row_begin; l < row_end; ++l)
      A_row_j[l * lda] = thread_j[l - thread_loc];
  }
}

constexpr int32_t grid_blocks = 64;
constexpr int32_t block_threads = 8 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  swap_cols_complex <cuDoubleComplex, cuDoubleComplex* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, (cuDoubleComplex*)A, lda);
}

void internal::Cholesky::swap_cols_float_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<float>* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  swap_cols_complex <cuComplex, cuComplex* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, (cuComplex*)A, lda);
}

void internal::Cholesky::swap_cols_double2_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_double2* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  swap_cols_complex <complex_double2, complex_double2* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_float4_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_float4* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  swap_cols_complex <complex_float4, complex_float4* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (i, j, N, A, lda);
}
