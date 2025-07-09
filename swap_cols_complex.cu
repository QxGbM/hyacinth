
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

template <class complex_t, class complex_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_cols_complex(int32_t i, int32_t j, int32_t N, complex_ptr A, int32_t lda) {
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  complex_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  complex_ptr A_col_i = &A[i * lda], A_col_j = &A[j * lda], A_row_j = &A[j];

  cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  conj conj_func;

  for (int32_t k = 0; k < N; k += elements) {
    int32_t num_items = min(elements, N - k);
    block_load.Load(&A_col_i[k], thread_i, num_items);
    block_load.Load(&A_col_j[k], thread_j, num_items);

    int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD + k;
    int32_t row_begin = max(thread_loc, i + 1);
    int32_t row_end = min(thread_loc + ITEMS_PER_THREAD, N);

    block_store.Store(&A_col_j[k], thread_i, num_items);
    block_store.Store(&A_col_i[k], thread_j, num_items);

    for (int32_t l = row_begin; l < row_end; ++l)
      A_row_j[l * lda] = conj_func(thread_i[l - thread_loc]);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    complex_t A_ii = A_col_i[i];
    complex_t A_ji = A_col_i[j]; 
    complex_t A_ij = A_col_j[i];

    A_col_i[i] = A_ji;
    A_col_i[j] = A_ii;
    A_col_j[i] = conj_func(A_ii);
    A_col_j[j] = A_ij;
  }
}

constexpr int32_t block_threads = 8 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  swap_cols_complex <cuDoubleComplex, cuDoubleComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, (cuDoubleComplex*)A, lda);
}

void internal::Cholesky::swap_cols_float_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<float>* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  swap_cols_complex <cuComplex, cuComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, (cuComplex*)A, lda);
}

void internal::Cholesky::swap_cols_double2_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_double2* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  swap_cols_complex <complex_double2, complex_double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, A, lda);
}

void internal::Cholesky::swap_cols_float4_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_float4* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  swap_cols_complex <complex_float4, complex_float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (i, j, N, A, lda);
}
