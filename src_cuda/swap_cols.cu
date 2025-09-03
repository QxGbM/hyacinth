
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::make_complex_double2(f.real, device::dd::negate(f.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::make_complex_float4(f.real, device::qf::negate(f.imag)); }
};

template <int32_t COMPLEX, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void pred_conj(matrix_t (&a)[ITEMS_PER_THREAD], matrix_t (&b)[ITEMS_PER_THREAD]) {
  if constexpr (COMPLEX) {
    conj conj_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    { a[i] = conj_func(a[i]); b[i] = conj_func(b[i]); }
  }
}

template <class matrix_t, class matrix_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t COMPLEX>
__global__ void swap_cols_kernel(int32_t j, int32_t M, int32_t N, matrix_ptr A, int64_t lda) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  constexpr int32_t thread_mask = BLOCK_THREADS - 1;
  constexpr int32_t block_mask = ~(elements_block - 1) & (elements - 1);

  int32_t col_j = j - M, block_offset = int32_t(blockIdx.x) * elements_block;
  int32_t M2 = (M + N) & (elements_block - 1), M1 = (M + N) - M2;

  __shared__ typename cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store;
  __shared__ matrix_t Aij[2];
  matrix_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  matrix_ptr A_col_j = &A[uint64_t(col_j) * lda], A_row_i = &A[M], A_row_j = &A[j];
  int32_t thread_locs[ITEMS_PER_THREAD];

  cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load);
  cub::BlockStore<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store(temp_store);
  if (threadIdx.x == 0)
    if constexpr(COMPLEX)
    { conj conj_func; Aij[0] = conj_func(A_col_j[M]); Aij[1] = conj_func(A_row_j[0]); }
    else
    { Aij[0] = A_col_j[M]; Aij[1] = A_row_j[0]; }

  #pragma unroll
  for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
    thread_locs[l] = (int32_t(threadIdx.x) - M) + l * BLOCK_THREADS;
  __syncthreads();

  for (int32_t k = block_offset; k < M1; k += elements) {
    block_load.Load(&A[k], thread_i);
    block_load.Load(&A_col_j[k], thread_j);

    block_store.Store(&A_col_j[k], thread_i);
    block_store.Store(&A[k], thread_j);
    pred_conj<COMPLEX>(thread_i, thread_j);

    #pragma unroll
    for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l) {
      int32_t col = k + thread_locs[l];
      if (1 <= col && col != col_j) {
        int64_t col_idx = uint64_t(col) * lda;
        A_row_i[col_idx] = thread_j[l];
        A_row_j[col_idx] = thread_i[l];
      }
    }
  }

  if (0 < M2 && block_offset == (M1 & block_mask)) {
    block_load.Load(&A[M1], thread_i, M2);
    block_load.Load(&A_col_j[M1], thread_j, M2);

    block_store.Store(&A_col_j[M1], thread_i, M2);
    block_store.Store(&A[M1], thread_j, M2);
    pred_conj<COMPLEX>(thread_i, thread_j);

    #pragma unroll
    for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l) {
      int32_t col = M1 + thread_locs[l];
      if (1 <= col && col != col_j && col < N) {
        int64_t col_idx = uint64_t(col) * lda;
        A_row_i[col_idx] = thread_j[l];
        A_row_j[col_idx] = thread_i[l];
      }
    }
  }

  if (threadIdx.x == (M & thread_mask) && block_offset == (M & block_mask))
    A_col_j[M] = Aij[0];
  if (threadIdx.x == (j & thread_mask) && block_offset == (j & block_mask))
    A_row_j[0] = Aij[1];
}

constexpr int32_t grid_blocks = 128;
constexpr int32_t block_threads = 128;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::swap_cols_f64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  swap_cols_kernel <double, double* __restrict__, grid_blocks, block_threads, items_per_thread, 0>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, A, lda);
}

void internal::Cholesky::swap_cols_f32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  swap_cols_kernel <float, float* __restrict__, grid_blocks, block_threads, items_per_thread, 0>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, A, lda);
}

void internal::Cholesky::swap_cols_f128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  swap_cols_kernel <double2, double2* __restrict__, grid_blocks, block_threads, items_per_thread, 0>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, A, lda);
}

void internal::Cholesky::swap_cols_f128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  swap_cols_kernel <float4, float4* __restrict__, grid_blocks, block_threads, items_per_thread, 0>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, A, lda);
}

void internal::Cholesky::swap_cols_cf64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  swap_cols_kernel <cuDoubleComplex, cuDoubleComplex* __restrict__, grid_blocks, block_threads, items_per_thread, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, (cuDoubleComplex*)A, lda);
}

void internal::Cholesky::swap_cols_cf32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  swap_cols_kernel <cuComplex, cuComplex* __restrict__, grid_blocks, block_threads, items_per_thread, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, (cuComplex*)A, lda);
}

void internal::Cholesky::swap_cols_cf128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  swap_cols_kernel <complex_double2, complex_double2* __restrict__, grid_blocks, block_threads, items_per_thread, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, A, lda);
}

void internal::Cholesky::swap_cols_cf128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  swap_cols_kernel <complex_float4, complex_float4* __restrict__, grid_blocks, block_threads, items_per_thread, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (j + M, M, N, A, lda);
}
