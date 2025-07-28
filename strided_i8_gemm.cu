
#include <internal.hpp>

#include <cub/cub.cuh>

template <int32_t GRID_X, int32_t GRID_Y, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_negate_i32_matrix(int32_t M, int32_t N, int32_t* __restrict__ A) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_Y * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;
  int32_t rem = M & (elements_block - 1), div = M - rem;
  int32_t M1 = max(div, rem), M2 = min(div, rem);

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  int32_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);

  for (int32_t col = (blockIdx.y << 1); col < N; col += (GRID_X << 1)) {
    int32_t* A_col_i = &A[col * M];
    int32_t* A_col_j = &A[(col + 1) * M];

    for (int32_t k = block_offset; k < M1; k += elements) {
      block_load.Load(&A_col_i[k], thread_i);
      block_load.Load(&A_col_j[k], thread_j);

      #pragma unroll
      for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
        thread_j[l] = -thread_j[l];
      block_store.Store(&A_col_j[k], thread_i);
      block_store.Store(&A_col_i[k], thread_j);
    }

    if (0 < M2 && blockIdx.x == 0) {
      block_load.Load(&A_col_i[M1], thread_i, M2);
      block_load.Load(&A_col_j[M1], thread_j, M2);

      #pragma unroll
      for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
        thread_j[l] = -thread_j[l];
      block_store.Store(&A_col_j[M1], thread_i, M2);
      block_store.Store(&A_col_i[M1], thread_j, M2);
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_x = 64;
constexpr int32_t grid_y = 16;
constexpr int32_t items_per_thread = 4;

void internal::int8::strided_r8i_ATA_gemm(cublasHandle_t handle, int32_t order, int32_t algnM, int32_t algnN, const int8_t* A, int32_t* C) {
  int32_t strideA = algnM * algnN, strideC = algnN * algnN;
  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);
  cudaMemsetAsync(C, 0, strideC * (2 * order - 1) * sizeof(int32_t), stream);

  int32_t one = 1;
  for (int32_t i = 0; i < order; ++i) {
    const int8_t* AT = &A[i * strideA];
    int32_t* C_i = &C[i * strideC];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, algnN * order, algnM, &one, 
      AT, CUDA_R_8I, algnM, A, CUDA_R_8I, algnM, &one, C_i, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }
}

void internal::int8::strided_c8i_AHA_gemm(cublasHandle_t handle, int32_t order, int32_t algnM, int32_t algnN, const int8_t* A, int32_t* C) {
  int32_t strideA = algnM * algnN, strideC = algnN * algnN;
  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);
  cudaMemsetAsync(C, 0, 2 * strideC * (2 * order - 1) * sizeof(int32_t), stream);

  int32_t one = 1;
  for (int32_t i = 0; i < order; ++i) {
    const int8_t* iAH = &A[(i * 2) * strideA];
    int32_t* C_i = &C[(i * 2) * strideC];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, algnN * order * 2, algnM, &one, 
      iAH, CUDA_R_8I, algnM, A, CUDA_R_8I, algnM, &one, C_i, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }

  swap_negate_i32_matrix <grid_x, grid_y, block_threads, items_per_thread>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (strideC, 4 * order - 2, C);

  for (int32_t i = 0; i < order; ++i) {
    const int8_t* rAH = &A[(i * 2 + 1) * strideA];
    int32_t* C_i = &C[(i * 2) * strideC];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, algnN * order * 2, algnM, &one, 
      rAH, CUDA_R_8I, algnM, A, CUDA_R_8I, algnM, &one, C_i, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }
}
