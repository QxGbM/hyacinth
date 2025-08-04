
#include <internal.hpp>

#include <cub/cub.cuh>
#include <algorithm>

template <int32_t GRID_X, int32_t GRID_Y, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_negate_i32_matrix(int32_t M, int32_t N, int32_t* __restrict__ A) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_Y * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;
  int32_t M2 = M & (elements_block - 1), M1 = M - M2;

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

void internal::int8::r8i_TN_gemm_stridedA(cudaStream_t stream, cublasHandle_t handle, int32_t iter_k, int32_t algnM, int32_t algnN, int32_t algnK, const int8_t* AT, int32_t strideA, int32_t orderA, const int8_t* B, int32_t* C, int32_t orderC) {
  int32_t strideC = algnM * algnN, strideB = algnK * algnN;
  int32_t order_begin = orderC - 2 * orderA;
  int32_t one = 1;
  
  for (int32_t i = 0; i < orderA; ++i) {
    std::pair<int32_t, int32_t> min_max = std::minmax(order_begin + i, 0);
    int32_t order_i = orderA + min_max.first;
    int32_t* C_i = &C[min_max.second * strideC];

    for (int32_t k = 0; k < algnK; k += iter_k) {
      const int8_t* AT_k = &AT[k + i * strideA];
      const int8_t* BN_k = &B[k + (orderA - order_i) * strideB];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnM, algnN * order_i, std::min(algnK - k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, BN_k, CUDA_R_8I, algnK, &one, C_i, CUDA_R_32I, algnM, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::normalize_i32(stream, strideC, order_i, C_i);
    }
  }
}

void internal::int8::c8i_HN_gemm_stridedA(cudaStream_t stream, cublasHandle_t handle, int32_t iter_k, int32_t algnM, int32_t algnN, int32_t algnK, const int8_t* AH, int32_t strideA, int32_t orderA, const int8_t* B, int32_t* C, int32_t orderC) {
  r8i_TN_gemm_stridedA(stream, handle, iter_k, algnM, 2 * algnN, algnK, AH, 2 * strideA, orderA, B, C, orderC);
  swap_negate_i32_matrix <grid_x, grid_y, block_threads, items_per_thread>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (algnM * algnN, 2 * orderC, C);
  r8i_TN_gemm_stridedA(stream, handle, iter_k, algnM, 2 * algnN, algnK, &AH[strideA], 2 * strideA, orderA, B, C, orderC);
}
