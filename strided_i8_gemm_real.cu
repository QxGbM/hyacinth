
#include <internal.hpp>

#include <cub/cub.cuh>
#include <algorithm>

template <int32_t BASE, int32_t ITEMS_PER_THREAD>
__device__ void signed_normalize(int32_t (&a)[ITEMS_PER_THREAD], int32_t (&c)[ITEMS_PER_THREAD]) {
  constexpr uint32_t iBASE = (uint32_t(1) << BASE) - 1;
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    int32_t val = a[i] + c[i];
    int32_t sign = val >> 31;
    c[i] = (val >> BASE) - sign;
    a[i] = (val & iBASE) | (sign & ~iBASE);
  }
}

template <int32_t ITEMS_PER_THREAD>
__device__ void signed_sum(int32_t (&a)[ITEMS_PER_THREAD], int32_t const (&c)[ITEMS_PER_THREAD]) {
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    a[i] += c[i];
}

template <int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE>
__global__ void normalize_kernel(uint64_t M, int32_t N, int32_t* __restrict__ A, uint64_t lda) {
  constexpr uint64_t elements_block = ITEMS_PER_THREAD * BLOCK_THREADS;
  constexpr uint64_t elements = GRID_BLOCKS * elements_block;
  uint64_t block_offset = blockIdx.x * elements_block;
  uint64_t M2 = M & (elements_block - 1), M1 = M - M2;
  uint64_t iter_k = uint64_t(N) * lda;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  int32_t a[ITEMS_PER_THREAD];

  for (uint64_t row = block_offset; row < M1; row += elements) {
    int32_t* A_row = &A[row];
    int32_t c[ITEMS_PER_THREAD]{};

    for (uint64_t k = 0; k < iter_k; k += lda) {
      block_load.Load(&A_row[k], a);
      signed_normalize<BASE>(a, c);
      block_store.Store(&A_row[k], a);
    }

    block_load.Load(&A_row[iter_k], a);
    signed_sum(a, c);
    block_store.Store(&A_row[iter_k], a);
  }

  if (0 < M2 && blockIdx.x == 0) {
    int32_t* A_row = &A[M1];
    int32_t c[ITEMS_PER_THREAD]{};

    for (uint64_t k = 0; k < iter_k; k += lda) {
      block_load.Load(&A_row[k], a, M2, 0);
      signed_normalize<BASE>(a, c);
      block_store.Store(&A_row[k], a, M2);
    }

    block_load.Load(&A_row[iter_k], a, M2);
    signed_sum(a, c);
    block_store.Store(&A_row[iter_k], a, M2);
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t items_per_thread = 4;

void internal::int8::r8i_TN_gemm_stridedA(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t iter_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t* C, int32_t orderC) {
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  int32_t order_begin = orderC - 2 * orderA;
  int32_t one = 1;
  
  for (int32_t i = 0; i < orderA; ++i) {
    std::pair<int32_t, int32_t> min_max = std::minmax(order_begin + i, 0);
    int32_t order_i = orderA + min_max.first;
    int32_t* C_i = &C[uint64_t(min_max.second) * strideC];

    for (int32_t k = 0; k < algnK; k += iter_k) {
      const int8_t* AT_k = &AT[uint64_t(k) + uint64_t(i) * strideA];
      const int8_t* AN_k = &A[uint64_t(k) + uint64_t(orderA - order_i) * strideA];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, C_i, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      normalize_kernel <grid_blocks, block_threads, items_per_thread, exp_base>
        <<< grid_blocks, block_threads, 0, stream >>> (strideC, order_i, C_i, strideC);
    }
  }
}

