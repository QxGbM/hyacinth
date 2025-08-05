
#include <internal.hpp>

#include <cub/cub.cuh>

constexpr int32_t grid_x = 16;
constexpr int32_t block_warps = 8;

__global__ void batch_subtract_transpose_kernel(int32_t N, int32_t* __restrict__ A, int32_t lda, uint64_t strideA) {
  __shared__ typename cub::WarpLoad<int32_t, 2>::TempStorage temp_load[block_warps];
  __shared__ typename cub::WarpStore<int32_t, 2>::TempStorage temp_store[block_warps];
  __shared__ int32_t tileA[64][65];
  int32_t iter_n = (N + 63) >> 6, threadA[2];
  int32_t i1 = threadIdx.x << 1, i2 = (threadIdx.x << 1) + 1;
  A = &A[uint64_t(blockIdx.z) * strideA];

  cub::WarpLoad<int32_t, 2> warp_load(temp_load[threadIdx.y]);
  cub::WarpStore<int32_t, 2> warp_store(temp_store[threadIdx.y]);

  for (int32_t col = blockIdx.y; col < iter_n; col += grid_x) {
    int32_t num_cols = min(64, N - (col << 6));

    if (blockIdx.x == 0) { // diagonal
      uint64_t A_ii = (uint64_t(col) * uint64_t(lda + 1)) << 6;
      for (int32_t k = threadIdx.y; k < num_cols; k += block_warps) {
        warp_load.Load(&A[A_ii + uint64_t(lda) * uint64_t(k)], threadA, num_cols, 0);
        tileA[k][i1] = threadA[0];
        tileA[k][i2] = threadA[1];
      }
      __syncthreads();

      for (int32_t k = threadIdx.y; k < num_cols; k += block_warps) {
        threadA[0] = tileA[k][i1] - tileA[i1][k];
        threadA[1] = tileA[k][i2] - tileA[i2][k];
        warp_store.Store(&A[A_ii + uint64_t(lda) * uint64_t(k)], threadA, num_cols);
      }
      __syncthreads();
    }

    for (int32_t row = (col + blockIdx.x + 1); row < iter_n; row += grid_x) { // lower triangle
      int32_t num_rows = min(64, N - (row << 6));
      uint64_t A_ij = (uint64_t(row) + uint64_t(col) * uint64_t(lda)) << 6;
      uint64_t A_ji = (uint64_t(col) + uint64_t(row) * uint64_t(lda)) << 6;

      for (int32_t k = threadIdx.y; k < num_cols; k += block_warps) {
        warp_load.Load(&A[A_ij + uint64_t(lda) * uint64_t(k)], threadA, num_rows, 0);
        tileA[k][i1] = threadA[0];
        tileA[k][i2] = threadA[1];
      }
      __syncthreads();

      for (int32_t k = threadIdx.y; k < num_rows; k += block_warps) {
        int32_t* U = &A[A_ji + uint64_t(lda) * uint64_t(k)];
        warp_load.Load(U, threadA, num_cols, 0);
        threadA[0] = threadA[0] - tileA[i1][k];
        threadA[1] = threadA[1] - tileA[i2][k];

        warp_store.Store(U, threadA, num_cols);
        tileA[i1][k] = -threadA[0];
        tileA[i2][k] = -threadA[1];
      }
      __syncthreads();

      for (int32_t k = threadIdx.y; k < num_cols; k += block_warps) {
        threadA[0] = tileA[k][i1];
        threadA[1] = tileA[k][i2];
        warp_store.Store(&A[A_ij + uint64_t(lda) * uint64_t(k)], threadA, num_rows);
      }
    }
  }
}

void internal::int8::c8i_HN_gemm_stridedA(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t iter_k, int32_t algnN, int32_t algnK, const int8_t* AH, int32_t orderA, int32_t* C, int32_t orderC) {
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  const int8_t* A_im = &AH[uint64_t(orderA) * strideA];
  int32_t* C_im = &C[uint64_t(orderC) * strideC];

  r8i_TN_gemm_stridedA(stream, handle, N, iter_k, algnN, algnK, AH, AH, orderA, C, orderC);
  r8i_TN_gemm_stridedA(stream, handle, N, iter_k, algnN, algnK, A_im, A_im, orderA, C, orderC);
  r8i_TN_gemm_stridedA(stream, handle, N, iter_k, algnN, algnK, AH, A_im, orderA, C_im, orderC);
  batch_subtract_transpose_kernel <<< dim3(grid_x, grid_x, orderC), dim3(32, block_warps, 1), 0, stream >>> (N, C_im, algnN, strideC);
}
