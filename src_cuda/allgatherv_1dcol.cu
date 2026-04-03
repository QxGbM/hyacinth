
#include <hyacin.h>
#include <vector>
#include <numeric>

extern "C" void hyacinXAllGatherV1Dcol_bufferSize(int32_t M, int32_t comm_size, hyacinPrecision_t Atype, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  *pinned_work_bytes = std::max(*pinned_work_bytes, uint64_t(comm_size) * sizeof(int32_t));
  if (0 < M) {
    int32_t a_bytes; hyacinXelem('A', Atype, nullptr, &a_bytes, nullptr);
    *dev_work_bytes = std::max(*dev_work_bytes, uint64_t(M) * uint64_t(2048) * uint64_t(comm_size) * uint64_t(a_bytes));
  }
}

#ifndef NO_NCCL

__global__ void imatrix_copy(int64_t M, const int32_t* __restrict__ A, int64_t lda, int32_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (y < M) B[y + int64_t(blockIdx.y) * ldb] = A[y + int64_t(blockIdx.y) * lda];
}

inline void matrix_move(cudaStream_t stream, int64_t M, int32_t N, int32_t j, int32_t* A, int64_t lda, int32_t wcols, int32_t* W) {
  uint32_t grid_x = uint32_t((uint64_t(M) + uint64_t(511)) >> 9);
  if (N <= j)
    imatrix_copy <<< dim3(grid_x, uint32_t(N)), 512, 0, stream >>> (M, A, lda, &A[int64_t(j) * lda], lda);
  else if (512 <= j) for (int32_t x = N; 0 < x; x -= 512) {
    int32_t x_start = std::max(x - 512, 0); uint32_t cols = uint32_t(x - x_start);
    imatrix_copy <<< dim3(grid_x, cols), 512, 0, stream >>> (M, &A[int64_t(x_start) * lda], lda, &A[int64_t(x_start + j) * lda], lda);
  }
  else if (0 < j) for (int32_t x = N; 0 < x; x -= wcols) {
    int32_t x_start = std::max(x - wcols, 0); uint32_t cols = uint32_t(x - x_start);
    imatrix_copy <<< dim3(grid_x, cols), 512, 0, stream >>> (M, &A[int64_t(x_start) * lda], lda, W, M);
    imatrix_copy <<< dim3(grid_x, cols), 512, 0, stream >>> (M, W, M, &A[int64_t(x_start + j) * lda], lda);
  }
}

inline void allgather_iter(cudaStream_t stream, int32_t comm_rank, int64_t M, int32_t cols, int32_t* N, std::vector<int64_t>& iN, int32_t* A, int64_t lda, int32_t* W, ncclComm_t row_comm) {
  uint64_t stride = (uint64_t(M) * uint64_t(cols) + uint64_t(63)) & (~uint64_t(63));
  uint32_t grid_x = uint32_t((uint64_t(M) + uint64_t(511)) >> 9), grid_y = uint32_t(std::min(cols, N[comm_rank]));
  if (0 < grid_y)
    imatrix_copy <<< dim3(grid_x, grid_y), 512, 0, stream >>> (M, &A[iN[comm_rank]], lda, &W[int64_t(comm_rank) * stride], M);
  ncclAllGather(&W[int64_t(comm_rank) * stride], W, stride, ncclInt32, row_comm, stream);

  int32_t len = int32_t(iN.size());
  for (int32_t i = 0; i < len; ++i) {
    int32_t n = std::min(cols, N[i]);
    if (0 < n && i != comm_rank)
      imatrix_copy <<< dim3(grid_x, n), 512, 0, stream >>> (M, &W[int64_t(i) * stride], M, &A[iN[i]], lda);
    iN[i] += int64_t(n) * lda; N[i] -= n;
  }
}

extern "C" int32_t hyacinXAllGatherV1Dcol(cudaStream_t stream, int32_t M, int32_t* K, hyacinPrecision_t Atype, void* A, int32_t lda, uint64_t dev_work_bytes, void* dev_work, void* pinned_work, ncclComm_t row_comm) {
  int32_t a_bytes; hyacinXelem('A', Atype, nullptr, &a_bytes, nullptr);
  int64_t Mi = int64_t(M) * int64_t(a_bytes / int32_t(sizeof(int32_t))), LDAi = int64_t(lda) * int64_t(a_bytes / int32_t(sizeof(int32_t)));

  int32_t comm_rank, comm_size; ncclCommUserRank(row_comm, &comm_rank); ncclCommCount(row_comm, &comm_size);
  int32_t* N = (int32_t*)pinned_work; N[comm_rank] = *K;
  ncclAllGather(&N[comm_rank], N, 1, ncclInt32, row_comm, stream);
  cudaStreamSynchronize(stream);
  int32_t offset_j = std::reduce(N, &N[comm_rank], 0); 
  *K = std::reduce(&N[comm_rank], &N[comm_size], offset_j);
  if (M <= 0) { return 0; }

  std::vector<int64_t> iN(comm_size);
  std::transform_exclusive_scan(N, &N[comm_size], iN.begin(), int64_t(0), std::plus<int64_t>(), [=](int32_t n) { return int64_t(n) * LDAi; });

  int32_t* Aptr = (int32_t*)A, *Wptr = (int32_t*)dev_work;
  int32_t wcols = int32_t(dev_work_bytes / uint64_t(Mi * sizeof(int32_t))) & (~63);
  if (0 < N[comm_rank])
    matrix_move(stream, Mi, N[comm_rank], offset_j, Aptr, LDAi, wcols, Wptr);

  int32_t maxK = std::reduce(N, &N[comm_size], 0, [](int32_t i, int32_t j) { return std::max(i, j); });
  for (int32_t iter = maxK; iter > 0; iter -= wcols)
    allgather_iter(stream, comm_rank, Mi, std::min(iter, wcols), N, iN, Aptr, LDAi, Wptr, row_comm);
  return offset_j;
}

#endif
