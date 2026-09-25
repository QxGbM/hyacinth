
#include <hyacin.h>
#ifndef NO_NCCL

#include <internal.hpp>
#include <numeric>
#include <stdexcept>

inline void matrix_move(cudaStream_t stream, int64_t M, int32_t N, int32_t j, uint8_t* A, int64_t lda, int32_t wcols, uint8_t* W) {
  if (N <= j) { cudaMemcpy2DAsync(&A[int64_t(j) * lda], lda, A, lda, M, N, cudaMemcpyDeviceToDevice, stream); }
  else if (512 <= j) for (int32_t x = N; 0 < x; x -= 512) {
    int32_t x_start = std::max(x - 512, 0); int32_t cols = x - x_start;
    cudaMemcpy2DAsync(&A[int64_t(x_start + j) * lda], lda, &A[int64_t(x_start) * lda], lda, M, cols, cudaMemcpyDeviceToDevice, stream);
  }
  else if (0 < j) for (int32_t x = N; 0 < x; x -= wcols) {
    int32_t x_start = std::max(x - wcols, 0); int32_t cols = x - x_start;
    cudaMemcpy2DAsync(W, M, &A[int64_t(x_start) * lda], lda, M, cols, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&A[int64_t(x_start + j) * lda], lda, W, M, M, cols, cudaMemcpyDeviceToDevice, stream);
  }
}

inline void allgather_iter(cudaStream_t stream, int32_t comm_rank, int64_t M, int32_t cols, std::vector<int32_t>& N, std::vector<int64_t>& iN, uint8_t* A, int64_t lda, uint8_t* W, ncclComm_t row_comm) {
  uint64_t stride = (uint64_t(M) * uint64_t(cols) + uint64_t(63)) & (~uint64_t(63));
  uint8_t* lW = &W[int64_t(comm_rank) * stride];
  if (0 < cols && 0 < N[comm_rank])
  { cudaMemcpy2DAsync(lW, M, &A[iN[comm_rank]], lda, M, std::min(cols, N[comm_rank]), cudaMemcpyDeviceToDevice, stream); }
  ncclAllGather(lW, W, stride, ncclUint8, row_comm, stream);

  int32_t len = int32_t(iN.size());
  for (int32_t i = 0; i < len; ++i) {
    int32_t n = std::min(cols, N[i]);
    if (0 < n && i != comm_rank)
    { cudaMemcpy2DAsync(&A[iN[i]], lda, W, M, M, n, cudaMemcpyDeviceToDevice, stream); }
    iN[i] += int64_t(n) * lda; N[i] -= n; W = &W[stride];
  }
}

extern "C" int32_t hyacinXAllGatherV1Dcol(hyacinHandle_t handle, int32_t M, int32_t* K, int32_t AElemBytes, void* A, int32_t lda) {
  if (handle.row_comm == nullptr) { return 0; }
  const int32_t wcols = 2048;
  int64_t Mi = int64_t(M) * int64_t(AElemBytes), LDAi = int64_t(lda) * int64_t(AElemBytes);

  Timer::register_comm(handle.cudaStream, handle.timer);
  int32_t comm_rank, comm_size, hK = K ? *K : 0; ncclCommUserRank(handle.row_comm, &comm_rank); ncclCommCount(handle.row_comm, &comm_size);
  uint64_t work_bytes = std::max(uint64_t(Mi) * uint64_t(wcols), uint64_t(sizeof(int32_t))) * uint64_t(comm_size);
  uint8_t* dev_k = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_k, work_bytes, handle.mempool, handle.cudaStream))
  { throw std::runtime_error("Workspace allocation failed at All-gather."); }

  uint8_t* lk = &dev_k[int64_t(comm_rank) * sizeof(int32_t)];
  std::vector<int32_t> local_k(comm_size);
  cudaMemcpyAsync(lk, &hK, sizeof(int32_t), cudaMemcpyHostToDevice, handle.cudaStream);
  ncclAllGather(lk, dev_k, 1, ncclInt32, handle.row_comm, handle.cudaStream);
  cudaMemcpyAsync(local_k.data(), dev_k, uint64_t(comm_size) * uint64_t(sizeof(int32_t)), cudaMemcpyDeviceToHost, handle.cudaStream);

  int32_t offset_j = std::reduce(local_k.begin(), local_k.begin() + comm_rank, 0);
  if (K) { *K = std::reduce(local_k.begin() + comm_rank, local_k.end(), offset_j); }
  if (Mi) {
    std::vector<int64_t> iN(comm_size);
    std::transform_exclusive_scan(local_k.begin(), local_k.end(), iN.begin(), int64_t(0), std::plus<int64_t>(), [=](int32_t n) { return int64_t(n) * LDAi; });

    uint8_t* Aptr = (uint8_t*)A;
    if (0 < local_k[comm_rank])
    { matrix_move(handle.cudaStream, Mi, local_k[comm_rank], offset_j, Aptr, LDAi, wcols, dev_k); }

    int32_t maxK = std::reduce(local_k.begin(), local_k.end(), 0, [](int32_t i, int32_t j) { return std::max(i, j); });
    for (int32_t iter = maxK; iter > 0; iter -= wcols)
    { allgather_iter(handle.cudaStream, comm_rank, Mi, std::min(iter, wcols), local_k, iN, Aptr, LDAi, dev_k, handle.row_comm); }
  }
  cudaFreeAsync(dev_k, handle.cudaStream);
  return offset_j;
}

#else
extern "C" int32_t hyacinXAllGatherV1Dcol(hyacinHandle_t, int32_t, int32_t*, int32_t, void*, int32_t) { return 0; }
#endif
