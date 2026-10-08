
#include <hyacin.h>
#ifndef NO_NCCL

#include <internal.hpp>
#include <vector>
#include <numeric>
#include <stdexcept>

template <class matrix_t> inline void allgather_iter(cudaStream_t stream, cublasHandle_t handle, int32_t comm_rank, int32_t M, int32_t cols, std::vector<int32_t>& N, std::vector<int64_t>& iN, matrix_t* A, int64_t lda, matrix_t* W, ncclComm_t row_comm) {
  uint64_t stride = uint64_t(M) * uint64_t(cols); matrix_t* lW = &W[int64_t(comm_rank) * stride];
  if (0 < cols && 0 < N[comm_rank])
  { internal::scatter_matcopy(stream, handle, 'A', M, std::min(cols, N[comm_rank]), nullptr, &A[iN[comm_rank]], lda, lW, M); }
  ncclAllGather(lW, W, stride * uint64_t(sizeof(matrix_t)), ncclUint8, row_comm, stream);

  int32_t len = int32_t(iN.size());
  for (int32_t i = 0; i < len; ++i) {
    int32_t n = std::min(cols, N[i]);
    if (0 < n && i != comm_rank)
    { internal::scatter_matcopy(stream, handle, 'A', M, n, nullptr, W, M, &A[iN[i]], lda); }
    iN[i] += int64_t(n) * int64_t(lda); N[i] -= n; W = &W[stride];
  }
}

template <class matrix_t> inline int32_t allgather(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t* K, matrix_t* A, int32_t lda, ncclComm_t row_comm) {
  int32_t comm_rank, comm_size; ncclCommUserRank(row_comm, &comm_rank); ncclCommCount(row_comm, &comm_size);
  int32_t* dev_k = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_k, uint64_t(comm_size) * uint64_t(sizeof(int32_t)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at All-gather.");

  std::vector<int32_t> local_k(comm_size);
  cudaMemcpyAsync(&dev_k[comm_rank], K, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  ncclAllGather(&dev_k[comm_rank], dev_k, 1, ncclInt32, row_comm, stream);
  cudaMemcpyAsync(local_k.data(), dev_k, uint64_t(comm_size) * uint64_t(sizeof(int32_t)), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); cudaFreeAsync(dev_k, stream);

  int32_t j = std::reduce(local_k.begin(), local_k.begin() + comm_rank, 0);
  *K = std::reduce(local_k.begin() + comm_rank, local_k.end(), j);
  if (0 < M) {
    std::vector<int64_t> iN(comm_size);
    std::transform_exclusive_scan(local_k.begin(), local_k.end(), iN.begin(), int64_t(0), std::plus<int64_t>(), [=](int32_t n) { return int64_t(n) * int64_t(lda); });
    int32_t N = local_k[comm_rank], maxK = std::reduce(local_k.begin(), local_k.end(), 0, [](int32_t i, int32_t j) { return std::max(i, j); }), wcols = std::min(maxK, 2048);
    matrix_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, uint64_t(M) * uint64_t(wcols) * uint64_t(comm_size) * uint64_t(sizeof(matrix_t)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at All-gather.");

    if (0 < N && 0 < j) {
      if (N <= j) {
        internal::scatter_matcopy(stream, handle, 'A', M, N, nullptr, A, lda, &A[int64_t(j) * int64_t(lda)], lda);
      } else if (512 <= j) for (int32_t x = N; 0 < x; x -= 512) {
        int32_t x_start = std::max(x - 512, 0); int32_t cols = x - x_start;
        internal::scatter_matcopy(stream, handle, 'A', M, cols, nullptr, &A[int64_t(x_start) * int64_t(lda)], lda, &A[int64_t(x_start + j) * int64_t(lda)], lda);
      } else for (int32_t x = N; 0 < x; x -= wcols) {
        int32_t x_start = std::max(x - wcols, 0); int32_t cols = x - x_start;
        internal::scatter_matcopy(stream, handle, 'A', M, cols, nullptr, &A[int64_t(x_start) * int64_t(lda)], lda, W, M);
        internal::scatter_matcopy(stream, handle, 'A', M, cols, nullptr, W, M, &A[int64_t(x_start + j) * int64_t(lda)], lda);
      }
    }

    for (int32_t iter = maxK; iter > 0; iter -= wcols)
    { allgather_iter(stream, handle, comm_rank, M, std::min(iter, wcols), local_k, iN, A, lda, W, row_comm); }
    cudaFreeAsync(W, stream);
  }
  return j;
}

extern "C" int32_t hyacinXAllGatherV1Dcol(const hyacinHandle_t* handle, int32_t M, int32_t* K, hyacinPrecision_t Atype, void* A, int32_t lda) {
  if (handle->row_comm == nullptr) { return 0; }
  Timer::register_comm(handle->cudaStream, handle->timer);
  int32_t* kptr = (int32_t*)handle->pinnedWorkspace, offset_j = 0; *kptr = K ? *K : 0;
  switch (Atype) {
    case HYACIN_F64: offset_j = allgather(handle->cudaStream, handle->mempool, handle->cublasHandle, M, kptr, (double*)A, lda, handle->row_comm); break;
    case HYACIN_F32: offset_j = allgather(handle->cudaStream, handle->mempool, handle->cublasHandle, M, kptr, (float*)A, lda, handle->row_comm); break;
    case HYACIN_F16: offset_j = allgather(handle->cudaStream, handle->mempool, handle->cublasHandle, M, kptr, (__half*)A, lda, handle->row_comm); break;
    case HYACIN_F64_COMPLEX: offset_j = allgather(handle->cudaStream, handle->mempool, handle->cublasHandle, M, kptr, (cuDoubleComplex*)A, lda, handle->row_comm); break;
    case HYACIN_F32_COMPLEX: offset_j = allgather(handle->cudaStream, handle->mempool, handle->cublasHandle, M, kptr, (cuComplex*)A, lda, handle->row_comm); break;
    case HYACIN_F16_COMPLEX: offset_j = allgather(handle->cudaStream, handle->mempool, handle->cublasHandle, M, kptr, (__half2*)A, lda, handle->row_comm); break;
    default: break;
  }
  if (K) { *K = *kptr; } return offset_j;
}

#else
extern "C" int32_t hyacinXAllGatherV1Dcol(hyacinHandle_t, int32_t, int32_t* K, hyacinPrecision_t, void*, int32_t) { if (K) { *K = 0; } return 0; }
#endif
