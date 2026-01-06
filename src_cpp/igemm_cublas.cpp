
#include <internal.hpp>
#include <crt_selector.hpp>

inline void gemm_accumulate(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft_lo, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor(stream, op, strideC, sft_lo, orderA, workspace, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = op | 1;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, strideC, sft_lo, orderA, workspace, orderC, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, std::min(rem, iter_h), &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, strideC, sft_lo, orderA, workspace, orderC, C);
    if (iter_h < rem) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor(stream, op_acc, strideC, sft_lo, orderA, workspace, orderC, C);
    }
  }
}

inline void gemm_accumulate_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor(stream, op, strideC, 0, orderA, workspace, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = 5;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, strideC, 0, orderA, workspace, orderC, C);
    }

    const int8_t* A_k = &A[int64_t(range_k)];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, std::min(rem, iter_h), &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, strideC, 0, orderA, workspace, orderC, C);
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor(stream, op_acc, strideC, 0, orderA, workspace, orderC, C);
    }
  }
}

void internal::int8::i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i) {
    int64_t AT_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, i, orderA, &AT[AT_i], A, i == 0 ? beta : 1, C, orderC, workspace);
  }
}

void internal::int8::i8GemmT(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  gemm_accumulate_diag(stream, handle, algnN, N, algnK, orderA, A, 4 | beta, C, orderC, workspace);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t A_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, (i << 1) - 1, orderA - i, &A[A_i - strideA], &A[A_i], 3, C, orderC, workspace);
  }
}

inline void gemm_normalize_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t beta, int32_t* C) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &beta, C, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &beta, C, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP); beta |= 1;
      internal::int8::normalize_remainder_i32tensor(stream, strideC, C, orderA);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, std::min(rem, iter_h), &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
      &beta, C, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (iter_h < rem) {
      internal::int8::normalize_remainder_i32tensor(stream, strideC, C, orderA);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &one, C, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
  }
}

void internal::int8::i8GemmR_CRT(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, int32_t n_moduli, int32_t iter, const int8_t* A, uint64_t* C, int32_t* workspace) {
  int32_t orderA = 2 * CRT::active_moduli(n_moduli, iter);
  int64_t strideC = int64_t(algnN) * int64_t(N);
  int32_t accum = int32_t(0 < iter), last = int32_t(n_moduli <= ((iter + 1) << 2));
  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, A, A, 0, workspace);
  internal::int8::accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, C);
}

void internal::int8::i8GemmC_CRT(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, int32_t n_moduli, int32_t iter, const int8_t* A, uint64_t* C, int32_t* workspace) {
  int32_t orderA = 2 * CRT::active_moduli(n_moduli, iter), orderC = CRT::order_p(n_moduli);
  int64_t strideA = int64_t(orderA) * int64_t(algnK) * int64_t(N), strideC = int64_t(algnN) * int64_t(N);
  int32_t accum = int32_t(0 < iter), last = int32_t(n_moduli <= ((iter + 1) << 2));

  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, A, A, 0, workspace);
  internal::int8::normalize_remainder_i32tensor(stream, strideC, workspace, orderA);
  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, &A[strideA], &A[strideA], 1, workspace);
  internal::int8::accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, C);

  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, A, &A[strideA], 0, workspace);
  internal::int8::accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, &C[int64_t(orderC) * strideC]);
}
