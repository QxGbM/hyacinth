
#include <internal.hpp>

inline void gemm_accum(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'A', beta, N, sft, 8, orderA, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', k == 0 ? beta : 1, N, sft, 8, orderA, workspace, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, workspace, M, orderC, C);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, workspace, M, orderC, C);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', 1, N, sft, 8, orderA, workspace, M, orderC, C);
    }
  }
}

inline void gemm_accum(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, const int8_t* BT, const int8_t* B, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  if (K <= iter_h) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, BT, CUDA_R_8I, K, B, CUDA_R_8I, K,
      &one, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'A', beta, N, sft, 8, orderA, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_h) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_h) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k], *BT_k = &BT[k], *BN_k = &B[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, BT_k, CUDA_R_8I, K, BN_k, CUDA_R_8I, K,
        &one, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', k == 0 ? beta : 1, N, sft, 8, orderA, workspace, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k], *BT_k = &BT[range_k], *BN_k = &B[range_k];
    if (rem <= iter_h) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, BT_k, CUDA_R_8I, K, BN_k, CUDA_R_8I, K,
        &one, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, workspace, M, orderC, C);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, workspace, M, orderC, C);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, BT_k, CUDA_R_8I, K, BN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', 1, N, sft, 8, orderA, workspace, M, orderC, C);
    }
  }
}

inline void i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* AT, const int8_t* A, int32_t orderA, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i) {
    int64_t AT_i = int64_t(i) * strideA;
    gemm_accum(stream, handle, M, N, K, i << 3, orderA, &AT[AT_i], A, int32_t(0 < i), C, orderC, workspace);
  }
}

inline void i8GemmU(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* A, int32_t orderA, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t AT_i = int64_t(i - 1) * strideA, AN_i = int64_t(i) * strideA;
    gemm_accum(stream, handle, M, N, K, (i << 4) - 8, orderA - i, &A[AT_i], &A[AN_i], int32_t(1 < i), C, orderC, workspace);
  }
}

inline void i8GemmU(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* A, const int8_t* B, int32_t orderA, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t AT_i = int64_t(i - 1) * strideA, AN_i = int64_t(i) * strideA;
    gemm_accum(stream, handle, M, N, K, (i << 4) - 8, orderA - i, &A[AT_i], &A[AN_i], &B[AT_i], &B[AN_i], int32_t(1 < i), C, orderC, workspace);
  }
}

inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'T', 1, N, 0, 16, orderA, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
    }
  }
}

inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, const int8_t* B, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_h) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, B, CUDA_R_8I, K, strideA, B, CUDA_R_8I, K, strideA,
      &one, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'T', 1, N, 0, 16, orderA, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_h) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_h) {
      const int8_t* A_k = &A[k], *B_k = &B[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, B_k, CUDA_R_8I, K, strideA, B_k, CUDA_R_8I, K, strideA,
        &one, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k], *B_k = &B[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, B_k, CUDA_R_8I, K, strideA, B_k, CUDA_R_8I, K, strideA,
        &one, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, B_k, CUDA_R_8I, K, strideA, B_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'U', 1, N, 0, 16, orderA, workspace, M, orderC, C);
    }
  }
}

void internal::int8::i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmU(stream, handle, algnN, N, algnM, workspace, orderA, C, orderC, scratch);
  gemm_accum_diag(stream, handle, algnN, N, algnM, orderA, workspace, C, orderC, scratch);
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmU(stream, handle, algnN, N, algnM, workspace, orderA, C, orderC, scratch);
  gemm_accum_diag(stream, handle, algnN, N, algnM, orderA, workspace, C, orderC, scratch);
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmU(stream, handle, algnN, N, algnM, workspace, orderA, C, orderC, scratch);
  gemm_accum_diag(stream, handle, algnN, N, algnM, orderA, workspace, C, orderC, scratch);
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmU(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, C, orderC, scratch);
  gemm_accum_diag(stream, handle, algnN, N, algnM, orderA, workspace, &workspace[strideA], C, orderC, scratch);
  i8GemmF(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, &C[strideC * int64_t(orderC)], orderC, scratch);
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmU(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, C, orderC, scratch);
  gemm_accum_diag(stream, handle, algnN, N, algnM, orderA, workspace, &workspace[strideA], C, orderC, scratch);
  i8GemmF(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, &C[strideC * int64_t(orderC)], orderC, scratch);
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmU(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, C, orderC, scratch);
  gemm_accum_diag(stream, handle, algnN, N, algnM, orderA, workspace, &workspace[strideA], C, orderC, scratch);
  i8GemmF(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, &C[strideC * int64_t(orderC)], orderC, scratch);
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}
