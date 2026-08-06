
#include <internal.hpp>

inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, int32_t K, int32_t moduli, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t iter, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, beta, N, orderA, iter, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, k == 0 ? beta : 1, N, orderA, iter, workspace, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, range_k == 0 ? beta : 1, N, orderA, iter, workspace, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, range_k == 0 ? beta : 1, N, orderA, iter, workspace, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, iter, workspace, M, orderC, C);
    }
  }
}

inline void gemm_accum_crt_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t moduli, int32_t orderA, const int8_t* A, const int8_t* B, int32_t iter, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_h) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, B, CUDA_R_8I, K, strideA, B, CUDA_R_8I, K, strideA,
      &one, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, 'U', beta, N, orderA, iter, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_h) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_h) {
      const int8_t* A_k = &A[k], *B_k = &B[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, B_k, CUDA_R_8I, K, strideA, B_k, CUDA_R_8I, K, strideA,
        &one, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 'U', k == 0 ? beta : 1, N, orderA, iter, workspace, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k], *B_k = &B[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, B_k, CUDA_R_8I, K, strideA, B_k, CUDA_R_8I, K, strideA,
        &one, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 'U', range_k == 0 ? beta : 1, N, orderA, iter, workspace, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 'U', range_k == 0 ? beta : 1, N, orderA, iter, workspace, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, B_k, CUDA_R_8I, K, strideA, B_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 'U', 1, N, orderA, iter, workspace, M, orderC, C);
    }
  }
}

void internal::int8::i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA * int64_t(8)];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    quantize_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, workspace, workspace, i, beta, C, orderC, scratch);
  }
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA * int64_t(8)];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    quantize_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, workspace, workspace, i, beta, C, orderC, scratch);
  }
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA * int64_t(8)];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    quantize_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, workspace, workspace, i, beta, C, orderC, scratch);
  }
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA * int64_t(24)];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    quantize_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accum_crt_diag(stream, handle, algnN, N, algnM, moduli, orderA, workspace, &workspace[stride], i, beta, C, orderC, scratch);
    gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, moduli, orderA, workspace, &workspace[stride], i, beta, &C[strideC * int64_t(orderC)], orderC, scratch);
  }
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA * int64_t(24)];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    quantize_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accum_crt_diag(stream, handle, algnN, N, algnM, moduli, orderA, workspace, &workspace[stride], i, beta, C, orderC, scratch);
    gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, moduli, orderA, workspace, &workspace[stride], i, beta, &C[strideC * int64_t(orderC)], orderC, scratch);
  }
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA * int64_t(24)];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    quantize_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accum_crt_diag(stream, handle, algnN, N, algnM, moduli, orderA, workspace, &workspace[stride], i, beta, C, orderC, scratch);
    gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, moduli, orderA, workspace, &workspace[stride], i, beta, &C[strideC * int64_t(orderC)], orderC, scratch);
  }
  vector_sums(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}
