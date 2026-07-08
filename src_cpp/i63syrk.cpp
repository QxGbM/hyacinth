
#include <internal.hpp>

inline void gemm_accumulate(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, op, N, sft, 8, orderA, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = op | 1;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, N, sft, 8, orderA, workspace, M, orderC, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, std::min(rem, iter_h), &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, N, sft, 8, orderA, workspace, M, orderC, C);
    if (iter_h < rem) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, op_acc, N, sft, 8, orderA, workspace, M, orderC, C);
    }
  }
}

inline void gemm_accumulate_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, op, N, 0, 16, orderA, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = 1;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, N, 0, 16, orderA, workspace, M, orderC, C);
    }

    const int8_t* A_k = &A[int64_t(range_k)];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, std::min(rem, iter_h), &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, N, 0, 16, orderA, workspace, M, orderC, C);
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, op_acc, N, 0, 16, orderA, workspace, M, orderC, C);
    }
  }
}

inline void i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i) {
    int64_t AT_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, M, N, K, i << 3, orderA, &AT[AT_i], A, i == 0 ? beta : 1, C, orderC, workspace);
  }
}

inline void i8GemmT(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t A_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, M, N, K, (i << 4) - 8, orderA - i, &A[A_i - strideA], &A[A_i], i == 1 ? (beta | 2) : 3, C, orderC, workspace);
  }
  gemm_accumulate_diag(stream, handle, M, N, K, orderA, A, orderA <= 1 ? beta : 1, C, orderC, workspace);
}

void internal::int8::i63ATA_f64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize_f64(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, algnN, N, algnM, workspace, orderA, 0, C, orderC, scratch);
  vsum_f64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63ATA_f32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize_f32(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, algnN, N, algnM, workspace, orderA, 0, C, orderC, scratch);
  vsum_f32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63ATA_f16_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize_f16(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, algnN, N, algnM, workspace, orderA, 0, C, orderC, scratch);
  vsum_f16(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_cf64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize_cf64(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, algnN, N, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, algnN, N, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideIm], orderC, scratch);
  vsum_cf64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_cf32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize_cf32(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, algnN, N, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, algnN, N, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideIm], orderC, scratch);
  vsum_cf32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_cf16_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<__half>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(N) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize_cf16(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, algnN, N, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, algnN, N, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, algnN, N, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideIm], orderC, scratch);
  vsum_cf16(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

inline void gemm_accumulate_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t moduli, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t iter, int32_t accum, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, accum, N, orderA, iter, workspace, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, k == 0 ? accum : 1, N, orderA, iter, workspace, M, orderC, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, range_k == 0 ? accum : 1, N, orderA, iter, workspace, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 1, N, orderA, iter, workspace, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, range_k == 0 ? accum : 1, N, orderA, iter, workspace, M, orderC, C);
    }
  }
}

void internal::int8::i63ATA_f64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t accum = int32_t(0 < i);
    quantize_f64_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, workspace, orderA, i, accum, C, orderC, scratch);
  }
  vsum_f64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63ATA_f32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t accum = int32_t(0 < i);
    quantize_f32_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, workspace, orderA, i, accum, C, orderC, scratch);
  }
  vsum_f32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63ATA_f16_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t accum = int32_t(0 < i);
    quantize_f16_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, workspace, orderA, i, accum, C, orderC, scratch);
  }
  vsum_f16(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_cf64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t accum = int32_t(0 < i);
    quantize_cf64_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, workspace, orderA, i, accum, C, orderC, scratch);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, &workspace[stride], &workspace[stride], orderA, i, 1, C, orderC, scratch);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, &workspace[stride], orderA, i, accum, &C[strideIm], orderC, scratch);
  }
  vsum_cf64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_cf32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t accum = int32_t(0 < i);
    quantize_cf32_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, workspace, orderA, i, accum, C, orderC, scratch);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, &workspace[stride], &workspace[stride], orderA, i, 1, C, orderC, scratch);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, &workspace[stride], orderA, i, accum, &C[strideIm], orderC, scratch);
  }
  vsum_cf32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}

void internal::int8::i63AHA_cf16_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<__half>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t accum = int32_t(0 < i);
    quantize_cf16_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, workspace, orderA, i, accum, C, orderC, scratch);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, &workspace[stride], &workspace[stride], orderA, i, 1, C, orderC, scratch);
    gemm_accumulate_crt(stream, handle, algnN, N, algnM, moduli, workspace, &workspace[stride], orderA, i, accum, &C[strideIm], orderC, scratch);
  }
  vsum_cf16(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(N)], strideC);
}
