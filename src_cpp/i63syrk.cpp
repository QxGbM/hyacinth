
#include <internal.hpp>
#include <crt_selector.hpp>

inline void gemm_accumulate(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft_lo, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideC = int64_t(M) * int64_t(N), strideACC = int64_t(M) * int64_t(N + 1);
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, op, strideC, sft_lo, orderA, workspace, strideC, orderC, C, strideACC);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = op | 1;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, strideC, sft_lo, orderA, workspace, strideC, orderC, C, strideACC);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, std::min(rem, iter_h), &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, strideC, sft_lo, orderA, workspace, strideC, orderC, C, strideACC);
    if (iter_h < rem) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, op_acc, strideC, sft_lo, orderA, workspace, strideC, orderC, C, strideACC);
    }
  }
}

inline void gemm_accumulate_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N), strideACC = int64_t(M) * int64_t(N + 1);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, op, strideC, 0, orderA, workspace, strideC, orderC, C, strideACC);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = 5;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, strideC, 0, orderA, workspace, strideC, orderC, C, strideACC);
    }

    const int8_t* A_k = &A[int64_t(range_k)];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, std::min(rem, iter_h), &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, strideC, 0, orderA, workspace, strideC, orderC, C, strideACC);
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, op_acc, strideC, 0, orderA, workspace, strideC, orderC, C, strideACC);
    }
  }
}

inline void i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i) {
    int64_t AT_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, i, orderA, &AT[AT_i], A, i == 0 ? beta : 1, C, orderC, workspace);
  }
}

inline void i8GemmT(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  gemm_accumulate_diag(stream, handle, algnN, N, algnK, orderA, A, 4 | beta, C, orderC, workspace);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t A_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, (i << 1) - 1, orderA - i, &A[A_i - strideA], &A[A_i], 3, C, orderC, workspace);
  }
}

void internal::int8::i63ATA_f64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(ldc) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize_f64(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
  vsum_f64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

void internal::int8::i63ATA_f32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(ldc) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  quantize_f32(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
  vsum_f32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

void internal::int8::i63AHA_cf64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(ldc) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize_cf64(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, N, ldc, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, N, ldc, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideIm], orderC, scratch);
  vsum_cf64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

void internal::int8::i63AHA_cf32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(ldc) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  quantize_cf32(stream, M, N, A, lda, umax, vec_expon, orderA, N, algnM, workspace);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, N, ldc, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, N, ldc, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideIm], orderC, scratch);
  vsum_cf32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

inline void gemm_accumulate_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t moduli, const int8_t* AT, const int8_t* A, 
  int32_t orderA, int32_t iter, int32_t accum, int32_t last, uint64_t* C, int32_t* workspace) {

  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N), strideACC = int64_t(M) * int64_t(N + 1);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, orderA, iter, workspace, strideC, C, strideACC);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, k == 0 ? accum : 1, strideC, orderA, iter, workspace, strideC, C, strideACC);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, range_k == 0 ? accum : 1, strideC, orderA, iter, workspace, strideC, C, strideACC);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 1 | (last << 1), strideC, orderA, iter, workspace, strideC, C, strideACC);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, (range_k == 0 ? accum : 1) | (last << 1), strideC, orderA, iter, workspace, strideC, C, strideACC);
    }
  }
}

void internal::int8::i63ATA_f64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(ldc) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = CRT::active_moduli(orderA, i);
    int32_t accum = int32_t(0 < i), last = int32_t(orderA <= ((i + 1) << 3));
    quantize_f64_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, orderA, i, accum, last, C, scratch);
  }
  vsum_f64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

void internal::int8::i63ATA_f32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(ldc) * int64_t(N + 1);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = CRT::active_moduli(orderA, i);
    int32_t accum = int32_t(0 < i), last = int32_t(orderA <= ((i + 1) << 3));
    quantize_f32_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, orderA, i, accum, last, C, scratch);
  }
  vsum_f32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

void internal::int8::i63AHA_cf64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(ldc) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = CRT::active_moduli(orderA, i);
    int32_t accum = int32_t(0 < i), last = int32_t(orderA <= ((i + 1) << 3));
    quantize_cf64_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, orderA, i, accum, 0, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, &workspace[stride], &workspace[stride], orderA, i, 1, last, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, &workspace[stride], orderA, i, accum, last, &C[strideIm], scratch);
  }
  vsum_cf64(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}

void internal::int8::i63AHA_cf32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(ldc) * int64_t(N + 1), strideIm = strideC * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = CRT::active_moduli(orderA, i);
    int32_t accum = int32_t(0 < i), last = int32_t(orderA <= ((i + 1) << 3));
    quantize_cf32_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, N, algnM, workspace);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, orderA, i, accum, 0, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, &workspace[stride], &workspace[stride], orderA, i, 1, last, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, &workspace[stride], orderA, i, accum, last, &C[strideIm], scratch);
  }
  vsum_cf32(stream, M, N, A, lda, umax, vec_expon, orderC, &C[strideC - int64_t(ldc)], strideC);
}
