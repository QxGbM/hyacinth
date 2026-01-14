
#include <internal.hpp>
#include <crt_selector.hpp>

inline void gemm_accumulate_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t n_moduli, int32_t iter, int32_t accum, int32_t last, uint64_t* C, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, k == 0 ? accum : 1, strideC, n_moduli, iter, workspace, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, range_k == 0 ? accum : 1, strideC, n_moduli, iter, workspace, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 1 | (last << 1), strideC, n_moduli, iter, workspace, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, (range_k == 0 ? accum : 1) | (last << 1), strideC, n_moduli, iter, workspace, C);
    }
  }
}

inline std::pair<int32_t, int32_t> umax_moduli(int32_t umax, int32_t k, int32_t c) {
  int32_t algnK = (k + 255) & (~255);
  int32_t b = 1 + ((int32_t(std::ceil(std::log2(algnK))) + (umax << 1) + 2 + c) >> 3);
  return std::make_pair(algnK, std::max(2, std::min(23, b)));
}

void internal::int8::i63ATA_f64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int32_t algnM, n_moduli; std::tie(algnM, n_moduli) = umax_moduli(umax, M, 0);
  int64_t strideA = int64_t(algnM) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t orderA = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    quantize_f64_modular(stream, M, N, i, A, lda, umax, vec_expon, orderA, workspace, algnM);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, workspace, workspace, n_moduli, i, accum, last, C, scratch);
  }
}

void internal::int8::i63ATA_f32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int32_t algnM, n_moduli; std::tie(algnM, n_moduli) = umax_moduli(umax, M, 0);
  int64_t strideA = int64_t(algnM) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t orderA = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    quantize_f32_modular(stream, M, N, i, A, lda, umax, vec_expon, orderA, workspace, algnM);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, workspace, workspace, n_moduli, i, accum, last, C, scratch);
  }
}

void internal::int8::i63AHA_cf64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int32_t algnM, n_moduli; std::tie(algnM, n_moduli) = umax_moduli(umax, M, 0);
  int32_t orderC = 1 + ((n_moduli << 3) / 63);
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(orderC) * int64_t(ldc) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t orderA = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    quantize_cf64_modular(stream, M, N, i, A, lda, umax, vec_expon, orderA, workspace, algnM);

    int64_t stride = int64_t(orderA) * strideA;
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, workspace, workspace, n_moduli, i, accum, 0, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, &workspace[stride], &workspace[stride], n_moduli, i, 1, last, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, workspace, &workspace[stride], n_moduli, i, accum, last, &C[strideC], scratch);
  }
}

void internal::int8::i63AHA_cf32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int32_t algnM, n_moduli; std::tie(algnM, n_moduli) = umax_moduli(umax, M, 0);
  int32_t orderC = 1 + ((n_moduli << 3) / 63);
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(orderC) * int64_t(ldc) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t orderA = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    quantize_cf32_modular(stream, M, N, i, A, lda, umax, vec_expon, orderA, workspace, algnM);

    int64_t stride = int64_t(orderA) * strideA;
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, workspace, workspace, n_moduli, i, accum, 0, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, &workspace[stride], &workspace[stride], n_moduli, i, 1, last, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, orderA, workspace, &workspace[stride], n_moduli, i, accum, last, &C[strideC], scratch);
  }
}
