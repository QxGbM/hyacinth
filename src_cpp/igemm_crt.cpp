
#include <internal.hpp>
#include <crt_selector.hpp>

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
  accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, C);
}

void internal::int8::i8GemmC_CRT(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, int32_t n_moduli, int32_t iter, const int8_t* A, uint64_t* C, int32_t* workspace) {
  int32_t orderA = 2 * CRT::active_moduli(n_moduli, iter), orderC = CRT::order_p(n_moduli);
  int64_t strideA = int64_t(orderA) * int64_t(algnK) * int64_t(N), strideC = int64_t(algnN) * int64_t(N);
  int32_t accum = int32_t(0 < iter), last = int32_t(n_moduli <= ((iter + 1) << 2));

  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, A, A, 0, workspace);
  normalize_remainder_i32tensor(stream, strideC, workspace, orderA);
  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, &A[strideA], &A[strideA], 1, workspace);
  accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, C);

  gemm_normalize_crt(stream, handle, algnN, N, algnK, orderA, A, &A[strideA], 0, workspace);
  accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, &C[int64_t(orderC) * strideC]);
}

inline std::pair<int32_t, int32_t> umax_moduli(int32_t umax, int32_t k, int32_t c) {
  int32_t b = 1 + ((int32_t(std::ceil(std::log2(k))) + (umax << 1) + c) >> 4);
  return std::make_pair((k + 255) & (~255), b < 2 ? 2 : (14 < b ? 14 : b));
}

void internal::int8::i63ATA_f64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, uint64_t* C, int32_t ldc, int8_t* workspace) {
  int32_t algnM, n_moduli; std::tie(algnM, n_moduli) = umax_moduli(umax, M, 0);
  int64_t strideA = int64_t(M) * int64_t(N), i8_bytes = strideA << 3;
  int32_t* scratch = (int32_t*)&workspace[i8_bytes];

  cudaMemsetAsync(workspace, 0, i8_bytes, stream);
  for (int32_t i = 0; (i << 2) < n_moduli; ++i) {
    quantize_f64_modular(stream, M, N, i, A, lda, umax, vec_expon, CRT::active_moduli(n_moduli, i) << 1, workspace, algnM);
    i8GemmR_CRT(stream, handle, N, ldc, algnM, n_moduli, i, workspace, C, scratch);
  }
}
