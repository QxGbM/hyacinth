
#include <internal.hpp>
#include <stdexcept>

inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, int32_t K, int32_t moduli, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t iter, int32_t beta, uint64_t* C, int32_t orderC, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, beta, N, orderA, iter, W, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, k == 0 ? beta : 1, N, orderA, iter, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, range_k == 0 ? beta : 1, N, orderA, iter, W, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, range_k == 0 ? beta : 1, N, orderA, iter, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, iter, W, M, orderC, C);
    }
  }
}

template <class matrix_t>
inline void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N) * int64_t(orderC);
  uint64_t mod_len = uint64_t(std::min(orderA, 8)), w_len = uint64_t(strideA) * mod_len, w_pad = uint64_t(algnN - N) * uint64_t(algnM);
  if constexpr(Complex) { w_len *= uint64_t(3); }

  int8_t* W = nullptr; int32_t* scratch = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&W, w_len + w_pad, stream))
    throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(N) * mod_len * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    internal::int8::quantize_modular(stream, M, N, i, A, lda, umax, vexp, moduli, N, algnM, W);
    if constexpr(Complex) {
      int64_t strideW = int64_t(moduli) * strideA, strideW2 = strideW * int64_t(2);
      gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, &W[strideW2], &W[strideW2], i, beta, C, orderC, scratch);
      gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, moduli, orderA, W, &W[strideW], i, beta, &C[strideC], orderC, scratch);
    } else { gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, W, W, i, beta, C, orderC, scratch); }
  }
  cudaFreeAsync(W, stream); cudaFreeAsync(scratch, stream);

  if constexpr(Complex) { C = &C[strideC * int64_t(2)]; } else { C = &C[strideC]; }
  internal::int8::vector_sums(stream, M, N, A, lda, umax, vexp, orderC, C);
}

namespace internal::int8 {

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

}