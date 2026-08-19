
#include <hyacin.h>
#include <internal.hpp>
#include <stdexcept>

inline void gemm_accum(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t beta, uint64_t* C, int32_t orderC, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'A', beta, N, sft, 8, orderA, W, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', k == 0 ? beta : 1, N, sft, 8, orderA, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, W, M, orderC, C);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, W, M, orderC, C);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', 1, N, sft, 8, orderA, W, M, orderC, C);
    }
  }
}

inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, uint64_t* C, int32_t orderC, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'T', 1, N, 0, 16, orderA, W, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, W, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', 1, N, 0, 16, orderA, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'U', 1, N, 0, 16, orderA, W, M, orderC, C);
    }
  }
}

inline void i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* AT, const int8_t* A, int32_t orderA, uint64_t* C, int32_t orderC, int32_t* W) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i)
    gemm_accum(stream, handle, M, N, K, i << 3, orderA, &AT[int64_t(i) * strideA], A, int32_t(0 < i), C, orderC, W);
}

inline void i8GemmU(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* A, int32_t orderA, uint64_t* C, int32_t orderC, int32_t* W) {
  int64_t strideA = int64_t(K) * int64_t(N);
  for (int32_t i = 1; i < orderA; ++i)
    gemm_accum(stream, handle, M, N, K, (i << 4) - 8, orderA - i, &A[int64_t(i - 1) * strideA], &A[int64_t(i) * strideA], int32_t(1 < i), C, orderC, W);
  gemm_accum_diag(stream, handle, M, N, K, orderA, A, C, orderC, W);
}

template <class matrix_t>
inline void AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  constexpr int32_t bits = Complex ? 62 : 60;
  int32_t orderB = std::min(orderC, (int32_t(std::ceil(std::log2(double(std::max(1, M))))) + bits + (orderA << 4)) / 63);
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t strideW = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t w_len = uint64_t(strideW), w_pad = uint64_t(algnN - N) * uint64_t(algnM), b_len = uint64_t(strideB);
  if constexpr(Complex) { w_len *= uint64_t(3); b_len *= uint64_t(2); }

  int8_t* W = nullptr; int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&W, w_len + w_pad, stream))
    throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(N) * uint64_t(orderA) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&B, b_len * sizeof(uint64_t), stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  internal::int8::quantize(stream, -1, M, A, lda, uint32_t(0), vexp, orderA, N, algnM, W);
  if constexpr(Complex) {
    int64_t strideW2 = strideW + strideW;
    i8GemmU(stream, handle, algnN, N, algnM, &W[strideW2], orderA, B, orderB, scratch);
    i8GemmF(stream, handle, algnN, N, algnM, W, &W[strideW], orderA, &B[strideB], orderB, scratch);
  } else { i8GemmU(stream, handle, algnN, N, algnM, W, orderA, B, orderB, scratch); }
  cudaFreeAsync(W, stream); cudaFreeAsync(scratch, stream);

  internal::int8::triangle_pack(stream, Complex, 0, N, orderB, B, uint32_t(0), beta, orderC, C);
  cudaFreeAsync(B, stream);
}

template <class matrix_t>
void herk_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, int32_t* uptr, hyacinAlgorithm_t alg) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  if (M <= 0 && beta == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * uint64_t(Complex + 1) * sizeof(uint32_t), stream); return; }
  *uptr = 0; internal::int8::vector_exponents(stream, M, N, A, lda, uptr, const_cast<int32_t*>(vexp)); int32_t umax = *uptr;

  int32_t bits = int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (Complex ? 2 : 0) + (umax << 1);
  int32_t orderA_limbs = int32_t(uint32_t(umax + 9) >> 3), orderA_crt = int32_t(uint32_t(bits + 11) >> 3);
  int32_t cost_limbs = int32_t(uint32_t(orderA_limbs * (orderA_limbs + 1)) >> 1), cost_crt = orderA_crt + int32_t(uint32_t(orderA_crt) >> 3);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS || (alg == HYACIN_ALG_AUTO && (orderA_limbs <= 3 || cost_limbs <= cost_crt)));
  if (use_limbs) { AHA_limbs(stream, handle, M, N, orderA_limbs, A, lda, vexp, beta, orderC, C); }
    else { internal::int8::i63AHA_crt(stream, handle, M, N, orderA_crt, A, lda, umax, vexp, beta, orderC, C); }
}

extern "C" void hyacinXherk(hyacinHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, hyacinAlgorithm_t alg) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch(Atype) {
    case HYACIN_F64: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const double*)A, lda, vexp, beta, orderC, C, (int32_t*)handle.pinnedWorkspace, alg); return;
    case HYACIN_F32: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const float*)A, lda, vexp, beta, orderC, C, (int32_t*)handle.pinnedWorkspace, alg); return;
    case HYACIN_F16: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half*)A, lda, vexp, beta, orderC, C, (int32_t*)handle.pinnedWorkspace, alg); return;
    case HYACIN_F64_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuDoubleComplex*)A, lda, vexp, beta, orderC, C, (int32_t*)handle.pinnedWorkspace, alg); return;
    case HYACIN_F32_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuComplex*)A, lda, vexp, beta, orderC, C, (int32_t*)handle.pinnedWorkspace, alg); return;
    case HYACIN_F16_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half2*)A, lda, vexp, beta, orderC, C, (int32_t*)handle.pinnedWorkspace, alg); return;
    default: return;
  }
}

namespace internal::int8 {

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { AHA_limbs(stream, handle, M, N, orderA, A, lda, vexp, beta, orderC, C); }

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { AHA_limbs(stream, handle, M, N, orderA, A, lda, vexp, beta, orderC, C); }

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { AHA_limbs(stream, handle, M, N, orderA, A, lda, vexp, beta, orderC, C); }

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { AHA_limbs(stream, handle, M, N, orderA, A, lda, vexp, beta, orderC, C); }

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { AHA_limbs(stream, handle, M, N, orderA, A, lda, vexp, beta, orderC, C); }

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { AHA_limbs(stream, handle, M, N, orderA, A, lda, vexp, beta, orderC, C); }

}
