
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>

extern "C" void hyacinXherkBatchInit(const hyacinHandle_t* handle, double epi, uint64_t M, int32_t N, hyacinPrecision_t Atype, int32_t* uA, int32_t* cPanels, int32_t* lPanels, uint64_t* stride) {
  if (stride) { uint64_t N64 = uint64_t(std::max(0, N)); *stride = (N64 * N64 + N64) >> 1; }
  if (uA != nullptr || cPanels != nullptr || lPanels != nullptr) {
    constexpr hyacinPrecision_t real_type[]{ HYACIN_F64, HYACIN_F64, HYACIN_F32, HYACIN_F32, HYACIN_F16, HYACIN_F16, HYACIN_DD, HYACIN_DD, HYACIN_TF, HYACIN_TF, HYACIN_QF, HYACIN_QF };
    constexpr int32_t type_mantissa[]{ 52, 52, 23, 23, 10, 10, 105, 105, 71, 71, 95, 95 };
    constexpr int32_t u_practical_limit = 80; // u <= 80 to satisfy implementation assumptions

    if (uint64_t(1) < M) { --M; M |= M >> 1; M |= M >> 2; M |= M >> 4; M |= M >> 8; M |= M >> 16; M |= M >> 32; ++M; } else { M = uint64_t(1); }
    int32_t exp_epi; double frac = std::frexp(std::abs(epi), &exp_epi);
    int32_t u = std::clamp(int32_t(frac == 0.5) - exp_epi, 0, type_mantissa[int32_t(Atype)]);
    int32_t Complex = int32_t(Atype != real_type[int32_t(Atype)]), u_corr = std::max(-1, std::min(u_practical_limit, handle->QuantizeBitCorrection + u));
    if (uA) { *uA = u_corr; } if (cPanels) { *cPanels = 1 + Complex; }
    if (lPanels) {
      *lPanels = (0 <= u_corr) ? ((63 + std::max(0, (u_corr + Complex) * 2) + int32_t(std::ceil(std::log2(double(M))))) / 63) : 0;
      if (4 <= *lPanels) { throw std::runtime_error("i64-limb (g.e. 4) not supported"); }
    }
  }
}

template <char mode> static inline void gemm_accum(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  const int32_t zero = 0, one = 1;
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, lda, A, CUDA_R_8I, lda, &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if constexpr(mode == 'R') { internal::int8::accumulate_limbs_cross_real(stream, beta, N, sft, orderA, W, M, orderC, C); } else
    { internal::int8::accumulate_limbs_cross_imag(stream, beta, N, sft, orderA, W, M, orderC, C); }
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda, &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      if constexpr(mode == 'R') { internal::int8::accumulate_limbs_cross_real(stream, k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C); } else
      { internal::int8::accumulate_limbs_cross_imag(stream, k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C); }
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem_k <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem_k, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda, &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      if constexpr(mode == 'R') { internal::int8::accumulate_limbs_cross_real(stream, range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C); } else
      { internal::int8::accumulate_limbs_cross_imag(stream, range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C); }
    } else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda, &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      if constexpr(mode == 'R') { internal::int8::accumulate_limbs_cross_real(stream, range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C); } else
      { internal::int8::accumulate_limbs_cross_imag(stream, range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C); }
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem_k - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, lda, &AN_k[iter_h], CUDA_R_8I, lda, &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      if constexpr(mode == 'R') { internal::int8::accumulate_limbs_cross_real(stream, 1, N, sft, orderA, W, M, orderC, C); } else
      { internal::int8::accumulate_limbs_cross_imag(stream, 1, N, sft, orderA, W, M, orderC, C); }
    }
  }
}

template <char upperMode> static inline void IgemmStridedBatched(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* A, int32_t lda, int64_t strideA, int32_t* C, int32_t ldc, int64_t strideC, int32_t nbatches) {
  constexpr int32_t StridedSyrkIncrementN = 512; const int32_t one = 1, zero = 0;
  if constexpr(upperMode == 'U' && 64 <= StridedSyrkIncrementN) {
    int32_t iterN = N % StridedSyrkIncrementN, iterM = iterN + (M - N);
    if (iterN) { cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iterM, iterN, K, &one, A, CUDA_R_8I, lda, strideA, A, CUDA_R_8I, lda, strideA, &zero, C, CUDA_R_32I, ldc, strideC, nbatches, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT); }
    while(iterN < N) {
      const int8_t* A_iter = &A[int64_t(iterN) * int64_t(lda)]; int32_t* C_iter = &C[int64_t(iterN) * int64_t(ldc)]; iterN += StridedSyrkIncrementN;
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iterM += StridedSyrkIncrementN, StridedSyrkIncrementN, K,
        &one, A, CUDA_R_8I, lda, strideA, A_iter, CUDA_R_8I, lda, strideA, &zero, C_iter, CUDA_R_32I, ldc, strideC, nbatches, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }} else {
    const int8_t* AN; if constexpr(upperMode == 'U') { AN = A; } else { AN = &A[int64_t(N) * int64_t(lda)]; }
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, lda, strideA, AN, CUDA_R_8I, lda, strideA, &zero, C, CUDA_R_32I, ldc, strideC, nbatches, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }
}

static inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  int64_t strideA = int64_t(N) * int64_t(lda), strideW = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    IgemmStridedBatched<'U'>(handle, M, N, K, A, lda, strideA, W, M, strideW, orderA);
    internal::int8::accumulate_limbs_diag(stream, beta, N, orderA, W, M, orderC, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      IgemmStridedBatched<'U'>(handle, M, N, iter_k, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_limbs_diag(stream, k == 0 ? beta : 1, N, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem_k <= iter_k) {
      IgemmStridedBatched<'U'>(handle, M, N, rem_k, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_limbs_diag(stream, range_k == 0 ? beta : 1, N, orderA, W, M, orderC, C);
    } else {
      IgemmStridedBatched<'U'>(handle, M, N, iter_h, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_limbs_diag(stream, range_k == 0 ? beta : 1, N, orderA, W, M, orderC, C);
      IgemmStridedBatched<'U'>(handle, M, N, rem_k - iter_h, &A_k[iter_h], lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_limbs_diag(stream, 1, N, orderA, W, M, orderC, C);
    }
  }
}

template <char mode> static inline void gemm_accum_iters(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  int64_t strideA = int64_t(N) * int64_t(lda);
  if constexpr(mode == 'R') {
    for (int32_t i = 1; i < orderA; ++i) { gemm_accum<mode>(stream, handle, M, N, K, i << 3, i, &A[int64_t(i) * strideA], A, lda, beta || int32_t(1 < i), orderC, C, W); }
    gemm_accum_diag(stream, handle, M, N, K, orderA, A, lda, beta || int32_t(1 < orderA), orderC, C, W);
  } else { for (int32_t i = 0; i < orderA; ++i) { gemm_accum<mode>(stream, handle, M, N, K, i << 3, orderA, &A[int64_t(i) * strideA], &A[int64_t(orderA) * strideA], lda, beta || int32_t(0 < i), orderC, C, W); }}
}

template <int32_t Complex> static inline void i8herk_limbs(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63), *scratch = nullptr;
  int64_t colsA = int64_t(N) * int64_t(orderA), strideA = int64_t(lda) * colsA;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * uint64_t(sizeof(int32_t)), mempool, stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");

  if constexpr(Complex) { colsA *= uint64_t(3); }
  if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), colsA, stream); }

  gemm_accum_iters<'R'>(stream, handle, algnN, N, algnM, orderA, A, lda, beta, orderC, C, scratch);
  if constexpr(Complex) { gemm_accum_iters<'I'>(stream, handle, algnN, N, algnM, orderA, &A[strideA], lda, beta, orderC, C, scratch); }
  cudaFreeAsync(scratch, stream);
}

static inline void gemm_accum_crt_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, uint32_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  int64_t strideA = int64_t(N) * int64_t(lda), strideW = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    IgemmStridedBatched<'U'>(handle, M, N, K, A, lda, strideA, W, M, strideW, orderA);
    internal::int8::accumulate_crt_diag_real(stream, 0, N, orderA, W, M, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;
    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      IgemmStridedBatched<'U'>(handle, M, N, iter_k, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_diag_real(stream, int32_t(0 < k), N, orderA, W, M, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem_k <= iter_h) {
      IgemmStridedBatched<'U'>(handle, M, N, rem_k, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_diag_real(stream, int32_t(0 < range_k), N, orderA, W, M, C);
    } else {
      IgemmStridedBatched<'U'>(handle, M, N, iter_h, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_diag_real(stream, int32_t(0 < range_k), N, orderA, W, M, C);
      IgemmStridedBatched<'U'>(handle, M, N, rem_k - iter_h, &A_k[iter_h], lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_diag_real(stream, 1, N, orderA, W, M, C);
    }
  }
}

static inline void gemm_accum_crt_2m(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, uint32_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  int64_t strideA = int64_t(2) * int64_t(N) * int64_t(lda), strideW = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    IgemmStridedBatched<'A'>(handle, M, N, K, A, lda, strideA, W, M, strideW, orderA);
    internal::int8::accumulate_crt_rotate(stream, 0, N, orderA, W, M, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;
    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      IgemmStridedBatched<'A'>(handle, M, N, iter_k, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_rotate(stream, int32_t(0 < k), N, orderA, W, M, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem_k <= iter_h) {
      IgemmStridedBatched<'A'>(handle, M, N, rem_k, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_rotate(stream, int32_t(0 < range_k), N, orderA, W, M, C);
    } else {
      IgemmStridedBatched<'A'>(handle, M, N, iter_h, A_k, lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_rotate(stream, int32_t(0 < range_k), N, orderA, W, M, C);
      IgemmStridedBatched<'A'>(handle, M, N, rem_k - iter_h, &A_k[iter_h], lda, strideA, W, M, strideW, orderA);
      internal::int8::accumulate_crt_rotate(stream, 1, N, orderA, W, M, C);
    }
  }
}

template <char mode> static inline void gemm_accum_crt_cextra(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, uint32_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536; constexpr char gemm_mode = mode == 'R' ? 'U' : 'A';
  int64_t strideA = int64_t(3) * int64_t(N) * int64_t(lda), strideW = int64_t(M) * int64_t(N);
  const int8_t* A_e = &A[int64_t(N) * int64_t(lda) * int64_t(U8CRT2M::len_2m_moduli) * int64_t(2)]; int32_t extra_len = orderA - U8CRT2M::len_2m_moduli;
  if (K <= iter_k) {
    IgemmStridedBatched<gemm_mode>(handle, M, N, K, A_e, lda, strideA, W, M, strideW, extra_len);
    if constexpr(mode == 'R') { internal::int8::accumulate_crt_extra_real(stream, 0, N, orderA, W, M, C); } else
    { internal::int8::accumulate_crt_extra_imag(stream, 0, N, orderA, W, M, C); }
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;
    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A_e[k];
      IgemmStridedBatched<gemm_mode>(handle, M, N, iter_k, A_k, lda, strideA, W, M, strideW, extra_len);
      if constexpr(mode == 'R') { internal::int8::accumulate_crt_extra_real(stream, int32_t(0 < k), N, orderA, W, M, C); } else
      { internal::int8::accumulate_crt_extra_imag(stream, int32_t(0 < k), N, orderA, W, M, C); }
    }

    const int8_t* A_k = &A_e[range_k];
    if (rem_k <= iter_h) {
      IgemmStridedBatched<gemm_mode>(handle, M, N, rem_k, A_k, lda, strideA, W, M, strideW, extra_len);
      if constexpr(mode == 'R') { internal::int8::accumulate_crt_extra_real(stream, int32_t(0 < range_k), N, orderA, W, M, C); } else
      { internal::int8::accumulate_crt_extra_imag(stream, int32_t(0 < range_k), N, orderA, W, M, C); }
    } else {
      IgemmStridedBatched<gemm_mode>(handle, M, N, iter_h, A_k, lda, strideA, W, M, strideW, extra_len);
      if constexpr(mode == 'R') { internal::int8::accumulate_crt_extra_real(stream, int32_t(0 < range_k), N, orderA, W, M, C); } else
      { internal::int8::accumulate_crt_extra_imag(stream, int32_t(0 < range_k), N, orderA, W, M, C); }
      IgemmStridedBatched<gemm_mode>(handle, M, N, rem_k - iter_h, &A_k[iter_h], lda, strideA, W, M, strideW, extra_len);
      if constexpr(mode == 'R') { internal::int8::accumulate_crt_extra_real(stream, 1, N, orderA, W, M, C); } else
      { internal::int8::accumulate_crt_extra_imag(stream, 1, N, orderA, W, M, C); }
    }
  }
}

template <int32_t Complex> static inline void i8herk_crt(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  if constexpr(Complex) { 
    int64_t colsA = int64_t(N) * int64_t(orderA <= U8CRT2M::len_2m_moduli ? (orderA * 2) : (orderA * 3 - U8CRT2M::len_2m_moduli)), strideB = (int64_t(N) * int64_t(N) + int64_t(N)) / int64_t(2);
    int32_t* scratch = nullptr; uint32_t* B = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(N) * uint64_t(std::max(orderA, U8CRT2M::len_2m_moduli)) * uint64_t(sizeof(int32_t)), mempool, stream))
      throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, uint64_t(strideB) * uint64_t((orderA + 3) / 4) * uint64_t(sizeof(uint64_t)), mempool, stream))
      throw std::runtime_error("Workspace (u32) allocation failed at Integer SY/HERK.");

    if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(algnM), 0, uint64_t(algnM - M), colsA, stream); }
    gemm_accum_crt_2m(stream, handle, algnN, N, algnM, orderA, A, lda, B, scratch);
    if (U8CRT2M::len_2m_moduli < orderA) {
      gemm_accum_crt_cextra<'R'>(stream, handle, algnN, N, algnM, orderA, A, lda, B, scratch);
      gemm_accum_crt_cextra<'I'>(stream, handle, algnN, N, algnM, orderA, &A[int64_t(lda) * int64_t(N)], lda, B, scratch); 
    }
    cudaFreeAsync(scratch, stream); internal::int8::reconstruct_complex(stream, beta, strideB, orderA, B, orderC, C); cudaFreeAsync(B, stream);
  } else {
    int64_t colsA = int64_t(N) * int64_t(orderA), strideB = (int64_t(N) * int64_t(N) + int64_t(N)) / int64_t(2);
    int32_t* scratch = nullptr; uint32_t* B = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * uint64_t(sizeof(int32_t)), mempool, stream))
      throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, uint64_t(strideB) * uint64_t((orderA + 3) / 4) * uint64_t(sizeof(uint32_t)), mempool, stream))
      throw std::runtime_error("Workspace (u32) allocation failed at Integer SY/HERK.");

    if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(algnM), 0, uint64_t(algnM - M), colsA, stream); }
    gemm_accum_crt_real(stream, handle, algnN, N, algnM, orderA, A, lda, B, scratch); cudaFreeAsync(scratch, stream); 
    internal::int8::reconstruct_real(stream, beta, strideB, orderA, B, orderC, C); cudaFreeAsync(B, stream);
  }
}

template <class matrix_t> static inline void herk_dispatcher(const hyacinHandle_t* handle, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t u, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  constexpr uint64_t elem = uint64_t(Complex ? sizeof(uint64_t) : sizeof(uint32_t));
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  if (u < 0 && beta == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); return; }

  char alg = handle->GramMatrixAlgorithm; std::pair<int32_t, int32_t> param = internal::gram_algorithm(alg, M, N, u + Complex, Complex, handle->TCGemmTargetK, handle->StrideBatchPerfPenalty);
  int32_t orderA = param.first, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63); 
  if (alg == 'L') {
    uint64_t strideA = uint64_t(algnM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }
    int8_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, a_len, mempool, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    internal::int8::quantize_limbs(stream, M, N, orderA, A, lda, vexp, W, algnM);
    i8herk_limbs<Complex>(stream, mempool, cublasH, M, N, orderA, W, algnM, beta, orderC, C);
    cudaFreeAsync(W, stream);
  } else {
    int32_t segM = param.second;
    uint64_t strideA = uint64_t(segM) * uint64_t(N), a_len = uint64_t(segM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(orderA <= U8CRT2M::len_2m_moduli ? (orderA * 2) : (orderA * 3 - U8CRT2M::len_2m_moduli)); } else { a_len += strideA * uint64_t(orderA); }
    int8_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, a_len, mempool, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    for (int32_t y = 0; y < M; y += segM) {
      int32_t rows = std::min(M - y, segM);
      internal::int8::quantize_crt(stream, rows, N, orderA, &A[y], lda, vexp, W, segM);
      i8herk_crt<Complex>(stream, mempool, cublasH, rows, N, orderA, W, segM, beta, orderC, C); beta = 1;
    }
    cudaFreeAsync(W, stream);
  }
}

template <class matrix_t> static inline void herk_batch_dispatcher(const hyacinHandle_t* handle, int32_t batches, int32_t segM, int32_t lastM, int32_t N, const matrix_t* A, int32_t lda, int32_t u, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  int32_t* segU = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&segU, uint64_t(sizeof(int32_t)) * uint64_t(batches), mempool, stream))
    throw std::runtime_error("Workspace (Batch Vector) allocation failed at Integer SY/HERK");
  internal::int8::vector_range(stream, batches, N, u + Complex, vexp, segU); 

  int32_t u_floor = Complex, u_ceil = Complex + u, M = (batches - 1) * segM + lastM; std::vector<std::pair<int32_t, int32_t>> limb_batches;
  while (u_floor <= u_ceil) {
    char algo = handle->GramMatrixAlgorithm; std::pair<int32_t, int32_t> param = internal::gram_algorithm(algo, M, N, u_floor, Complex, handle->TCGemmTargetK, handle->StrideBatchPerfPenalty);
    if (algo == 'L') { u_floor = 1 + param.second; limb_batches.emplace_back(param); } else { u_floor = 1 + u_ceil; }
  }
  std::vector<int32_t> segmentsU(batches); cudaMemcpyAsync(&segmentsU[0], segU, uint64_t(batches) * uint64_t(sizeof(int32_t)), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); cudaFreeAsync(segU, stream);

  std::vector<int32_t> batchRows(int32_t(limb_batches.size()) + 1, 0);
  for (int32_t i = 0; i < batches; ++i) {
    int32_t uc = segmentsU[i]; if (0 <= uc) {
      int32_t idx = std::distance(limb_batches.begin(), std::find_if(limb_batches.begin(), limb_batches.end(), [=](const std::pair<int32_t, int32_t>& t) { return uc <= t.second; }));
      batchRows[idx] += (i + 1 == batches) ? lastM : segM; segmentsU[i] = idx;
    } else { segmentsU[i] = -1; }
  }

  int32_t ceil_batch_rows = batchRows.back(); char algo = handle->GramMatrixAlgorithm;
  std::pair<int32_t, int32_t> ceil_batch = internal::gram_algorithm(algo, ceil_batch_rows, N, u_ceil, Complex, handle->TCGemmTargetK, handle->StrideBatchPerfPenalty);
  if (algo == 'L') { limb_batches.emplace_back(ceil_batch); }
  int32_t algnN = (N + 63) & (~63), n_limb_batches = int32_t(limb_batches.size());
  for (int32_t b = 0; b < n_limb_batches; ++b) if (0 < batchRows[b]) {
    int32_t orderA = limb_batches[b].first, algnM = (batchRows[b] + 255) & (~255);
    uint64_t strideA = uint64_t(algnM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }
    int8_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, a_len, mempool, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    int32_t rowsW = 0, i = 0;
    while (i < batches && segmentsU[i] != b) { ++i; }
    while (i < batches) {
      int32_t i_end = i; while (i_end < batches && segmentsU[i_end] == b) { ++i_end; }
      int32_t rowsA = i * segM, rows = std::min(M, i_end * segM) - rowsA;
      internal::int8::quantize_limbs(stream, rows, N, orderA, &A[rowsA], lda, vexp, &W[rowsW], algnM);
      rowsW += rows; i = i_end; while (i < batches && segmentsU[i] != b) { ++i; }
    }
    i8herk_limbs<Complex>(stream, mempool, cublasH, rowsW, N, orderA, W, algnM, beta, orderC, C); beta = 1;
    cudaFreeAsync(W, stream);
  }

  if (algo == 'C' && 0 < ceil_batch_rows) {
    int32_t orderA = ceil_batch.first, algnM = ceil_batch.second;
    uint64_t strideA = uint64_t(algnM) * uint64_t(N), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(orderA <= U8CRT2M::len_2m_moduli ? (orderA * 2) : (orderA * 3 - U8CRT2M::len_2m_moduli)); } else { a_len += strideA * uint64_t(orderA); }
    int8_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, a_len, mempool, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    int32_t rowsW = 0, i = 0; 
    while (i < batches && segmentsU[i] != n_limb_batches) { ++i; }
    while (i < batches) {
      int32_t i_end = i; while (i_end < batches && segmentsU[i_end] == n_limb_batches) { ++i_end; }
      int32_t rowsA = i * segM, rows = std::min(M, i_end * segM) - rowsA;
      if (algnM < (rowsW + rows)) {
        int32_t seg1 = algnM - rowsW, seg2 = rows - seg1;
        if (0 < seg1) { internal::int8::quantize_crt(stream, seg1, N, orderA, &A[rowsA], lda, vexp, &W[rowsW], algnM); }
        i8herk_crt<Complex>(stream, mempool, cublasH, algnM, N, orderA, W, algnM, beta, orderC, C); beta = 1;
        while (algnM < seg2) {
          internal::int8::quantize_crt(stream, algnM, N, orderA, &A[rowsA + seg1], lda, vexp, W, algnM);
          i8herk_crt<Complex>(stream, mempool, cublasH, algnM, N, orderA, W, algnM, beta, orderC, C); beta = 1;
          seg1 += algnM; seg2 -= algnM;
        }
        if (0 < seg2) { internal::int8::quantize_crt(stream, seg2, N, orderA, &A[rowsA + seg1], lda, vexp, W, algnM); } rowsW = seg2;
      } else { internal::int8::quantize_crt(stream, rows, N, orderA, &A[rowsA], lda, vexp, &W[rowsW], algnM); rowsW += rows; }
      i = i_end; while (i < batches && segmentsU[i] != n_limb_batches) { ++i; }
    }
    if (0 < rowsW) { i8herk_crt<Complex>(stream, mempool, cublasH, rowsW, N, orderA, W, algnM, beta, orderC, C); beta = 1; }
    cudaFreeAsync(W, stream);
  }
}

extern "C" void hyacinXherkBatch(const hyacinHandle_t* handle, int32_t batches, int32_t segM, int32_t lastM, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t u, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  if (batches <= 0 || N <= 0 || orderC <= 0) { return; }
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  
  if (batches == 1) switch (Atype) {
    case HYACIN_F64: herk_dispatcher(handle, lastM, N, (const double*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F32: herk_dispatcher(handle, lastM, N, (const float*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F16: herk_dispatcher(handle, lastM, N, (const __half*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F64_COMPLEX: herk_dispatcher(handle, lastM, N, (const cuDoubleComplex*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F32_COMPLEX: herk_dispatcher(handle, lastM, N, (const cuComplex*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F16_COMPLEX: herk_dispatcher(handle, lastM, N, (const __half2*)A, lda, u, vexp, beta, orderC, C); return;
    default: return;
  } else switch (Atype) {
    case HYACIN_F64: herk_batch_dispatcher(handle, batches, segM, lastM, N, (const double*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F32: herk_batch_dispatcher(handle, batches, segM, lastM, N, (const float*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F16: herk_batch_dispatcher(handle, batches, segM, lastM, N, (const __half*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F64_COMPLEX: herk_batch_dispatcher(handle, batches, segM, lastM, N, (const cuDoubleComplex*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F32_COMPLEX: herk_batch_dispatcher(handle, batches, segM, lastM, N, (const cuComplex*)A, lda, u, vexp, beta, orderC, C); return;
    case HYACIN_F16_COMPLEX: herk_batch_dispatcher(handle, batches, segM, lastM, N, (const __half2*)A, lda, u, vexp, beta, orderC, C); return;
    default: return;
  }
}
