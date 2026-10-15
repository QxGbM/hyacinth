
#include <hyacin.h>
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>
#include <tuple>
#include <vector>
#include <algorithm>
#include <stdexcept>

const int32_t u_practical_limit = 80; // u <= 80 to satisfy implementation assumptions
const int32_t StridedSyrkIncrementN = 512;
const cublasGemmAlgo_t cublas_algo = CUBLAS_GEMM_DEFAULT;

// mappings vector for datatypes
const std::vector<hyacinPrecision_t> real_type
({ HYACIN_F64, HYACIN_F64, HYACIN_F32, HYACIN_F32, HYACIN_F16, HYACIN_F16, HYACIN_DD, HYACIN_DD, HYACIN_TF, HYACIN_TF, HYACIN_QF, HYACIN_QF });
const std::vector<hyacinPrecision_t> complex_type
({ HYACIN_F64_COMPLEX, HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_TF_COMPLEX, HYACIN_TF_COMPLEX, HYACIN_QF_COMPLEX, HYACIN_QF_COMPLEX });
const std::vector<int32_t> type_bytes
({ sizeof(double), sizeof(cuDoubleComplex), sizeof(float), sizeof(cuComplex), sizeof(__half), sizeof(__half2), sizeof(double2), sizeof(complex_double2), sizeof(float3), sizeof(complex_float3), sizeof(float4), sizeof(complex_float4) });
const std::vector<int32_t> type_mantissa
({ 52, 52, 23, 23, 10, 10, 105, 105, 71, 71, 95, 95 });

extern "C" void hyacinXherkBatchInit(const hyacinHandle_t* handle, double epi, uint64_t M, int32_t N, hyacinPrecision_t Atype, int32_t* uA, int32_t* cPanels, int32_t* lPanels, uint64_t* stride, hyacinPrecision_t* Gtype, int32_t* gElemBytes) {
  if (stride) { uint64_t N64 = uint64_t(std::max(0, N)); *stride = (N64 * N64 + N64) >> 1; }
  hyacinPrecision_t AtypeReal = real_type[int32_t(Atype)];
  if (uint64_t(1) < M) { --M; M |= M >> 1; M |= M >> 2; M |= M >> 4; M |= M >> 8; M |= M >> 16; M |= M >> 32; ++M; } else { M = uint64_t(1); }
  int32_t exp_epi; double bitsM = std::log2(double(M)), frac = std::frexp(std::abs(epi), &exp_epi);
  int32_t u = std::clamp(int32_t(frac == 0.5) - exp_epi, 0, type_mantissa[int32_t(Atype)]);
  if (uA != nullptr || cPanels != nullptr || lPanels != nullptr) {
    int32_t Complex = int32_t(Atype != AtypeReal), u_corr = std::max(-1, std::min(u_practical_limit, handle->QuantizeBitCorrection + u));
    if (uA) { *uA = u_corr; } if (cPanels) { *cPanels = 1 + Complex; }
    if (lPanels) { *lPanels = (0 <= u_corr) ? (((Complex + 63 + u_corr + u_corr) + int32_t(std::ceil(bitsM))) / 63) : 0; if (4 <= *lPanels) { throw std::runtime_error("i64-limb (g.e. 4) not supported"); }}
  }
  if (Gtype != nullptr || gElemBytes != nullptr) {
    int32_t bits = (handle->GramBitCorrection + 1) + int32_t(std::floor(0.25 * bitsM)) + (u + u);
    hyacinPrecision_t GtypeReal = 
      bits <= type_mantissa[int32_t(HYACIN_F32)] ? HYACIN_F32 : (
      (AtypeReal == HYACIN_F16 || bits <= type_mantissa[int32_t(HYACIN_F64)]) ? HYACIN_F64 : (
      bits <= type_mantissa[int32_t(HYACIN_TF)] ? HYACIN_TF : (
      (bits <= type_mantissa[int32_t(HYACIN_QF)] && (!handle->DeviceIsF64Capable)) ? HYACIN_QF : HYACIN_DD)));
    hyacinPrecision_t g = (Atype == AtypeReal) ? GtypeReal : complex_type[int32_t(GtypeReal)];
    if (Gtype) { *Gtype = g; } if (gElemBytes) { *gElemBytes = type_bytes[g]; }
  }
}

inline void gemm_accum(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  const int32_t zero = 0, one = 1;
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, lda, A, CUDA_R_8I, lda,
      &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
    internal::int8::accumulate_i32tensor(stream, 'A', beta, N, sft, orderA, W, M, orderC, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem_k <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem_k, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C);
    } else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem_k - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, lda, &AN_k[iter_h], CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', 1, N, sft, orderA, W, M, orderC, C);
    }
  }
}

template <char upperMode> inline void IgemmStridedBatched(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const int8_t* A, int32_t lda, int64_t strideA, int32_t* C, int32_t ldc, int64_t strideC, int32_t nbatches) {
  const int32_t one = 1, zero = 0;
  if constexpr(upperMode == 'U' && 64 <= StridedSyrkIncrementN) {
    int32_t iterN = N % StridedSyrkIncrementN, iterM = iterN + (M - N);
    if (iterN) { cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iterM, iterN, K, &one, A, CUDA_R_8I, lda, strideA, A, CUDA_R_8I, lda, strideA, &zero, C, CUDA_R_32I, ldc, strideC, nbatches, CUBLAS_COMPUTE_32I, cublas_algo); }
    while(iterN < N) {
      const int8_t* A_iter = &A[int64_t(iterN) * int64_t(lda)]; int32_t* C_iter = &C[int64_t(iterN) * int64_t(ldc)]; iterN += StridedSyrkIncrementN;
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, iterM += StridedSyrkIncrementN, StridedSyrkIncrementN, K,
        &one, A, CUDA_R_8I, lda, strideA, A_iter, CUDA_R_8I, lda, strideA, &zero, C_iter, CUDA_R_32I, ldc, strideC, nbatches, CUBLAS_COMPUTE_32I, cublas_algo);
  }} else {
    const int8_t* AN; if constexpr(upperMode == 'U') { AN = A; } else { AN = &A[int64_t(N) * int64_t(lda)]; }
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, lda, strideA, AN, CUDA_R_8I, lda, strideA, &zero, C, CUDA_R_32I, ldc, strideC, nbatches, CUBLAS_COMPUTE_32I, cublas_algo);
  }
}

inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  int64_t strideA = int64_t(N) * int64_t(lda), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    IgemmStridedBatched<'U'>(handle, M, N, K, A, lda, strideA, W, M, strideC, orderA);
    internal::int8::accumulate_i32tensor(stream, 'T', beta, N, 0, orderA, W, M, orderC, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      IgemmStridedBatched<'U'>(handle, M, N, iter_k, A_k, lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? 'T' : 'U', k == 0 ? beta : 1, N, 0, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem_k <= iter_k) {
      IgemmStridedBatched<'U'>(handle, M, N, rem_k, A_k, lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', range_k == 0 ? beta : 1, N, 0, orderA, W, M, orderC, C);
    } else {
      IgemmStridedBatched<'U'>(handle, M, N, iter_h, A_k, lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', range_k == 0 ? beta : 1, N, 0, orderA, W, M, orderC, C);
      IgemmStridedBatched<'U'>(handle, M, N, rem_k - iter_h, &A_k[iter_h], lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_i32tensor(stream, 'U', 1, N, 0, orderA, W, M, orderC, C);
    }
  }
}

template <char mode> inline void gemm_accum_iters(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t orderC, uint64_t* C, int32_t* W) {
  int64_t strideA = int64_t(N) * int64_t(lda);
  if constexpr(mode == 'U') {
    for (int32_t i = 1; i < orderA; ++i) { gemm_accum(stream, handle, M, N, K, i << 3, i, &A[int64_t(i) * strideA], A, lda, int32_t(1 < i), orderC, C, W); }
    gemm_accum_diag(stream, handle, M, N, K, orderA, A, lda, int32_t(1 < orderA), orderC, C, W);
  } else { for (int32_t i = 0; i < orderA; ++i) { gemm_accum(stream, handle, M, N, K, i << 3, orderA, &A[int64_t(i) * strideA], &A[int64_t(orderA) * strideA], lda, int32_t(0 < i), orderC, C, W); }}
}

template <int32_t Complex> inline void i8herk_limbs(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t bits = Complex ? 62 : 60;
  int32_t orderB = (bits + int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (orderA << 4)) / 63, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideA = int64_t(lda) * colsA, strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB); if constexpr(Complex) { b_len += b_len; }

  int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), mempool, stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, b_len * sizeof(uint64_t), mempool, stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  if constexpr(Complex) { colsA *= uint64_t(3); }
  if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), colsA, stream); }

  if constexpr(Complex) {
    gemm_accum_iters<'U'>(stream, handle, algnN, N, algnM, orderA, A, lda, orderB, B, scratch);
    gemm_accum_iters<'A'>(stream, handle, algnN, N, algnM, orderA, &A[strideA], lda, orderB, &B[strideB], scratch);
  } else { gemm_accum_iters<'U'>(stream, handle, algnN, N, algnM, orderA, A, lda, orderB, B, scratch); }
  cudaFreeAsync(scratch, stream);

  if constexpr(Complex) { internal::int8::triangle_pack_complex(stream, N, orderB, B, beta, orderC, C); }
    else { internal::int8::triangle_pack(stream, N, orderB, B, beta, orderC, C); }
  cudaFreeAsync(B, stream);
}

template <char mode, int32_t Complex> inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  int64_t strideA = int64_t(N) * int64_t(lda), strideC = int64_t(M) * int64_t(N); if constexpr(Complex) { strideA *= int64_t(3); }
  if (K <= iter_k) {
    IgemmStridedBatched<mode>(handle, M, N, K, A, lda, strideA, W, M, strideC, orderA);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, 0, N, orderA, W, M, orderC, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;
    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      IgemmStridedBatched<mode>(handle, M, N, iter_k, A_k, lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < k), N, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem_k <= iter_h) {
      IgemmStridedBatched<mode>(handle, M, N, rem_k, A_k, lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
    } else {
      IgemmStridedBatched<mode>(handle, M, N, iter_h, A_k, lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
      IgemmStridedBatched<mode>(handle, M, N, rem_k - iter_h, &A_k[iter_h], lda, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, W, M, orderC, C);
    }
  }
}

template <int32_t Complex> inline void i8herk_crt(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  int32_t orderB = (U8CRT::range[orderA - 1] + 63) / 63, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB), w_len = uint64_t(algnN - N) * uint64_t(algnM);
  if constexpr(Complex) { b_len += b_len; w_len += uint64_t(3) * uint64_t(algnM) * uint64_t(colsA); } else { w_len += uint64_t(algnM) * uint64_t(colsA); }

  int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), mempool, stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, b_len * sizeof(uint64_t), mempool, stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  if constexpr(Complex) { colsA *= uint64_t(3); }
  if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(algnM), 0, uint64_t(algnM - M), colsA, stream); }

  gemm_accum_crt<'U', Complex>(stream, handle, algnN, N, algnM, orderA, A, lda, orderB, B, scratch);
  if constexpr(Complex) {
    gemm_accum_crt<'A', Complex>(stream, handle, algnN, N, algnM, orderA, &A[int64_t(lda) * int64_t(N)], lda, orderB, &B[strideB], scratch);
    internal::int8::triangle_pack_complex(stream, N, orderB, B, beta, orderC, C);
  } else { internal::int8::triangle_pack(stream, N, orderB, B, beta, orderC, C); }
  cudaFreeAsync(scratch, stream); cudaFreeAsync(B, stream);
}

template <class matrix_t> inline void herk_dispatcher(const hyacinHandle_t* handle, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t u, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  constexpr uint64_t elem = uint64_t(Complex ? sizeof(uint64_t) : sizeof(uint32_t));
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  if (u < 0 && beta == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); return; }

  int32_t orderA, segM, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63); char alg = handle->GramMatrixAlgorithm;
  std::tie(orderA, segM) = internal::gram_algorithm(alg, M, u + Complex, Complex);
  if (alg == 'L') {
    uint64_t strideA = uint64_t(algnM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }
    int8_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, a_len, mempool, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    internal::int8::quantize_limbs(stream, M, N, orderA, A, lda, vexp, W, algnM);
    i8herk_limbs<Complex>(stream, mempool, cublasH, M, N, orderA, W, algnM, beta, orderC, C);
    cudaFreeAsync(W, stream);
  } else { 
    uint64_t strideA = uint64_t(segM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(segM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }
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

template <class matrix_t> inline void herk_batch_dispatcher(const hyacinHandle_t* handle, int32_t batches, int32_t segM, int32_t lastM, int32_t N, const matrix_t* A, int32_t lda, int32_t u, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  int32_t* segU = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&segU, uint64_t(sizeof(int32_t)) * uint64_t(batches), mempool, stream))
    throw std::runtime_error("Workspace (Batch Vector) allocation failed at Integer SY/HERK");
  internal::int8::vector_range(stream, batches, N, u, vexp, segU); 

  int32_t u_floor = Complex, u_ceil = Complex + u, M = (batches - 1) * segM + lastM; std::vector<std::pair<int32_t, int32_t>> limb_batches;
  while (u_floor <= u_ceil) {
    char algo = handle->GramMatrixAlgorithm; std::pair<int32_t, int32_t> param = internal::gram_algorithm(algo, M, u_floor, Complex);
    if (algo == 'L') { u_floor = 1 + param.second; limb_batches.emplace_back(param); } else { u_floor = 1 + u_ceil; }
  }
  std::vector<int32_t> segmentsU(batches); cudaMemcpyAsync(&segmentsU[0], segU, uint64_t(batches) * uint64_t(sizeof(int32_t)), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); cudaFreeAsync(segU, stream);

  std::vector<int32_t> batchRows(int32_t(limb_batches.size()) + 1, 0);
  for (int32_t i = 0; i < batches; ++i) {
    int32_t uc = segmentsU[i] + Complex; if (0 <= uc) {
      int32_t idx = std::distance(limb_batches.begin(), std::find_if(limb_batches.begin(), limb_batches.end(), [=](const auto& t) { return uc <= t.second; }));
      batchRows[idx] += (i + 1 == batches) ? lastM : segM; segmentsU[i] = idx;
    } else { segmentsU[i] = -1; }
  }

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

  int32_t ceil_batch_rows = batchRows[n_limb_batches]; char algo = handle->GramMatrixAlgorithm;
  std::pair<int32_t, int32_t> ceil_batch = internal::gram_algorithm(algo, ceil_batch_rows, u_ceil, Complex);
  if (algo == 'C' && 0 < ceil_batch_rows) {
    int32_t orderA = ceil_batch.first, algnM = ceil_batch.second;
    uint64_t strideA = uint64_t(algnM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }
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
