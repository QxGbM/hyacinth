
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

extern "C" void hyacinXGautoType(const hyacinHandle_t* handle, double epi, uint64_t M, int32_t N, hyacinPrecision_t Atype, int32_t* uA, int32_t* cPanels, int32_t* lPanels, uint64_t* stride, hyacinPrecision_t* Gtype, int32_t* gElemBytes) {
  if (stride) { uint64_t N64 = uint64_t(N); *stride = (N64 * N64 + N64) >> 1; }
  if (uA == nullptr && cPanels == nullptr && lPanels == nullptr && Gtype == nullptr && gElemBytes == nullptr) { return; }

  hyacinPrecision_t AtypeReal = real_type[int32_t(Atype)];
  if (uint64_t(1) < M) { --M; M |= M >> 1; M |= M >> 2; M |= M >> 4; M |= M >> 8; M |= M >> 16; M |= M >> 32; ++M; } else { M = uint64_t(1); }
  double epi_nrm = std::min(1., std::max(std::abs(epi), std::ldexp(1., -type_mantissa[int32_t(Atype)]))), bitsM = std::log2(double(M));
  int32_t u = int32_t(std::ceil(-std::log2(epi_nrm))), u_corr = handle->QuantizeBitCorrection, g_corr = handle->GramBitCorrection;
  if (uA != nullptr || cPanels != nullptr || lPanels != nullptr) {
    u_corr = std::max(-1, std::min(u_practical_limit, u_corr + u)); if (uA) { *uA = u_corr; }
    int32_t c = 1 + int32_t(Atype != AtypeReal); if (cPanels) { *cPanels = c; }
    if (lPanels) { *lPanels = (0 <= u_corr) ? (((c + 63 + u_corr + u_corr) + int32_t(std::ceil(bitsM))) / 63) : 0; if (4 <= *lPanels) { throw std::runtime_error("i64-limb (g.e. 4) not supported"); }}
  }
  if (Gtype != nullptr || gElemBytes != nullptr) {
    int32_t bits = (g_corr + 1) + int32_t(std::floor(0.25 * bitsM)) + (u + u);
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

template <int32_t Complex>
inline void i8herk_limbs(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t bits = Complex ? 62 : 60;
  int32_t orderB = (bits + int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (orderA << 4)) / 63, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideA = int64_t(lda) * colsA, strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB); if constexpr(Complex) { b_len += b_len; }

  int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), mempool, stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, b_len * sizeof(uint64_t), mempool, stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  if (M < algnM) {
    uint64_t cols = uint64_t(colsA); if constexpr(Complex) { cols *= uint64_t(3); }
    cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), cols, stream);
  }

  if constexpr(Complex) {
    gemm_accum_iters<'U'>(stream, handle, algnN, N, algnM, orderA, A, lda, orderB, B, scratch);
    gemm_accum_iters<'A'>(stream, handle, algnN, N, algnM, orderA, &A[strideA], lda, orderB, &B[strideB], scratch);
  } else { gemm_accum_iters<'U'>(stream, handle, algnN, N, algnM, orderA, A, lda, orderB, B, scratch); }
  cudaFreeAsync(scratch, stream);

  if constexpr(Complex) { internal::int8::triangle_pack(stream, 0, N, orderB, B, (const ulonglong4_32a*)nullptr, uint32_t(0), beta, orderC, C); }
    else { internal::int8::triangle_pack(stream, 0, N, orderB, B, (const ulonglong2*)nullptr, uint32_t(0), beta, orderC, C); }
  cudaFreeAsync(B, stream);
}

template <char mode, int32_t Complex> inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 130816, iter_h = 65536;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N); if constexpr(Complex) { strideA *= int64_t(3); }
  if (K <= iter_k) {
    IgemmStridedBatched<mode>(handle, M, N, K, A, K, strideA, W, M, strideC, orderA);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, 0, N, orderA, W, M, orderC, C);
  } else {
    int32_t rem_k = K % iter_k; rem_k = rem_k < iter_h ? (rem_k + iter_k) : rem_k;
    int32_t range_k = K - rem_k;
    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      IgemmStridedBatched<mode>(handle, M, N, iter_k, A_k, K, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < k), N, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem_k <= iter_h) {
      IgemmStridedBatched<mode>(handle, M, N, rem_k, A_k, K, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
    } else {
      IgemmStridedBatched<mode>(handle, M, N, iter_h, A_k, K, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
      IgemmStridedBatched<mode>(handle, M, N, rem_k - iter_h, &A_k[iter_h], K, strideA, W, M, strideC, orderA);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, W, M, orderC, C);
    }
  }
}

template <class matrix_t> inline void i8herk_crt(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int32_t segM, const matrix_t* A, int32_t lda, const int32_t* vexp, uint32_t corr, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  using sum_t = std::conditional_t<Complex, ulonglong4_32a, ulonglong2>; std::div_t divM = std::div(M, segM);
  int32_t orderB = (U8CRT::range[orderA - 1] + 63) / 63, algnM = (divM.quot + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB), w_len = uint64_t(algnN - N) * uint64_t(algnM);
  if constexpr(Complex) { b_len += b_len; w_len += uint64_t(3) * uint64_t(algnM) * uint64_t(colsA); } else { w_len += uint64_t(algnM) * uint64_t(colsA); }

  int8_t* W = nullptr; int32_t* scratch = nullptr; uint64_t* B = nullptr; sum_t *vsum = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, w_len, mempool, stream))
    throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), mempool, stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, b_len * sizeof(uint64_t), mempool, stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");
   if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vsum, uint64_t(N) * sizeof(sum_t), mempool, stream))
    throw std::runtime_error("Workspace (vsum) allocation failed at Integer SY/HERK.");

  if (divM.quot < algnM) {
    uint64_t cols = uint64_t(colsA); if constexpr(Complex) { cols *= uint64_t(3); }
    cudaMemset2DAsync(&W[divM.quot], uint64_t(algnM), 0, uint64_t(algnM - divM.quot), cols, stream);
  }

  for (int32_t i = 0; i < segM; ++i) {
    int32_t rows = divM.quot + int32_t(segM <= (divM.rem + i));
    internal::int8::quantize_crt(stream, rows, N, orderA, A, lda, corr, vexp, W, algnM, vsum); A = &A[rows];
    gemm_accum_crt<'U', Complex>(stream, handle, algnN, N, algnM, orderA, W, orderB, B, scratch);
    if constexpr(Complex) { gemm_accum_crt<'A', Complex>(stream, handle, algnN, N, algnM, orderA, &W[int64_t(algnM) * int64_t(N)], orderB, &B[strideB], scratch); }
    internal::int8::triangle_pack(stream, rows, N, orderB, B, vsum, corr, beta || int32_t(0 < i), orderC, C);
  }
  cudaFreeAsync(W, stream); cudaFreeAsync(scratch, stream); cudaFreeAsync(B, stream); cudaFreeAsync(vsum, stream);
}

template <class matrix_t> inline void herk_dispatcher(const hyacinHandle_t* handle, int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, int32_t* uptr) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  constexpr uint64_t elem = uint64_t(Complex ? sizeof(uint64_t) : sizeof(uint32_t));
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  int32_t uc = *uptr, i = *beta; *beta = 1;
  if (uc < 0 && 0 < M) {
    int32_t* vbuf = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vbuf, 2048, mempool, stream))
      throw std::runtime_error("Workspace (vbuf) allocation failed at Integer SY/HERK");
    *uptr = handle->DeviceSMs; internal::int8::vector_range(stream, M, N, A, lda, uptr, vexp, vbuf); cudaStreamSynchronize(stream); cudaFreeAsync(vbuf, stream); uc = *uptr + Complex;
  } if (uc < 0 && i == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); return; }

  int32_t orderA, segM; char alg = handle->GramMatrixAlgorithm;
  std::tie(orderA, segM) = internal::gram_algorithm(alg, M, uc, Complex);
  if (alg == 'L') {
    int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
    uint64_t strideA = uint64_t(algnM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }
    int8_t* W = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&W, a_len, mempool, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    internal::int8::quantize_limbs(stream, M, N, orderA, A, lda, vexp, W, algnM);
    i8herk_limbs<Complex>(stream, mempool, cublasH, M, N, orderA, W, algnM, i, orderC, C);
    cudaFreeAsync(W, stream);
  } else { i8herk_crt(stream, mempool, cublasH, M, N, orderA, segM, A, lda, vexp, uint32_t(uc), i, orderC, C); }
}

extern "C" void hyacinXherk(const hyacinHandle_t* handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t u_hint, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  int32_t* uptr = (int32_t*)handle->pinnedWorkspace; *uptr = u_hint + int32_t((0 <= u_hint) && (Atype != real_type[int32_t(Atype)]));
  switch (Atype) {
    case HYACIN_F64: herk_dispatcher(handle, M, N, (const double*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F32: herk_dispatcher(handle, M, N, (const float*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F16: herk_dispatcher(handle, M, N, (const __half*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F64_COMPLEX: herk_dispatcher(handle, M, N, (const cuDoubleComplex*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F32_COMPLEX: herk_dispatcher(handle, M, N, (const cuComplex*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F16_COMPLEX: herk_dispatcher(handle, M, N, (const __half2*)A, lda, vexp, &beta, orderC, C, uptr); return;
    default: return;
  }
}

extern "C" void hyacinXherkBatchInit(const hyacinHandle_t* handle, double epi, int32_t N, hyacinPrecision_t Atype, uint64_t* Bbytes) {
  if (handle->BatchK <= 0 || N <= 0) { if (Bbytes) { *Bbytes = uint64_t(0); } return; }
  int32_t Complex = int32_t(Atype != real_type[int32_t(Atype)]), batchK = handle->BatchK;
  double epi_nrm = std::min(1., std::max(std::abs(epi), std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  int32_t u_ceil = Complex + std::min(u_practical_limit, handle->QuantizeBitCorrection + int32_t(std::ceil(-std::log2(epi_nrm))));
  int32_t batches = 1, u_floor = Complex; char algo = handle->GramMatrixAlgorithm;
  if (algo != 'C' && algo != 'c') while (u_floor <= u_ceil) {
    char algi = algo; int32_t ui = u_floor; internal::gram_algorithm(algi, batchK, ui, Complex);
    if (algi == 'L') { u_floor = 1 + ui; ++batches; } else { u_floor = 1 + u_ceil; }
  }

  auto iter = handle->BatchTensor, iter_end = &iter[batches];
  if (iter) { iter->U = -1; iter->Order = batches; iter->Rows = 0; }
  int32_t order = type_bytes[int32_t(Atype)], panelsLimb = Complex ? 3 : 1;
  while (++iter != iter_end) { iter->Prefix = order; iter->Rows = 0; order += panelsLimb * iter->Order; }
  if (Bbytes) { *Bbytes = uint64_t(N) * uint64_t(batchK) * uint64_t(order); }
}

template <class matrix_t> inline matrix_t* herk_batch_dispatcher(const hyacinHandle_t* handle, int32_t M, int32_t R, int32_t N, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, int8_t* B, int32_t* uptr) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  int32_t uc = -1, K = handle->BatchK; auto& arena = *(handle->BatchTensor); const matrix_t* A = &((const matrix_t*)B)[arena.Rows];
  if (0 < M) {
    int32_t* vbuf = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vbuf, uint64_t(2048), mempool, stream))
      throw std::runtime_error("Workspace (vbuf) allocation failed at Integer SY/HERK");
    *uptr = handle->DeviceSMs; internal::int8::vector_range(stream, M, N, A, K, uptr, vexp, vbuf); cudaFreeAsync(vbuf, stream); cudaStreamSynchronize(stream); uc = *uptr + Complex;
  }

  if (0 <= uc) {
    auto iter_end = &handle->BatchTensor[arena.Order], iter = std::find_if(&handle->BatchTensor[1], iter_end, [=](const auto& t) { return uc <= t.U; });
    if (iter == iter_end) { arena.Rows += M; arena.U = std::max(arena.U, uc); } else {
      auto& t = *iter; int32_t row = t.Rows; int64_t prefix = int64_t(N) * int64_t(K) * int64_t(t.Prefix);
      if (M <= K - row) { internal::int8::quantize_limbs(stream, M, N, t.Order, A, K, vexp, &B[prefix + int64_t(row)], K); t.Rows = row + M; } else
      { i8herk_limbs<Complex>(stream, mempool, cublasH, row, N, t.Order, &B[prefix], K, *beta, orderC, C); internal::int8::quantize_limbs(stream, M, N, t.Order, A, K, vexp, &B[prefix], K); t.Rows = M; *beta = 1; }
    }
  }

  if (arena.Rows + R <= K) { return &((matrix_t*)B)[arena.Rows]; } else if (R <= K) {
    herk_dispatcher(handle, arena.Rows, N, (const matrix_t*)B, K, vexp, beta, orderC, C, &(arena.U));
    arena.Rows = 0; arena.U = -1; return (matrix_t*)B;
  } else { return nullptr; }
}

extern "C" void* hyacinXherkBatch(const hyacinHandle_t* handle, int32_t CommitLines, int32_t ReserveLines, int32_t N, hyacinPrecision_t Atype, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, int8_t* Bdata) {
  if (handle->BatchK <= 0 || N <= 0 || orderC <= 0) { return nullptr; }
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  int32_t* uptr = (int32_t*)handle->pinnedWorkspace; *uptr = -1;
  switch (Atype) {
    case HYACIN_F64: return herk_batch_dispatcher<double>(handle, CommitLines, ReserveLines, N, vexp, beta, orderC, C, Bdata, uptr);
    case HYACIN_F32: return herk_batch_dispatcher<float>(handle, CommitLines, ReserveLines, N, vexp, beta, orderC, C, Bdata, uptr);
    case HYACIN_F16: return herk_batch_dispatcher<__half>(handle, CommitLines, ReserveLines, N, vexp, beta, orderC, C, Bdata, uptr);
    case HYACIN_F64_COMPLEX: return herk_batch_dispatcher<cuDoubleComplex>(handle, CommitLines, ReserveLines, N, vexp, beta, orderC, C, Bdata, uptr);
    case HYACIN_F32_COMPLEX: return herk_batch_dispatcher<cuComplex>(handle, CommitLines, ReserveLines, N, vexp, beta, orderC, C, Bdata, uptr);
    case HYACIN_F16_COMPLEX: return herk_batch_dispatcher<__half2>(handle, CommitLines, ReserveLines, N, vexp, beta, orderC, C, Bdata, uptr);
    default: return nullptr;
  }
}

template <class matrix_t> inline void herk_flush_dispatcher(const hyacinHandle_t* handle, int32_t N, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, int8_t* Bdata) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  constexpr uint64_t elem = uint64_t(Complex ? sizeof(uint64_t) : sizeof(uint32_t));
  cudaStream_t stream = handle->cudaStream; cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; 
  if (0 < handle->BatchK) {
    auto& arena = *(handle->BatchTensor); int64_t stride = int64_t(N) * int64_t(handle->BatchK);
    for (int32_t i = 1; i < arena.Order; ++i)
    { auto& t = handle->BatchTensor[i]; if (0 < t.Rows) { i8herk_limbs<Complex>(stream, mempool, cublasH, t.Rows, N, t.Order, &Bdata[stride * int64_t(t.Prefix)], handle->BatchK, beta, orderC, C); beta = 1; t.Rows = 0; }}
    if (0 < arena.Rows)
    { herk_dispatcher(handle, arena.Rows, N, (const matrix_t*)Bdata, handle->BatchK, vexp, &beta, orderC, C, &(arena.U)); arena.Rows = 0; arena.U = -1; }
  } if (beta == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); }
}

extern "C" void hyacinXherkBatchFlush(const hyacinHandle_t* handle, int32_t N, hyacinPrecision_t Atype, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, int8_t* Bdata) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  switch (Atype) {
    case HYACIN_F64: herk_flush_dispatcher<double>(handle, N, vexp, beta, orderC, C, Bdata); break;
    case HYACIN_F32: herk_flush_dispatcher<float>(handle, N, vexp, beta, orderC, C, Bdata); break;
    case HYACIN_F16: herk_flush_dispatcher<__half>(handle, N, vexp, beta, orderC, C, Bdata); break;
    case HYACIN_F64_COMPLEX: herk_flush_dispatcher<cuDoubleComplex>(handle, N, vexp, beta, orderC, C, Bdata); break;
    case HYACIN_F32_COMPLEX: herk_flush_dispatcher<cuComplex>(handle, N, vexp, beta, orderC, C, Bdata); break;
    case HYACIN_F16_COMPLEX: herk_flush_dispatcher<__half2>(handle, N, vexp, beta, orderC, C, Bdata); break;
    default: break;
  }
}
