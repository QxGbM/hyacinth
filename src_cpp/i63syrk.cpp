
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <crt_constants.hpp>
#include <algorithm>
#include <stdexcept>

const int32_t u_practical_limit = 80; // u <= 80 to satisfy implementation assumptions
const int32_t b_practical_limit = U8CRT::range[22]; // b <= 177 to satisfy implementation assumptions
// mappings vector for datatypes
const std::vector<hyacinPrecision_t> real_type({ HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF, HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF });
const std::vector<hyacinPrecision_t> complex_type({ HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX, HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX });
const std::vector<int32_t> type_bytes({ sizeof(double), sizeof(float), sizeof(__half), sizeof(double2), sizeof(float4), sizeof(cuDoubleComplex), sizeof(cuComplex), sizeof(__half2), sizeof(complex_double2), sizeof(complex_float4) });
const std::vector<int32_t> type_mantissa({ 52, 23, 10, 105, 95, 52, 23, 10, 105, 95 });
const cublasGemmAlgo_t cublas_algo = CUBLAS_GEMM_DEFAULT;

extern "C" int32_t hyacinXquantizeScale(hyacinHandle_t handle, double epi, int32_t u_corr, int32_t localM, int32_t globalM, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t beta, int32_t* vexp, int32_t* cPanels, int32_t* lPanels) {
  if (N <= 0) { return -1; }
  double epi_nrm = std::min(1., std::max(std::abs(epi), std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  int32_t u = std::min(u_practical_limit, u_corr + int32_t(std::ceil(-std::log2(epi_nrm))));
  if (cPanels != nullptr || lPanels != nullptr) {
    int32_t c = 1 + int32_t(Atype != real_type[int32_t(Atype)]);
    if (cPanels) { *cPanels = c; } if (lPanels) { *lPanels = ((c + 63 + u + u) + int32_t(std::ceil(std::log2(double(std::max(1, globalM)))))) / 63; }
  }
  
  if (0 < localM) {
    Timer::register_kernel(handle.cudaStream, handle.timer);
    switch(Atype) {
      case HYACIN_F64: internal::int8::vector_exponents(handle.cudaStream, localM, N, (const double*)A, lda, u, beta, vexp); return u;
      case HYACIN_F32: internal::int8::vector_exponents(handle.cudaStream, localM, N, (const float2*)A, lda, u, beta, vexp); return u;
      case HYACIN_F16: internal::int8::vector_exponents(handle.cudaStream, localM, N, (const __half*)A, lda, u, beta, vexp); return u;
      case HYACIN_F64_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, localM, N, (const cuDoubleComplex*)A, lda, u, beta, vexp); return u;
      case HYACIN_F32_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, localM, N, (const cuComplex*)A, lda, u, beta, vexp); return u;
      case HYACIN_F16_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, localM, N, (const __half2*)A, lda, u, beta, vexp); return u;
      default: return u;
    }
  } else { return u; }
}

int32_t internal::int8::gram_algorithm(char& alg, int32_t M, int32_t& u) {
  u = std::max(0, u); int32_t bitsM = 2 + int32_t(std::ceil(std::log2(double(std::max(1, M))))), bits = bitsM + (u + u);
  if (bits < 0 || b_practical_limit < bits) { throw std::runtime_error("Int range exceeded all suitable Gram matrix algorithm"); } 
  int32_t orderA_limbs = (u <= 7) ? 1 : int32_t(uint32_t(u + 9) >> 3);
  int32_t orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::lower_bound(&U8CRT::range[1], &U8CRT::range[23], bits)));
  int32_t cost_limbs = int32_t(uint32_t(orderA_limbs * (orderA_limbs + 1)) >> 1), cost_crt = orderA_crt + int32_t(uint32_t(orderA_crt) >> 3);
  bool use_limbs = (alg == 'L' || alg == 'l') || (alg != 'C' && alg != 'c' && (orderA_limbs <= 3 || cost_limbs <= cost_crt));
  if (use_limbs) { alg = 'L'; u = (u <= 7) ? 7 : ((orderA_limbs << 3) - 2); return orderA_limbs; } else { alg = 'C'; u = (U8CRT::range[orderA_crt - 1] - bitsM) / 2; return orderA_crt; }
}

extern "C" hyacinPrecision_t hyacinXGautoType(int32_t g_corr, int32_t globalM, hyacinPrecision_t Atype, int32_t u, int32_t* gElemBytes) {
  hyacinPrecision_t AtypeReal = real_type[int32_t(Atype)];
  int32_t bits = (g_corr + 1) + int32_t(std::ceil(0.25 * std::log2(double(std::max(1, globalM))))) + (u + u);
  hyacinPrecision_t GtypeReal = 
    (bits <= type_mantissa[int32_t(HYACIN_F32)] && (AtypeReal == HYACIN_F32 || AtypeReal == HYACIN_F16)) ? HYACIN_F32 : (
    bits <= type_mantissa[int32_t(HYACIN_F64)] ? HYACIN_F64 : (
    (bits <= type_mantissa[int32_t(HYACIN_QF)] && !internal::device_is_f64_capable()) ? HYACIN_QF : HYACIN_DD));
  hyacinPrecision_t Gtype = (Atype == AtypeReal) ? GtypeReal : complex_type[int32_t(GtypeReal)];
  if (gElemBytes) { *gElemBytes = type_bytes[Gtype]; }
  return Gtype;
}

inline void gemm_accum(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, lda, A, CUDA_R_8I, lda,
      &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
    internal::int8::accumulate_i32tensor(stream, 'A', beta, N, sft, orderA, W, M, orderC, C);
  } else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C);
    } else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, orderA, W, M, orderC, C);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, lda, &AN_k[iter_h], CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'A', 1, N, sft, orderA, W, M, orderC, C);
    }
  }
}

inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(lda), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, lda, strideA, A, CUDA_R_8I, lda, strideA,
      &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
    internal::int8::accumulate_i32tensor(stream, 'T', beta, N, 0, orderA, W, M, orderC, C);
  } else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, lda, strideA, A_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? 'T' : 'U', k == 0 ? beta : 1, N, 0, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, lda, strideA, A_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', range_k == 0 ? beta : 1, N, 0, orderA, W, M, orderC, C);
    } else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, A_k, CUDA_R_8I, lda, strideA, A_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', range_k == 0 ? beta : 1, N, 0, orderA, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, lda, strideA, &A_k[iter_h], CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_i32tensor(stream, 'U', 1, N, 0, orderA, W, M, orderC, C);
    }
  }
}

inline void i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t lda, int32_t orderC, uint64_t* C, int32_t* W) {
  int64_t strideA = int64_t(N) * int64_t(lda);
  for (int32_t i = 0; i < orderA; ++i)
    gemm_accum(stream, handle, M, N, K, i << 3, orderA, &AT[int64_t(i) * strideA], A, lda, int32_t(0 < i), orderC, C, W);
}

inline void i8GemmU(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t orderC, uint64_t* C, int32_t* W) {
  int64_t strideA = int64_t(N) * int64_t(lda);
  for (int32_t i = 1; i < orderA; ++i)
    gemm_accum(stream, handle, M, N, K, i << 3, i, &A[int64_t(i) * strideA], A, lda, int32_t(1 < i), orderC, C, W);
  gemm_accum_diag(stream, handle, M, N, K, orderA, A, lda, int32_t(1 < orderA), orderC, C, W);
}

template <int32_t Complex>
inline void i8herk_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t bits = Complex ? 62 : 60;
  int32_t orderB = (bits + int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (orderA << 4)) / 63, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideA = int64_t(lda) * colsA, strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB); if constexpr(Complex) { b_len += b_len; }

  int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&B, b_len * sizeof(uint64_t), stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  if constexpr(Complex) {
    if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), uint64_t(colsA) * uint64_t(3), stream); }
    int64_t strideA2 = strideA + strideA;
    i8GemmU(stream, handle, algnN, N, algnM, orderA, &A[strideA2], lda, orderB, B, scratch);
    i8GemmF(stream, handle, algnN, N, algnM, orderA, A, &A[strideA], lda, orderB, &B[strideB], scratch);
  } else {
    if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), uint64_t(colsA), stream); }
    i8GemmU(stream, handle, algnN, N, algnM, orderA, A, lda, orderB, B, scratch);
  }
  cudaFreeAsync(scratch, stream);

  if constexpr(Complex) { internal::int8::triangle_pack(stream, 0, N, orderB, B, (const ulonglong4_32a*)nullptr, uint32_t(0), beta, orderC, C); }
    else { internal::int8::triangle_pack(stream, 0, N, orderB, B, (const ulonglong2*)nullptr, uint32_t(0), beta, orderC, C); }
  cudaFreeAsync(B, stream);
}

inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, 0, N, orderA, W, M, orderC, C);
  } else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < k), N, orderA, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
    } else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, cublas_algo);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, W, M, orderC, C);
    }
  }
}

template <class matrix_t>
inline void i8herk_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, const int32_t* vexp, uint32_t corr, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  using sum_t = std::conditional_t<Complex, ulonglong4_32a, ulonglong2>;
  int32_t orderB = (U8CRT::range[orderA - 1] + 63) / 63, algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideW = int64_t(algnM) * colsA, strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB), w_len = uint64_t(algnN - N) * uint64_t(algnM);
  if constexpr(Complex) { b_len += b_len; w_len += uint64_t(3) * uint64_t(strideW); } else { w_len += uint64_t(strideW); }

  int8_t* W = nullptr; int32_t* scratch = nullptr; uint64_t* B = nullptr; sum_t *vsum = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&W, w_len, stream))
    throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&B, b_len * sizeof(uint64_t), stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");
   if (cudaSuccess != cudaMallocAsync((void**)&vsum, uint64_t(N) * sizeof(sum_t), stream))
    throw std::runtime_error("Workspace (vsum) allocation failed at Integer SY/HERK.");

  internal::int8::quantize_crt(stream, M, N, orderA, A, lda, corr, vexp, W, algnM, vsum);
  if constexpr(Complex) {
    if (M < algnM) { cudaMemset2DAsync(&W[M], uint64_t(algnM), 0, uint64_t(algnM - M), uint64_t(colsA) * uint64_t(3), stream); }
    int64_t strideW2 = strideW + strideW;
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, orderA, &W[strideW2], &W[strideW2], orderB, B, scratch);
    gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, orderA, W, &W[strideW], orderB, &B[strideB], scratch);
  } else {
    if (M < algnM) { cudaMemset2DAsync(&W[M], uint64_t(algnM), 0, uint64_t(algnM - M), uint64_t(colsA), stream); }
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, orderA, W, W, orderB, B, scratch);
  }
  cudaFreeAsync(W, stream); cudaFreeAsync(scratch, stream);

  internal::int8::triangle_pack(stream, M, N, orderB, B, vsum, corr, beta, orderC, C);
  cudaFreeAsync(B, stream); cudaFreeAsync(vsum, stream);
}

template <class matrix_t>
inline void herk_dispatcher(cudaStream_t stream, cublasHandle_t handle, char alg, int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, int32_t* uptr) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  constexpr uint64_t elem = uint64_t(Complex ? sizeof(uint64_t) : sizeof(uint32_t));
  int32_t u = *uptr, i = *beta; *beta = 1;
  if (u == HYACIN_QUERY_U && 0 < M) { internal::int8::vector_range(stream, M, N, A, lda, uptr, vexp); u = *uptr; }
  if (u < 0 && i == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); return; }

  int32_t uc = u + Complex, orderA = internal::int8::gram_algorithm(alg, M, uc);
  if (alg == 'L') {
    int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
    uint64_t strideA = uint64_t(algnM) * uint64_t(N) * uint64_t(orderA), a_len = uint64_t(algnM) * uint64_t(algnN - N);
    if constexpr(Complex) { a_len += strideA * uint64_t(3); } else { a_len += strideA; }

    int8_t* W = nullptr;
    if (cudaSuccess != cudaMallocAsync((void**)&W, a_len, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK");

    internal::int8::quantize_limbs(stream, M, N, orderA, A, lda, vexp, W, algnM);
    i8herk_limbs<Complex>(stream, handle, M, N, orderA, W, algnM, i, orderC, C);
    cudaFreeAsync(W, stream);
  } else { i8herk_crt(stream, handle, M, N, orderA, A, lda, vexp, uint32_t(uc), i, orderC, C); }
}

extern "C" void hyacinXherk(hyacinHandle_t handle, char alg, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t u_hint, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  int32_t* uptr = (int32_t*)handle.pinnedWorkspace; *uptr = u_hint;
  switch(Atype) {
    case HYACIN_F64: herk_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const double*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F32: herk_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const float*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F16: herk_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const __half*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F64_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const cuDoubleComplex*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F32_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const cuComplex*)A, lda, vexp, &beta, orderC, C, uptr); return;
    case HYACIN_F16_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const __half2*)A, lda, vexp, &beta, orderC, C, uptr); return;
    default: return;
  }
}

extern "C" void hyacinXherkBatchCreate(void** param, char alg, double epi, int32_t u_corr, int32_t batchK, int32_t N, hyacinPrecision_t Atype, uint64_t* bytesBatch) {
  int32_t panels, Complex = int32_t(Atype != real_type[int32_t(Atype)]), elemBytes = type_bytes[int32_t(Atype)];
  double epi_nrm = std::min(1., std::max(std::abs(epi), std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  int32_t u = Complex + std::min(u_practical_limit, u_corr + int32_t(std::ceil(-std::log2(epi_nrm))));
  Batch::BatchArgs* p = (Batch::BatchArgs*)(*param = new Batch::BatchArgs(alg, u, batchK, Complex, elemBytes, panels));
  if (bytesBatch) { *bytesBatch = uint64_t(N) * uint64_t(p->batchMaxK) * uint64_t(panels); }
}

extern "C" void hyacinXherkBatchDestroy(void* param) {
  if (param) { delete reinterpret_cast<Batch::BatchArgs*>(param); }
}

template <class matrix_t>
inline void herk_batch_dispatcher(cudaStream_t stream, cublasHandle_t handle, char alg, int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, int32_t* uptr, Batch::BatchArgs* param, int8_t* batch) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  int32_t uc = *uptr, orderA, row, ldw = param->batchMaxK; int64_t prefix_elem; char op;
  if (uc == HYACIN_QUERY_U && 0 < M) { internal::int8::vector_range(stream, M, N, A, lda, uptr, vexp); uc = *uptr + Complex; } else { uc += Complex; }
  std::tie(uc, orderA, prefix_elem, row, alg) = param->processA(M, N, uc, alg, op);
  int8_t* W = &batch[prefix_elem];

  switch(op) {
    case 'E': {
      herk_dispatcher(stream, handle, alg, M, N, A, lda, vexp, beta, orderC, C, uptr);
    } break;
    case 'Z': {
      if (alg == 'L') { internal::int8::quantize_limbs(stream, M, N, orderA, A, lda, vexp, &W[row], ldw); } else
      if (alg == 'C') { internal::scatter_matcopy(stream, handle, 'A', M, N, nullptr, A, lda, &((matrix_t*)W)[row], ldw); }
    } break;
    case 'F': { int32_t i = *beta; *beta = 1;
      if (alg == 'L') {
        i8herk_limbs<Complex>(stream, handle, row, N, orderA, W, ldw, i, orderC, C);
        internal::int8::quantize_limbs(stream, M, N, orderA, A, lda, vexp, W, ldw);
      } else if (alg == 'C') {
        i8herk_crt(stream, handle, row, N, orderA, (const matrix_t*)W, ldw, vexp, uint32_t(uc), i, orderC, C);
        internal::scatter_matcopy(stream, handle, 'A', M, N, nullptr, A, lda, (matrix_t*)W, ldw);
      }
    } break;
    case 'S': default: break;
  }
}

extern "C" void hyacinXherkBatchProcessA(hyacinHandle_t handle, char alg, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t u_hint, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, void* param, int8_t* batch) {
  if (M <= 0 || N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  int32_t* uptr = (int32_t*)handle.pinnedWorkspace; *uptr = u_hint;
  switch(Atype) {
    case HYACIN_F64: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const double*)A, lda, vexp, beta, orderC, C, uptr, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F32: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const float*)A, lda, vexp, beta, orderC, C, uptr, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F16: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const __half*)A, lda, vexp, beta, orderC, C, uptr, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F64_COMPLEX: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const cuDoubleComplex*)A, lda, vexp, beta, orderC, C, uptr, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F32_COMPLEX: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const cuComplex*)A, lda, vexp, beta, orderC, C, uptr, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F16_COMPLEX: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, alg, M, N, (const __half2*)A, lda, vexp, beta, orderC, C, uptr, (Batch::BatchArgs*)param, batch); return;
    default: return;
  }
}

template <class matrix_t>
inline void herk_flush_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t N, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, Batch::BatchArgs* param, int8_t* batch) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  int32_t lda = param->batchMaxK; std::vector<std::tuple<int32_t, int32_t, int64_t, int32_t, char>> list; param->flush(N, list);
  for (const auto& [uc, orderA, prefix_elem, M, alg] : list) {
    if (alg == 'L') { i8herk_limbs<Complex>(stream, handle, M, N, orderA, &batch[prefix_elem], lda, beta, orderC, C); beta = 1; } else
    if (alg == 'C') { i8herk_crt(stream, handle, M, N, orderA, (const matrix_t*)(&batch[prefix_elem]), lda, vexp, uint32_t(uc), beta, orderC, C); beta = 1; }
  }
}

extern "C" void hyacinXherkBatchFlush(hyacinHandle_t handle, int32_t N, hyacinPrecision_t Atype, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, void* param, int8_t* batch) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch(Atype) {
    case HYACIN_F64: herk_flush_dispatcher<double>(handle.cudaStream, handle.cublasHandle, N, vexp, beta, orderC, C, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F32: herk_flush_dispatcher<float>(handle.cudaStream, handle.cublasHandle, N, vexp, beta, orderC, C, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F16: herk_flush_dispatcher<__half>(handle.cudaStream, handle.cublasHandle, N, vexp, beta, orderC, C, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F64_COMPLEX: herk_flush_dispatcher<cuDoubleComplex>(handle.cudaStream, handle.cublasHandle, N, vexp, beta, orderC, C, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F32_COMPLEX: herk_flush_dispatcher<cuComplex>(handle.cudaStream, handle.cublasHandle, N, vexp, beta, orderC, C, (Batch::BatchArgs*)param, batch); return;
    case HYACIN_F16_COMPLEX: herk_flush_dispatcher<__half2>(handle.cudaStream, handle.cublasHandle, N, vexp, beta, orderC, C, (Batch::BatchArgs*)param, batch); return;
    default: return;
  }
}

