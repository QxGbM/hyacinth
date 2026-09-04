
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <crt_constants.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>

const int32_t u_practical_limit = 80; // u <= 80 to satisfy implementation assumptions
// mappings vector for datatypes
const std::vector<hyacinPrecision_t> real_type({ HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF, HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF });
const std::vector<hyacinPrecision_t> complex_type({ HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX, HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX });
const std::vector<int32_t> type_bytes({ sizeof(double), sizeof(float), sizeof(__half), sizeof(double2), sizeof(float4), sizeof(cuDoubleComplex), sizeof(cuComplex), sizeof(half2), sizeof(complex_double2), sizeof(complex_float4) });
const std::vector<int32_t> type_mantissa({ 52, 23, 10, 105, 95, 52, 23, 10, 105, 95 });

extern "C" int32_t hyacinXquantizeScale(hyacinHandle_t handle, double epi, int32_t u_corr, int32_t globalM, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* vexp, int32_t* dimC) {
  Timer::register_kernel(handle.cudaStream, handle.timer);
  double epi_nrm = std::min(1., std::max(std::abs(epi), std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  int32_t u = std::min(u_practical_limit, u_corr + int32_t(std::ceil(-std::log2(epi_nrm))));
  if (dimC) { dimC[0] = 1 + int32_t(Atype != real_type[int32_t(Atype)]); dimC[1] = ((dimC[0] + 61) + int32_t(std::ceil(std::log2(double(std::max(1, globalM))))) + (u << 1)) / 63; }
  if (0 < u) { switch(Atype) {
    case HYACIN_F64: internal::int8::vector_exponents(handle.cudaStream, M, N, (const double*)A, lda, &u, vexp); break;
    case HYACIN_F32: internal::int8::vector_exponents(handle.cudaStream, M, N, (const float*)A, lda, &u, vexp); break;
    case HYACIN_F16: internal::int8::vector_exponents(handle.cudaStream, M, N, (const __half*)A, lda, &u, vexp); break;
    case HYACIN_F64_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, M, N, (const cuDoubleComplex*)A, lda, &u, vexp); break;
    case HYACIN_F32_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, M, N, (const cuComplex*)A, lda, &u, vexp); break;
    case HYACIN_F16_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, M, N, (const __half2*)A, lda, &u, vexp); break;
    default: u = 0; break; }
#ifndef NO_NCCL
    if (handle.col_comm) { Timer::register_comm(handle.cudaStream, handle.timer); ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMin, handle.col_comm, handle.cudaStream); }
#endif
  } return u;
}

extern "C" hyacinPrecision_t hyacinXGautoType(int32_t g_corr, int32_t globalM, hyacinPrecision_t Atype, int32_t u, int32_t* gElemBytes) {
  hyacinPrecision_t AtypeReal = real_type[int32_t(Atype)];
  int32_t bits = (g_corr + 1) + int32_t(std::ceil(0.5 * std::log2(double(std::max(1, globalM))))) + (u << 1);
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
      &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'A', beta, N, sft, 8, orderA, W, M, orderC, C);
  } else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', k == 0 ? beta : 1, N, sft, 8, orderA, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, W, M, orderC, C);
    } else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_h, &one, AT_k, CUDA_R_8I, lda, AN_k, CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', range_k == 0 ? beta : 1, N, sft, 8, orderA, W, M, orderC, C);
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, lda, &AN_k[iter_h], CUDA_R_8I, lda,
        &zero, W, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'A', 1, N, sft, 8, orderA, W, M, orderC, C);
    }
  }
}

inline void gemm_accum_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(lda), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, lda, strideA, A, CUDA_R_8I, lda, strideA,
      &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, 'T', beta, N, 0, 16, orderA, W, M, orderC, C);
  } else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, lda, strideA, A_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? 'T' : 'U', k == 0 ? beta : 1, N, 0, 16, orderA, W, M, orderC, C);
    }

    const int8_t* A_k = &A[range_k];
    if (rem <= iter_k) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, A_k, CUDA_R_8I, lda, strideA, A_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', range_k == 0 ? beta : 1, N, 0, 16, orderA, W, M, orderC, C);
    } else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, A_k, CUDA_R_8I, lda, strideA, A_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, range_k == 0 ? 'T' : 'U', range_k == 0 ? beta : 1, N, 0, 16, orderA, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, lda, strideA, &A_k[iter_h], CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, 'U', 1, N, 0, 16, orderA, W, M, orderC, C);
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
    gemm_accum(stream, handle, M, N, K, (i << 4) - 8, orderA - i, &A[int64_t(i - 1) * strideA], &A[int64_t(i) * strideA], lda, int32_t(1 < i), orderC, C, W);
  gemm_accum_diag(stream, handle, M, N, K, orderA, A, lda, int32_t(1 < orderA), orderC, C, W);
}

template <int32_t Complex>
inline std::pair<int32_t, uint64_t> i8_size(int32_t K, int32_t N, int32_t orderA) {
  int32_t algnM = (K + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA);
  uint64_t a_len = uint64_t(strideA), a_pad = uint64_t(algnN - N) * uint64_t(algnM);
  if constexpr(Complex) { return std::make_pair(algnM, (a_len * uint64_t(3)) + a_pad); }
    else { return std::make_pair(algnM, a_len + a_pad); }
}

template <int32_t Complex>
inline void i8herk_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t bits = Complex ? 62 : 60;
  int32_t orderB = (bits + int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (orderA << 4)) / 63;
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideA = int64_t(lda) * colsA, strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB); if constexpr(Complex) { b_len *= uint64_t(2); }

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

  internal::int8::triangle_pack<Complex>(stream, 0, N, orderB, B, nullptr, uint32_t(0), beta, orderC, C);
  cudaFreeAsync(B, stream);
}

inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t lda, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(lda), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, lda, strideA, A, CUDA_R_8I, lda, strideA,
      &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, 0, N, orderA, W, M, orderC, C);
  } else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, lda, strideA, AN_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < k), N, orderA, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, lda, strideA, AN_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
    } else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, lda, strideA, AN_k, CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, int32_t(0 < range_k), N, orderA, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, lda, strideA, &AN_k[iter_h], CUDA_R_8I, lda, strideA,
        &zero, W, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, W, M, orderC, C);
    }
  }
}

template <int32_t Complex>
inline void i8herk_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, int8_t* A, int32_t lda, const uint64_t* vsum, uint32_t corr, int32_t beta, int32_t orderC, uint64_t* C) {
  int32_t orderB = (63 + (orderA << 3)) / 63;
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t colsA = int64_t(N) * int64_t(orderA), strideA = int64_t(lda) * colsA, strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t b_len = uint64_t(strideB); if constexpr(Complex) { b_len *= uint64_t(2); }

  int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(colsA) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&B, b_len * sizeof(uint64_t), stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  if constexpr(Complex) {
    if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), uint64_t(colsA) * uint64_t(3), stream); }
    int64_t strideA2 = strideA * int64_t(2);
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, orderA, &A[strideA2], &A[strideA2], lda, orderB, B, scratch);
    gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, orderA, A, &A[strideA], lda, orderB, &B[strideB], scratch);
  } else {
    if (M < algnM) { cudaMemset2DAsync(&A[M], uint64_t(lda), 0, uint64_t(algnM - M), uint64_t(colsA), stream); }
    gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, orderA, A, A, lda, orderB, B, scratch);
  }
  cudaFreeAsync(scratch, stream);

  internal::int8::triangle_pack<Complex>(stream, M, N, orderB, B, vsum, corr, beta, orderC, C);
  cudaFreeAsync(B, stream);
}

template <class matrix_t>
inline void herk_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vexp, int32_t* beta, int32_t orderC, uint64_t* C, int32_t* uptr, hyacinAlgorithm_t alg) {
  constexpr int32_t Complex = int32_t(std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>);
  constexpr uint64_t elem = uint64_t(Complex ? sizeof(uint64_t) : sizeof(uint32_t));
  int32_t u = *uptr, i = *beta; *beta = 1;
  if (u <= 0 && 0 < M) { internal::int8::vector_exponents(stream, M, N, A, lda, uptr, const_cast<int32_t*>(vexp)); u = *uptr; }
  if (u <= 0 && i == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); return; }

  int32_t uc = u + Complex, bits = int32_t(std::ceil(std::log2(double(std::max(1, M))))) + ((uc + 1) << 1);
  int32_t orderA_limbs = (uc <= 7) ? 1 : int32_t(uint32_t(uc + 9) >> 3);
  int32_t orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::find_if(&U8CRT::range[0], &U8CRT::range[23], [=](int32_t r) { return bits <= r; })));
  int32_t cost_limbs = int32_t(uint32_t(orderA_limbs * (orderA_limbs + 1)) >> 1), cost_crt = orderA_crt + int32_t(uint32_t(orderA_crt) >> 3);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS || (alg == HYACIN_ALG_AUTO && (orderA_limbs <= 3 || cost_limbs <= cost_crt)));
  if (use_limbs) {
    int32_t ldw; uint64_t w_len; std::tie(ldw, w_len) = i8_size<Complex>(M, N, orderA_limbs);
    int8_t* W = nullptr;
    if (cudaSuccess != cudaMallocAsync((void**)&W, w_len, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");

    internal::int8::quantize(stream, M, A, lda, uint32_t(0), vexp, orderA_limbs, N, ldw, W);
    i8herk_limbs<Complex>(stream, handle, M, N, orderA_limbs, W, ldw, i, orderC, C);
    cudaFreeAsync(W, stream);
  } else {
    int32_t ldw; uint64_t w_len; std::tie(ldw, w_len) = i8_size<Complex>(M, N, orderA_crt);
    int8_t* W = nullptr; uint64_t* vsum = nullptr;
    if (cudaSuccess != cudaMallocAsync((void**)&W, w_len, stream))
      throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
    if (cudaSuccess != cudaMallocAsync((void**)&vsum, uint64_t(N) * uint64_t(4) * elem, stream))
      throw std::runtime_error("Workspace (Sums) allocation failed at Integer SY/HERK.");

    internal::int8::quantize(stream, M, A, lda, u, vexp, orderA_crt, N, ldw, W);
    internal::int8::vector_sums(stream, M, N, A, lda, u, vexp, vsum);
    i8herk_crt<Complex>(stream, handle, M, N, orderA_crt, W, ldw, vsum, u, i, orderC, C);
    cudaFreeAsync(W, stream); cudaFreeAsync(vsum, stream);
  }
}

extern "C" void hyacinXherk(hyacinHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t u_hint, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C, hyacinAlgorithm_t alg) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  int32_t* uptr = (int32_t*)handle.pinnedWorkspace; *uptr = u_hint;
  switch(Atype) {
    case HYACIN_F64: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const double*)A, lda, vexp, &beta, orderC, C, uptr, alg); return;
    case HYACIN_F32: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const float*)A, lda, vexp, &beta, orderC, C, uptr, alg); return;
    case HYACIN_F16: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half*)A, lda, vexp, &beta, orderC, C, uptr, alg); return;
    case HYACIN_F64_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuDoubleComplex*)A, lda, vexp, &beta, orderC, C, uptr, alg); return;
    case HYACIN_F32_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuComplex*)A, lda, vexp, &beta, orderC, C, uptr, alg); return;
    case HYACIN_F16_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half2*)A, lda, vexp, &beta, orderC, C, uptr, alg); return;
    default: return;
  }
}

template <class Atype, class Btype>
inline void herk_batch_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const Atype* A, int32_t lda, const int32_t* vexp,
  int32_t batchK, int32_t Nbatches, const int32_t* batchU, int32_t* batchLoc, Btype* B, int32_t* beta, int32_t orderC, uint64_t* C, int32_t u, hyacinAlgorithm_t alg) {
  int32_t b = std::distance(batchU, std::find_if(batchU, &batchU[Nbatches], [=](int32_t ub) { return u <= ub; }));
  if (b < Nbatches) {
    int32_t K = batchLoc[b]; int64_t strideB = int64_t(batchK) * int64_t(N) * int64_t(b);
    if (batchK < (K + M)) { herk_dispatcher(stream, handle, K, N, &B[strideB], batchK, vexp, beta, orderC, C, &u, alg); batchLoc[b] = M; }
      else { strideB += int64_t(K); batchLoc[b] = K + M; }
    internal::scatter_matcopy(stream, handle, 'A', M, N, nullptr, A, lda, &B[strideB], batchK);
  } else { herk_dispatcher(stream, handle, M, N, A, lda, vexp, beta, orderC, C, &u, alg); }
}

template <class Atype>
inline void herk_batch_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const Atype* A, int32_t lda, const int32_t* vexp,
  int32_t batchK, int32_t Nbatches, const int32_t* batchU, int32_t* batchLoc, hyacinPrecision_t Btype, void* B, int32_t* beta, int32_t orderC, uint64_t* C, int32_t* uptr, hyacinAlgorithm_t alg) {
  int32_t u = *uptr;
  if (u <= 0) { internal::int8::vector_exponents(stream, M, N, A, lda, uptr, const_cast<int32_t*>(vexp)); if ((u = *uptr) <= 0) { return; }}
  if constexpr(std::is_same_v<Atype, cuDoubleComplex> || std::is_same_v<Atype, cuComplex> || std::is_same_v<Atype, __half2>) switch(Btype) {
    case HYACIN_F64_COMPLEX: herk_batch_dispatcher(stream, handle, M, N, A, lda, vexp, batchK, Nbatches, batchU, batchLoc, (cuDoubleComplex*)B, beta, orderC, C, u, alg); return;
    case HYACIN_F32_COMPLEX: herk_batch_dispatcher(stream, handle, M, N, A, lda, vexp, batchK, Nbatches, batchU, batchLoc, (cuComplex*)B, beta, orderC, C, u, alg); return;
    case HYACIN_F16_COMPLEX: herk_batch_dispatcher(stream, handle, M, N, A, lda, vexp, batchK, Nbatches, batchU, batchLoc, (__half2*)B, beta, orderC, C, u, alg); return;
    default: return;
  } else switch(Btype) {
    case HYACIN_F64: herk_batch_dispatcher(stream, handle, M, N, A, lda, vexp, batchK, Nbatches, batchU, batchLoc, (double*)B, beta, orderC, C, u, alg); return;
    case HYACIN_F32: herk_batch_dispatcher(stream, handle, M, N, A, lda, vexp, batchK, Nbatches, batchU, batchLoc, (float*)B, beta, orderC, C, u, alg); return;
    case HYACIN_F16: herk_batch_dispatcher(stream, handle, M, N, A, lda, vexp, batchK, Nbatches, batchU, batchLoc, (__half*)B, beta, orderC, C, u, alg); return;
    default: return;
  }
}

extern "C" void hyacinXherkBatch(hyacinHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t u_hint, const int32_t* vexp,
  int32_t batchK, int32_t Nbatches, const int32_t* batchU, int32_t* batchLoc, hyacinPrecision_t Btype, void* B, int32_t* beta, int32_t orderC, uint64_t* C, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  int32_t* uptr = (int32_t*)handle.pinnedWorkspace; *uptr = u_hint;
  if (batchK <= M || Nbatches <= 0) switch(Atype) {
    case HYACIN_F64: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const double*)A, lda, vexp, beta, orderC, C, uptr, alg); return;
    case HYACIN_F32: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const float*)A, lda, vexp, beta, orderC, C, uptr, alg); return;
    case HYACIN_F16: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half*)A, lda, vexp, beta, orderC, C, uptr, alg); return;
    case HYACIN_F64_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuDoubleComplex*)A, lda, vexp, beta, orderC, C, uptr, alg); return;
    case HYACIN_F32_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuComplex*)A, lda, vexp, beta, orderC, C, uptr, alg); return;
    case HYACIN_F16_COMPLEX: herk_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half2*)A, lda, vexp, beta, orderC, C, uptr, alg); return;
    default: return;
  } else switch(Atype) {
    case HYACIN_F64: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const double*)A, lda, vexp, batchK, Nbatches, batchU, batchLoc, Btype, B, beta, orderC, C, uptr, alg); return;
    case HYACIN_F32: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const float*)A, lda, vexp, batchK, Nbatches, batchU, batchLoc, Btype, B, beta, orderC, C, uptr, alg); return;
    case HYACIN_F16: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half*)A, lda, vexp, batchK, Nbatches, batchU, batchLoc, Btype, B, beta, orderC, C, uptr, alg); return;
    case HYACIN_F64_COMPLEX: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuDoubleComplex*)A, lda, vexp, batchK, Nbatches, batchU, batchLoc, Btype, B, beta, orderC, C, uptr, alg); return;
    case HYACIN_F32_COMPLEX: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const cuComplex*)A, lda, vexp, batchK, Nbatches, batchU, batchLoc, Btype, B, beta, orderC, C, uptr, alg); return;
    case HYACIN_F16_COMPLEX: herk_batch_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, (const __half2*)A, lda, vexp, batchK, Nbatches, batchU, batchLoc, Btype, B, beta, orderC, C, uptr, alg); return;
    default: return;
  }
}

template <class Btype>
inline void batch_flush_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t N, const int32_t* vexp,
  int32_t batchK, int32_t Nbatches, const int32_t* batchU, int32_t* batchLoc, const Btype* B, int32_t beta, int32_t orderC, uint64_t* C, hyacinAlgorithm_t alg) {
  int64_t strideB = int64_t(batchK) * int64_t(N);
  for (int32_t b = 0; b < Nbatches; ++b) {
    int32_t K = batchLoc[b], u = batchU[b]; batchLoc[b] = 0;
    if (0 < K && u < 0) { herk_dispatcher(stream, handle, K, N, &B[strideB * int64_t(b)], batchK, vexp, &beta, orderC, C, &u, alg); }
  }
  constexpr uint64_t elem = uint64_t((std::is_same_v<Btype, cuDoubleComplex> || std::is_same_v<Btype, cuComplex> || std::is_same_v<Btype, __half2>) ? sizeof(uint64_t) : sizeof(uint32_t));
  if (beta == 0) { cudaMemsetAsync(C, 0, uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * elem, stream); }
}

extern "C" void hyacinXherkBatchFlush(hyacinHandle_t handle, int32_t N, const int32_t* vexp,
  int32_t batchK, int32_t Nbatches, const int32_t* batchU, int32_t* batchLoc, hyacinPrecision_t Btype, const void* B, int32_t beta, int32_t orderC, uint64_t* C, hyacinAlgorithm_t alg) {
  if (N <= 0 || orderC <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch(Btype) {
    case HYACIN_F64: batch_flush_dispatcher(handle.cudaStream, handle.cublasHandle, N, vexp, batchK, Nbatches, batchU, batchLoc, (const double*)B, beta, orderC, C, alg); return;
    case HYACIN_F32: batch_flush_dispatcher(handle.cudaStream, handle.cublasHandle, N, vexp, batchK, Nbatches, batchU, batchLoc, (const float*)B, beta, orderC, C, alg); return;
    case HYACIN_F16: batch_flush_dispatcher(handle.cudaStream, handle.cublasHandle, N, vexp, batchK, Nbatches, batchU, batchLoc, (const __half*)B, beta, orderC, C, alg); return;
    case HYACIN_F64_COMPLEX: batch_flush_dispatcher(handle.cudaStream, handle.cublasHandle, N, vexp, batchK, Nbatches, batchU, batchLoc, (const cuDoubleComplex*)B, beta, orderC, C, alg); return;
    case HYACIN_F32_COMPLEX: batch_flush_dispatcher(handle.cudaStream, handle.cublasHandle, N, vexp, batchK, Nbatches, batchU, batchLoc, (const cuComplex*)B, beta, orderC, C, alg); return;
    case HYACIN_F16_COMPLEX: batch_flush_dispatcher(handle.cudaStream, handle.cublasHandle, N, vexp, batchK, Nbatches, batchU, batchLoc, (const __half2*)B, beta, orderC, C, alg); return;
    default: return;
  }
}
