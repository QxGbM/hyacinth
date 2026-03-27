
#include <hyacin.h>
#include <internal.hpp>
#include <cuComplex.h>

#include <numeric>
#include <tuple>

inline std::pair<hyacinPrecision_t, int64_t> real_precision(hyacinPrecision_t prec) {
  switch(prec) {
    case HYACIN_F64: case HYACIN_F64_COMPLEX: return std::make_pair(HYACIN_F64, sizeof(double));
    case HYACIN_F32: case HYACIN_F32_COMPLEX: return std::make_pair(HYACIN_F32, sizeof(float));
    case HYACIN_DD: case HYACIN_DD_COMPLEX: return std::make_pair(HYACIN_DD, sizeof(double2));
    case HYACIN_QF: case HYACIN_QF_COMPLEX: return std::make_pair(HYACIN_QF, sizeof(float4));
    default: return std::make_pair(hyacinPrecision_t(0), int64_t(0));
  }
}

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t> ext_params(int32_t localM, int32_t globalM, int32_t N, int32_t& umax, int32_t distribute, hyacinPrecision_t Gtype, hyacinAlgorithm_t alg) {
  hyacinPrecision_t GtypeReal; int64_t em_bytes;
  std::tie(GtypeReal, em_bytes) = real_precision(Gtype);
  int32_t Complex = int32_t(Gtype != GtypeReal);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS || alg == HYACIN_ALG_LIMBS_ND);
  int32_t det_reduc = distribute && (alg == HYACIN_ALG_LIMBS || alg == HYACIN_ALG_CRT);

  int32_t algnM = (localM + 255) & (~255);
  int32_t algnN = (N + 63) & (~63);
  int32_t bits_M = std::max(1, det_reduc ? globalM : localM);
  int32_t bits_E = int32_t(std::ceil(std::log2(double(bits_M)))) + 2 + Complex;
  umax = use_limbs ? (((umax + 10) & (~7)) - 3) : ((((bits_E + (umax << 1)) | 7) - bits_E) / 2);
  int32_t bits = bits_E + (umax << 1);

  int32_t orderA = ((use_limbs ? umax : bits) + 8) >> 3;
  int64_t i8_bytes = int64_t(N) * int64_t(use_limbs ? orderA : 8) * ((int64_t(algnM) << Complex) + (int64_t(algnN) * sizeof(int32_t)));
  if (distribute && (!det_reduc)) i8_bytes = std::max(i8_bytes, (int64_t(N) * int64_t(N) * em_bytes) << Complex);

  int32_t orderC = ((use_limbs ? bits : (orderA << 3)) + 63) / 63;
  int32_t orderD = (orderC + det_reduc) << Complex;
  int64_t acc_bytes = int64_t(algnN) * int64_t(N + 1) * int64_t(orderD) * sizeof(uint64_t);
  int64_t vec_bytes = int64_t(algnN) * sizeof(int32_t);
  return std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes);
}

inline void vexp_dispatcher(cudaStream_t stream, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* vexp) {
  switch(Atype) {
    case HYACIN_F64:
      internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, vexp); break;
    case HYACIN_F32:
      internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, vexp); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::vexp_cf64(stream, M, N, (const std::complex<double>*)A, lda, vexp); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::vexp_cf32(stream, M, N, (const std::complex<float>*)A, lda, vexp); break;
    default: break;
  }
}

inline void igemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, 
  int32_t umax, const int32_t* vexp, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* acc, int32_t algnN, int8_t* iA, hyacinAlgorithm_t alg) {
  if (alg == HYACIN_ALG_LIMBS || alg == HYACIN_ALG_LIMBS_ND) switch(Atype) {
    case HYACIN_F64:
      internal::int8::i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32:
      internal::int8::i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    default: break;
  }
  else if (alg == HYACIN_ALG_CRT || alg == HYACIN_ALG_CRT_ND) switch(Atype) {
    case HYACIN_F64:
      internal::int8::i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32:
      internal::int8::i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); break;
    default: break;
  }
}

inline void deq_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t algnN, int32_t umax, const uint64_t* acc, int32_t bits, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t Gtype) {
  switch (Gtype) {
    case HYACIN_F64:
      internal::int8::dequantize_i63_f64(stream, bits, order, M, N, acc, algnN, umax, vexp, (double*)G, ldg); break;
    case HYACIN_F32:
      internal::int8::dequantize_i63_f32(stream, bits, order, M, N, acc, algnN, umax, vexp, (float*)G, ldg); break;
    case HYACIN_DD:
      internal::int8::dequantize_i63_f128_dd(stream, bits, order, M, N, acc, algnN, umax, vexp, (double2*)G, ldg); break;
    case HYACIN_QF:
      internal::int8::dequantize_i63_f128_qf(stream, bits, order, M, N, acc, algnN, umax, vexp, (float4*)G, ldg); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::dequantize_i63_cf64(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<double>*)G, ldg); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::dequantize_i63_cf32(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<float>*)G, ldg); break;
    case HYACIN_DD_COMPLEX:
      internal::int8::dequantize_i63_cf128_dd(stream, bits, order, M, N, acc, algnN, umax, vexp, (complex_double2*)G, ldg); break;
    case HYACIN_QF_COMPLEX:
      internal::int8::dequantize_i63_cf128_qf(stream, bits, order, M, N, acc, algnN, umax, vexp, (complex_float4*)G, ldg); break;
    default: break;
  }
}

extern "C" void hyacinXsyherk_bufferSize(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Gtype, hyacinAlgorithm_t alg, uint64_t* dev_work_bytes) {
  if (M <= 0 || N <= 0) { *dev_work_bytes = uint64_t(0); return; }
  int32_t Complex, det_reduc, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(M, M, N, umax, 0, Gtype, alg);
  *dev_work_bytes = uint64_t(i8_bytes + acc_bytes + vec_bytes);
}

extern "C" void hyacinXsyherk1Drow_bufferSize(int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Gtype, hyacinAlgorithm_t alg, uint64_t* dev_work_bytes) {
  if (globalM <= 0 || N <= 0) { *dev_work_bytes = uint64_t(0); return; }
  int32_t Complex, det_reduc, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(localM, globalM, N, umax, 1, Gtype, alg);
  *dev_work_bytes = uint64_t(i8_bytes + acc_bytes + vec_bytes);
}

extern "C" void hyacinXsyherk(cublasHandle_t handle, int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, void* dev_work, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0) { return; }
  int32_t Complex, det_reduc, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(M, M, N, umax, 0, Gtype, alg);

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes], *vexp = &acc[acc_bytes];

  vexp_dispatcher(stream, M, N, Atype, A, lda, (int32_t*)vexp);
  igemm_dispatcher(stream, handle, M, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
  deq_dispatcher(stream, M, N, algnN, umax, (uint64_t*)acc, 63, orderC, G, ldg, (const int32_t*)vexp, Gtype);
}

#ifndef NO_NCCL

inline void deq_nd_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const uint64_t* acc, int32_t bits, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t Gtype, void* dev_work, ncclComm_t col_comm) {
  int64_t elements = int64_t(N) * int64_t(N);
  void* A = (ldg == N) ? G : dev_work;
  switch (Gtype) {
    case HYACIN_F64:
      internal::int8::dequantize_i63_f64(stream, bits, order, M, N, acc, algnN, umax, vexp, (double*)A, N);
      ncclAllReduce(A, A, elements, ncclFloat64, ncclSum, col_comm, stream);
      if (A != G) { double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &one, (double*)A, N, &zero, (double*)G, ldg, (double*)G, ldg); }
      break;
    case HYACIN_F32:
      internal::int8::dequantize_i63_f32(stream, bits, order, M, N, acc, algnN, umax, vexp, (float*)A, N);
      ncclAllReduce(A, A, elements, ncclFloat, ncclSum, col_comm, stream);
      if (A != G) { float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &one, (float*)A, N, &zero, (float*)G, ldg, (float*)G, ldg); }
      break;
    case HYACIN_F64_COMPLEX:
      internal::int8::dequantize_i63_cf64(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<double>*)A, N);
      ncclAllReduce(A, A, elements << 1, ncclFloat64, ncclSum, col_comm, stream);
      if (A != G) { cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &one, (cuDoubleComplex*)A, N, &zero, (cuDoubleComplex*)G, ldg, (cuDoubleComplex*)G, ldg); }
      break;
    case HYACIN_F32_COMPLEX:
      internal::int8::dequantize_i63_cf32(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<float>*)A, N);
      ncclAllReduce(A, A, elements << 1, ncclFloat, ncclSum, col_comm, stream);
      if (A != G) { cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &one, (cuComplex*)A, N, &zero, (cuComplex*)G, ldg, (cuComplex*)G, ldg); }
      break;
    default: break;
  }
}

extern "C" void hyacinXsyherk1Drow(cublasHandle_t handle, int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, void* dev_work, hyacinAlgorithm_t alg, ncclComm_t col_comm) {
  if (globalM <= 0 || N <= 0) { return; }
  int32_t Complex, det_reduc, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(localM, globalM, N, umax, 1, Gtype, alg);

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes], *vexp = &acc[acc_bytes];
  vexp_dispatcher(stream, localM, N, Atype, A, lda, (int32_t*)vexp);

  if (det_reduc) {
    int64_t stride = int64_t(algnN) * int64_t(N) + int64_t(algnN), acc_len = stride * (int64_t(orderC + 1) << Complex);
    ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMax, col_comm, stream);
    if (0 < localM) { 
      igemm_dispatcher(stream, handle, localM, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
      internal::int8::accumulate_conv_i63_u47(stream, orderC, Complex, stride, (uint64_t*)acc);
    }
    else { cudaMemsetAsync(acc, 0, acc_bytes, stream); }
    ncclAllReduce(acc, acc, acc_len, ncclUint64, ncclSum, col_comm, stream);
    deq_dispatcher(stream, globalM, N, algnN, umax, (uint64_t*)acc, 47, orderC, G, ldg, (const int32_t*)vexp, Gtype);
  }
  else {
    if (0 < localM)
      igemm_dispatcher(stream, handle, localM, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
    deq_nd_dispatcher(stream, handle, localM, N, algnN, umax, (uint64_t*)acc, 63, orderC, G, ldg, (const int32_t*)vexp, Gtype, iA, col_comm);
  }
}

#endif
