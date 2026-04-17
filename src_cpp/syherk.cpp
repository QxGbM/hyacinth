
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <tuple>
#include <stdexcept>

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t> ext_params(int32_t localM, int32_t globalM, int32_t N, int32_t& umax, int32_t distribute, hyacinPrecision_t Gtype, hyacinAlgorithm_t alg) {
  hyacinPrecision_t GtypeReal; hyacinXelem('R', Gtype, &GtypeReal, nullptr, nullptr);
  int32_t Complex = int32_t(Gtype != GtypeReal);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS || alg == HYACIN_ALG_LIMBS_ND);
  int32_t det_reduc = distribute && (alg == HYACIN_ALG_LIMBS || alg == HYACIN_ALG_CRT);

  int32_t algnM = (localM + 255) & (~255);
  int32_t algnN = (N + 63) & (~63);
  int32_t bits_M = std::max(1, det_reduc ? globalM : localM);
  int32_t bits_E = int32_t(std::ceil(std::log2(double(bits_M)))) + (Complex ? 6 : 4);
  umax = use_limbs ? (((umax + 10) & (~7)) - 3) : ((((bits_E + (umax << 1)) | 7) - bits_E) / 2);
  int32_t bits = bits_E + (umax << 1);

  int32_t orderA = ((use_limbs ? umax : bits) + 8) >> 3;
  int64_t i8_bytes = int64_t(N) * int64_t(use_limbs ? orderA : 8) * ((int64_t(algnM) << Complex) + (int64_t(algnN) * sizeof(int32_t)));
  if (distribute && (alg == HYACIN_ALG_LIMBS_ND || alg == HYACIN_ALG_CRT_ND)) {
    int32_t em_bytes; hyacinXelem('A', Gtype, nullptr, &em_bytes, nullptr);
    i8_bytes = std::max(i8_bytes, (int64_t(N) * int64_t(N) * int64_t(em_bytes) + int64_t(255)) & int64_t(~255));
  }

  int32_t orderC = ((use_limbs ? bits : (orderA << 3)) + 63) / 63;
  int32_t orderD = (orderC + det_reduc) << Complex;
  int64_t acc_bytes = int64_t(algnN) * int64_t(N + 1) * int64_t(orderD) * sizeof(uint64_t);
  int64_t vec_bytes = int64_t(algnN) * sizeof(int32_t);
  return std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes);
}

inline void vexp_dispatcher(cudaStream_t stream, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* vexp) {
  switch(Atype) {
    case HYACIN_F64:
      internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, vexp); return;
    case HYACIN_F32:
      internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, vexp); return;
    case HYACIN_F64_COMPLEX:
      internal::int8::vexp_cf64(stream, M, N, (const std::complex<double>*)A, lda, vexp); return;
    case HYACIN_F32_COMPLEX:
      internal::int8::vexp_cf32(stream, M, N, (const std::complex<float>*)A, lda, vexp); return;
    default: return;
  }
}

inline void igemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, 
  int32_t umax, const int32_t* vexp, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* acc, int32_t algnN, int8_t* iA, hyacinAlgorithm_t alg) {
  if (alg == HYACIN_ALG_LIMBS || alg == HYACIN_ALG_LIMBS_ND) switch(Atype) {
    case HYACIN_F64:
      internal::int8::i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    case HYACIN_F32:
      internal::int8::i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    case HYACIN_F64_COMPLEX:
      internal::int8::i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    case HYACIN_F32_COMPLEX:
      internal::int8::i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    default: return;
  }
  else if (alg == HYACIN_ALG_CRT || alg == HYACIN_ALG_CRT_ND) switch(Atype) {
    case HYACIN_F64:
      internal::int8::i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    case HYACIN_F32:
      internal::int8::i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    case HYACIN_F64_COMPLEX:
      internal::int8::i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    case HYACIN_F32_COMPLEX:
      internal::int8::i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, vexp, algnM, orderA, orderC, acc, algnN, iA); return;
    default: return;
  }
}

inline void deq_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t algnN, int32_t umax, const uint64_t* acc, int32_t bits, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t Gtype) {
  switch (Gtype) {
    case HYACIN_F64:
      internal::int8::dequantize_i63_f64(stream, bits, order, M, N, acc, algnN, umax, vexp, (double*)G, ldg); return;
    case HYACIN_F32:
      internal::int8::dequantize_i63_f32(stream, bits, order, M, N, acc, algnN, umax, vexp, (float*)G, ldg); return;
    case HYACIN_DD:
      internal::int8::dequantize_i63_f128_dd(stream, bits, order, M, N, acc, algnN, umax, vexp, (double2*)G, ldg); return;
    case HYACIN_QF:
      internal::int8::dequantize_i63_f128_qf(stream, bits, order, M, N, acc, algnN, umax, vexp, (float4*)G, ldg); return;
    case HYACIN_F64_COMPLEX:
      internal::int8::dequantize_i63_cf64(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<double>*)G, ldg); return;
    case HYACIN_F32_COMPLEX:
      internal::int8::dequantize_i63_cf32(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<float>*)G, ldg); return;
    case HYACIN_DD_COMPLEX:
      internal::int8::dequantize_i63_cf128_dd(stream, bits, order, M, N, acc, algnN, umax, vexp, (complex_double2*)G, ldg); return;
    case HYACIN_QF_COMPLEX:
      internal::int8::dequantize_i63_cf128_qf(stream, bits, order, M, N, acc, algnN, umax, vexp, (complex_float4*)G, ldg); return;
    default: return;
  }
}

inline int32_t cublas_dispatcher(cublasHandle_t handle, int32_t M, int32_t N, const void* A, int32_t lda, void* G, int32_t ldg, hyacinPrecision_t type) {
  double one_f64 = 1., zero_f64 = 0.; float one_f32 = 1.f, zero_f32 = 0.f;
  switch (type) {
    case HYACIN_F64:
      cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N, M, &one_f64, (const double*)A, lda, &zero_f64, (double*)G, ldg); return 0;
    case HYACIN_F32:
      cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N, M, &one_f32, (const float*)A, lda, &zero_f32, (float*)G, ldg); return 0;
    case HYACIN_F64_COMPLEX:
      cublasZherk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, N, M, &one_f64, (const cuDoubleComplex*)A, lda, &zero_f64, (cuDoubleComplex*)G, ldg); return 0;
    case HYACIN_F32_COMPLEX:
      cublasCherk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, N, M, &one_f32, (const cuComplex*)A, lda, &zero_f32, (cuComplex*)G, ldg); return 0;
    default: return 0;
  }
}

extern "C" int32_t hyacinXsyherk(hyacinHandle_t handle, int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0) { return -1; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  if (alg == CUBLAS_FLOAT_ND) { return cublas_dispatcher(handle.cublasHandle, M, N, A, lda, G, ldg, Atype); }
  int32_t Complex, det_reduc, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(M, M, N, umax, 0, Gtype, alg);

  void* dev_work = nullptr;
  if (cudaSuccess != cudaMallocAsync(&dev_work, uint64_t(i8_bytes + acc_bytes + vec_bytes), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes], *vexp = &acc[acc_bytes];
  vexp_dispatcher(handle.cudaStream, M, N, Atype, A, lda, (int32_t*)vexp);
  igemm_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
  deq_dispatcher(handle.cudaStream, M, N, algnN, umax, (uint64_t*)acc, 63, orderC, G, ldg, (const int32_t*)vexp, Gtype);
  cudaFreeAsync(dev_work, handle.cudaStream);
  return umax;
}

#ifndef NO_NCCL

inline void all_reduce_in_place(cudaStream_t stream, cublasHandle_t handle, int32_t pred, int32_t M, int32_t N, double* A, int32_t lda, double* C, ncclComm_t comm, void* timer) {
  Timer::register_comm(stream, timer); ncclAllReduce(C, C, int64_t(M) * int64_t(N), ncclFloat64, ncclSum, comm, stream);
  if (pred) { double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, C, M, &zero, A, lda, A, lda); }
}

inline void all_reduce_in_place(cudaStream_t stream, cublasHandle_t handle, int32_t pred, int32_t M, int32_t N, float* A, int32_t lda, float* C, ncclComm_t comm, void* timer) {
  Timer::register_comm(stream, timer); ncclAllReduce(C, C, int64_t(M) * int64_t(N), ncclFloat32, ncclSum, comm, stream);
  if (pred) { float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, C, M, &zero, A, lda, A, lda); }
}

inline void all_reduce_in_place(cudaStream_t stream, cublasHandle_t handle, int32_t pred, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, cuDoubleComplex* C, ncclComm_t comm, void* timer) {
  Timer::register_comm(stream, timer); ncclAllReduce(C, C, int64_t(M) * int64_t(N) * int64_t(2), ncclFloat64, ncclSum, comm, stream);
  if (pred) { cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, C, M, &zero, A, lda, A, lda); }
}

inline void all_reduce_in_place(cudaStream_t stream, cublasHandle_t handle, int32_t pred, int32_t M, int32_t N, cuComplex* A, int32_t lda, cuComplex* C, ncclComm_t comm, void* timer) {
  Timer::register_comm(stream, timer); ncclAllReduce(C, C, int64_t(M) * int64_t(N) * int64_t(2), ncclFloat32, ncclSum, comm, stream);
  if (pred) { cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, C, M, &zero, A, lda, A, lda); }
}

inline void deq_nd_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const uint64_t* acc, int32_t bits, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t type, void* dev_work, ncclComm_t col_comm, void* timer) {
  int32_t pred = (ldg != N); void* data = pred ? dev_work : G;
  switch (type) {
    case HYACIN_F64:
      internal::int8::dequantize_i63_f64(stream, bits, order, M, N, acc, algnN, umax, vexp, (double*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (double*)G, ldg, (double*)data, col_comm, timer); return;
    case HYACIN_F32:
      internal::int8::dequantize_i63_f32(stream, bits, order, M, N, acc, algnN, umax, vexp, (float*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (float*)G, ldg, (float*)data, col_comm, timer); return;
    case HYACIN_F64_COMPLEX:
      internal::int8::dequantize_i63_cf64(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<double>*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (cuDoubleComplex*)G, ldg, (cuDoubleComplex*)data, col_comm, timer); return;
    case HYACIN_F32_COMPLEX:
      internal::int8::dequantize_i63_cf32(stream, bits, order, M, N, acc, algnN, umax, vexp, (std::complex<float>*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (cuComplex*)G, ldg, (cuComplex*)data, col_comm, timer); return;
    default: return;
  }
}

inline int32_t cublas_nd_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const void* A, int32_t lda, void* G, int32_t ldg, hyacinPrecision_t Gtype, ncclComm_t col_comm, void* timer) {
  double one_f64 = 1., zero_f64 = 0.; float one_f32 = 1.f, zero_f32 = 0.f;
  int32_t pred = (ldg != N); void* data = G;
  if (pred) {
    int32_t em_bytes; hyacinXelem('A', Gtype, nullptr, &em_bytes, nullptr);
    uint64_t work_bytes = uint64_t(N) * uint64_t(N) * uint64_t(em_bytes);
    if (cudaSuccess != cudaMallocAsync(&data, work_bytes, stream))
      throw std::runtime_error("Workspace allocation failed at 1-D row Float SY/HERK.");
  }

  switch (Gtype) {
    case HYACIN_F64:
      cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N, M, &one_f64, (const double*)A, lda, &zero_f64, (double*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (double*)G, ldg, (double*)data, col_comm, timer); break;
    case HYACIN_F32:
      cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N, M, &one_f32, (const float*)A, lda, &zero_f32, (float*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (float*)G, ldg, (float*)data, col_comm, timer); break;
    case HYACIN_F64_COMPLEX:
      cublasZherk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, N, M, &one_f64, (const cuDoubleComplex*)A, lda, &zero_f64, (cuDoubleComplex*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (cuDoubleComplex*)G, ldg, (cuDoubleComplex*)data, col_comm, timer); break;
    case HYACIN_F32_COMPLEX:
      cublasCherk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, N, M, &one_f32, (const cuComplex*)A, lda, &zero_f32, (cuComplex*)data, N);
      all_reduce_in_place(stream, handle, pred, N, N, (cuComplex*)G, ldg, (cuComplex*)data, col_comm, timer); break;
    default: break;
  }

  if (pred)
    cudaFreeAsync(data, stream);
  return 0;
}

extern "C" int32_t hyacinXsyherk1Drow(hyacinHandle_t handle, int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg, ncclComm_t col_comm) {
  if (globalM <= 0 || N <= 0) { return -1; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  if (alg == CUBLAS_FLOAT_ND) { return cublas_nd_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, A, lda, G, ldg, Atype, col_comm, handle.timer); }
  int32_t Complex, det_reduc, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, det_reduc, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(localM, globalM, N, umax, 1, Gtype, alg);

  void* dev_work = nullptr;
  if (cudaSuccess != cudaMallocAsync(&dev_work, uint64_t(i8_bytes + acc_bytes + vec_bytes), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at 1-D row Integer SY/HERK.");
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes], *vexp = &acc[acc_bytes];
  vexp_dispatcher(handle.cudaStream, localM, N, Atype, A, lda, (int32_t*)vexp);

  if (det_reduc) {
    int64_t stride = int64_t(algnN) * int64_t(N) + int64_t(algnN), acc_len = stride * (int64_t(orderC + 1) << Complex);
    Timer::register_comm(handle.cudaStream, handle.timer);
    ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMax, col_comm, handle.cudaStream);
    if (0 < localM) {
      Timer::register_kernel(handle.cudaStream, handle.timer);
      igemm_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
      internal::int8::accumulate_conv_i63_u47(handle.cudaStream, orderC, Complex, stride, (uint64_t*)acc);
    }
    else { cudaMemsetAsync(acc, 0, acc_bytes, handle.cudaStream); }
    Timer::register_comm(handle.cudaStream, handle.timer);
    ncclAllReduce(acc, acc, acc_len, ncclUint64, ncclSum, col_comm, handle.cudaStream);
    Timer::register_kernel(handle.cudaStream, handle.timer);
    deq_dispatcher(handle.cudaStream, globalM, N, algnN, umax, (uint64_t*)acc, 47, orderC, G, ldg, (const int32_t*)vexp, Gtype);
  }
  else {
    if (0 < localM)
      igemm_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
    deq_nd_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, algnN, umax, (uint64_t*)acc, 63, orderC, G, ldg, (const int32_t*)vexp, Gtype, iA, col_comm, handle.timer);
  }
  cudaFreeAsync(dev_work, handle.cudaStream);
  return umax;
}

#endif
