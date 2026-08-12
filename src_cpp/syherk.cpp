
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <tuple>
#include <stdexcept>

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t> ext_params(int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Gtype, hyacinAlgorithm_t alg) {
  hyacinPrecision_t GtypeReal = Gtype; hyacinXelem('R', &GtypeReal);
  int32_t Complex = int32_t(Gtype != GtypeReal);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS);

  int32_t algnM = (localM + 255) & (~255);
  int32_t algnN = (N + 63) & (~63);
  int32_t bits = int32_t(std::ceil(std::log2(double(std::max(1, globalM))))) + (Complex ? 2 : 0) + (use_limbs ? 0 : 2) + (umax << 1);

  int32_t orderA = (use_limbs ? (umax + 9) : (bits + 9)) >> 3;
  int64_t i8_bytes = int64_t(N) * int64_t(use_limbs ? orderA : 8) * ((int64_t(algnM) * int64_t(Complex ? 3 : 1)) + (int64_t(algnN) * sizeof(int32_t)));

  int32_t orderC = (use_limbs ? (bits + 62) : ((orderA << 3) + 63)) / 63;
  int64_t acc_bytes = (int64_t(N) * int64_t(N + (use_limbs ? 0 : 1)) * int64_t(orderC << Complex) * sizeof(uint64_t) + int64_t(255)) & (~int64_t(255));
  int64_t vec_bytes = (int64_t(N) * sizeof(int32_t) + int64_t(255)) & (~int64_t(255));
  return std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes);
}

inline void vexp_dispatcher(cudaStream_t stream, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* vexp) {
  switch(Atype) {
    case HYACIN_F64: internal::int8::vector_exponents(stream, M, N, (const double*)A, lda, vexp); return;
    case HYACIN_F32: internal::int8::vector_exponents(stream, M, N, (const float*)A, lda, vexp); return;
    case HYACIN_F16: internal::int8::vector_exponents(stream, M, N, (const __half*)A, lda, vexp); return;
    case HYACIN_F64_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const cuDoubleComplex*)A, lda, vexp); return;
    case HYACIN_F32_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const cuComplex*)A, lda, vexp); return;
    case HYACIN_F16_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const __half2*)A, lda, vexp); return;
    default: return;
  }
}

inline void igemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, 
  int32_t umax, const int32_t* vexp, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* acc, hyacinAlgorithm_t alg) {
  if (alg == HYACIN_ALG_LIMBS) switch(Atype) {
    case HYACIN_F64: internal::int8::i63AHA_limbs(stream, handle, M, N, (const double*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F32: internal::int8::i63AHA_limbs(stream, handle, M, N, (const float*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F16: internal::int8::i63AHA_limbs(stream, handle, M, N, (const __half*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F64_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, (const cuDoubleComplex*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F32_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, (const cuComplex*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F16_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, (const __half2*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    default: return;
  }
  else if (alg == HYACIN_ALG_CRT) switch(Atype) {
    case HYACIN_F64: internal::int8::i63AHA_crt(stream, handle, M, N, (const double*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F32: internal::int8::i63AHA_crt(stream, handle, M, N, (const float*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F16: internal::int8::i63AHA_crt(stream, handle, M, N, (const __half*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F64_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, (const cuDoubleComplex*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F32_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, (const cuComplex*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    case HYACIN_F16_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, (const __half2*)A, lda, umax, vexp, algnM, algnN, orderA, orderC, acc); return;
    default: return;
  }
}

inline void deq_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t umax, const uint64_t* acc, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t Gtype) {
  switch (Gtype) {
    case HYACIN_F64: internal::int8::dequantize(stream, order, M, N, acc, umax, vexp, (double*)G, ldg); return;
    case HYACIN_F32: internal::int8::dequantize(stream, order, M, N, acc, umax, vexp, (float*)G, ldg); return;
    case HYACIN_DD: internal::int8::dequantize(stream, order, M, N, acc, umax, vexp, (double2*)G, ldg); return;
    case HYACIN_QF: internal::int8::dequantize(stream, order, M, N, acc, umax, vexp, (float4*)G, ldg); return;
    case HYACIN_F64_COMPLEX: internal::int8::dequantize_complex(stream, order, M, N, acc, umax, vexp, (cuDoubleComplex*)G, ldg); return;
    case HYACIN_F32_COMPLEX: internal::int8::dequantize_complex(stream, order, M, N, acc, umax, vexp, (cuComplex*)G, ldg); return;
    case HYACIN_DD_COMPLEX: internal::int8::dequantize_complex(stream, order, M, N, acc, umax, vexp, (complex_double2*)G, ldg); return;
    case HYACIN_QF_COMPLEX: internal::int8::dequantize_complex(stream, order, M, N, acc, umax, vexp, (complex_float4*)G, ldg); return;
    default: return;
  }
}

extern "C" void hyacinXsyherk(hyacinHandle_t handle, int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(M, M, N, umax, Gtype, alg);

  void* dev_work = nullptr;
  if (cudaSuccess != cudaMallocAsync(&dev_work, uint64_t(i8_bytes + acc_bytes + vec_bytes), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  int8_t* acc = (int8_t*)(dev_work), *vexp = &acc[acc_bytes + i8_bytes];
  vexp_dispatcher(handle.cudaStream, M, N, Atype, A, lda, (int32_t*)vexp);
  igemm_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, algnN, orderA, orderC, (uint64_t*)acc, alg);
  deq_dispatcher(handle.cudaStream, alg == HYACIN_ALG_CRT ? M : 0, N, umax, (uint64_t*)acc, orderC, G, ldg, (const int32_t*)vexp, Gtype);
  cudaFreeAsync(dev_work, handle.cudaStream);
}

#ifndef NO_NCCL

extern "C" void hyacinXsyherk1Drow(hyacinHandle_t handle, int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg, ncclComm_t col_comm) {
  if (globalM <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes;
  std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes) = ext_params(localM, globalM, N, umax, Gtype, alg);

  void* dev_work = nullptr;
  if (cudaSuccess != cudaMallocAsync(&dev_work, uint64_t(i8_bytes + acc_bytes + vec_bytes), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at 1-D row Integer SY/HERK.");
  int8_t* acc = (int8_t*)(dev_work), *vexp = &acc[acc_bytes + i8_bytes];
  vexp_dispatcher(handle.cudaStream, localM, N, Atype, A, lda, (int32_t*)vexp);

  int64_t stride = int64_t(N) * int64_t(N);
  Timer::register_comm(handle.cudaStream, handle.timer);
  ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMax, col_comm, handle.cudaStream);
  if (0 < localM) {
    Timer::register_kernel(handle.cudaStream, handle.timer);
    igemm_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, Atype, A, lda, umax, (const int32_t*)vexp, algnM, algnN, orderA, orderC, (uint64_t*)acc, alg);
  }
  else { cudaMemsetAsync(acc, 0, acc_bytes, handle.cudaStream); }
  hyacinXAllReduce1Drow(handle, orderC, Complex, stride, (uint64_t*)acc, col_comm);
  Timer::register_kernel(handle.cudaStream, handle.timer);
  deq_dispatcher(handle.cudaStream, alg == HYACIN_ALG_CRT ? globalM : 0, N, umax, (uint64_t*)acc, orderC, G, ldg, (const int32_t*)vexp, Gtype);
  cudaFreeAsync(dev_work, handle.cudaStream);
}

#endif
