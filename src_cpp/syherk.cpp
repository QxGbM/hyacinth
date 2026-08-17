
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <tuple>
#include <stdexcept>

inline std::tuple<int32_t, int32_t, int32_t, uint64_t, uint64_t> ext_params(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Gtype, hyacinAlgorithm_t alg) {
  hyacinPrecision_t GtypeReal = Gtype; hyacinXelem('R', &GtypeReal);
  int32_t Complex = int32_t(Gtype != GtypeReal);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS);
  int32_t bits = int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (Complex ? 2 : 0) + (use_limbs ? 0 : 2) + (umax << 1);
  int32_t orderA = (use_limbs ? (umax + 9) : (bits + 9)) >> 3;
  int32_t orderC = (use_limbs ? (bits + 62) : ((orderA << 3) + 63)) / 63;
  uint64_t acc_bytes = uint64_t(N) * uint64_t(N + (use_limbs ? 0 : 1)) * uint64_t(orderC) * uint64_t(Complex + 1) * sizeof(uint64_t);
  uint64_t tp_bytes = ((uint64_t(N) * uint64_t(N + 1)) / uint64_t(2)) * uint64_t(orderC) * uint64_t(Complex + 1) * sizeof(uint64_t);
  return std::tie(Complex, orderA, orderC, acc_bytes, tp_bytes);
}

inline void vexp_dispatcher(cudaStream_t stream, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t umax, int32_t* vexp) {
  switch(Atype) {
    case HYACIN_F64: internal::int8::vector_exponents(stream, M, N, (const double*)A, lda, umax, vexp); return;
    case HYACIN_F32: internal::int8::vector_exponents(stream, M, N, (const float*)A, lda, umax, vexp); return;
    case HYACIN_F16: internal::int8::vector_exponents(stream, M, N, (const __half*)A, lda, umax, vexp); return;
    case HYACIN_F64_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const cuDoubleComplex*)A, lda, umax, vexp); return;
    case HYACIN_F32_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const cuComplex*)A, lda, umax, vexp); return;
    case HYACIN_F16_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const __half2*)A, lda, umax, vexp); return;
    default: return;
  }
}

inline void igemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, 
  uint32_t umax, const int32_t* vexp, int32_t orderA, int32_t orderC, uint64_t* acc, hyacinAlgorithm_t alg) {
  if (alg == HYACIN_ALG_LIMBS) switch(Atype) {
    case HYACIN_F64: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const double*)A, lda, vexp, orderC, acc); return;
    case HYACIN_F32: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const float*)A, lda, vexp, orderC, acc); return;
    case HYACIN_F16: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const __half*)A, lda, vexp, orderC, acc); return;
    case HYACIN_F64_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const cuDoubleComplex*)A, lda, vexp, orderC, acc); return;
    case HYACIN_F32_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const cuComplex*)A, lda, vexp, orderC, acc); return;
    case HYACIN_F16_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const __half2*)A, lda, vexp, orderC, acc); return;
    default: return;
  }
  else if (alg == HYACIN_ALG_CRT) switch(Atype) {
    case HYACIN_F64: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const double*)A, lda, umax, vexp, orderC, acc); return;
    case HYACIN_F32: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const float*)A, lda, umax, vexp, orderC, acc); return;
    case HYACIN_F16: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const __half*)A, lda, umax, vexp, orderC, acc); return;
    case HYACIN_F64_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const cuDoubleComplex*)A, lda, umax, vexp, orderC, acc); return;
    case HYACIN_F32_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const cuComplex*)A, lda, umax, vexp, orderC, acc); return;
    case HYACIN_F16_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const __half2*)A, lda, umax, vexp, orderC, acc); return;
    default: return;
  }
}

inline void deq_dispatcher(cudaStream_t stream, int32_t N, const uint64_t* acc, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t Gtype) {
  switch (Gtype) {
    case HYACIN_F64: internal::int8::dequantize(stream, N, order, acc, vexp, (double*)G, ldg); return;
    case HYACIN_F32: internal::int8::dequantize(stream, N, order, acc, vexp, (float*)G, ldg); return;
    case HYACIN_DD: internal::int8::dequantize(stream, N, order, acc, vexp, (double2*)G, ldg); return;
    case HYACIN_QF: internal::int8::dequantize(stream, N, order, acc, vexp, (float4*)G, ldg); return;
    case HYACIN_F64_COMPLEX: internal::int8::dequantize_complex(stream, N, order, acc, vexp, (cuDoubleComplex*)G, ldg); return;
    case HYACIN_F32_COMPLEX: internal::int8::dequantize_complex(stream, N, order, acc, vexp, (cuComplex*)G, ldg); return;
    case HYACIN_DD_COMPLEX: internal::int8::dequantize_complex(stream, N, order, acc, vexp, (complex_double2*)G, ldg); return;
    case HYACIN_QF_COMPLEX: internal::int8::dequantize_complex(stream, N, order, acc, vexp, (complex_float4*)G, ldg); return;
    default: return;
  }
}

extern "C" void hyacinXsyherk(hyacinHandle_t handle, int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, orderA, orderC; uint64_t acc_bytes, tp_bytes;
  std::tie(Complex, orderA, orderC, acc_bytes, tp_bytes) = ext_params(M, N, umax, Gtype, alg);

  uint64_t* acc = nullptr, *tp = nullptr; int32_t* vexp = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&acc, acc_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&tp, tp_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  vexp_dispatcher(handle.cudaStream, M, N, Atype, A, lda, umax, vexp);
  igemm_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, Atype, A, lda, uint32_t(umax), vexp, orderA, orderC, acc, alg);
  internal::int8::triangle_pack(handle.cudaStream, Complex, alg == HYACIN_ALG_CRT ? M : 0, N, orderC, acc, uint32_t(umax), 0, orderC, tp);

  deq_dispatcher(handle.cudaStream, N, tp, orderC, G, ldg, vexp, Gtype);
  cudaFreeAsync(acc, handle.cudaStream);
  cudaFreeAsync(tp, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#ifndef NO_NCCL

extern "C" void hyacinXsyherk1Drow(hyacinHandle_t handle, int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg, ncclComm_t col_comm) {
  if (globalM <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, orderA, orderC; uint64_t acc_bytes, tp_bytes;
  std::tie(Complex, orderA, orderC, acc_bytes, tp_bytes) = ext_params(globalM, N, umax, Gtype, alg);

  uint64_t* acc = nullptr, *tp = nullptr; int32_t* vexp = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&acc, acc_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&tp, tp_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  vexp_dispatcher(handle.cudaStream, localM, N, Atype, A, lda, umax, vexp);

  Timer::register_comm(handle.cudaStream, handle.timer);
  ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMin, col_comm, handle.cudaStream);
  if (0 < localM) {
    Timer::register_kernel(handle.cudaStream, handle.timer);
    igemm_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, Atype, A, lda, uint32_t(umax), vexp, orderA, orderC, acc, alg);
    internal::int8::triangle_pack(handle.cudaStream, Complex, alg == HYACIN_ALG_CRT ? localM : 0, N, orderC, acc, uint32_t(umax), 0, orderC, tp);
  }
  else { cudaMemsetAsync(tp, 0, tp_bytes, handle.cudaStream); }

  hyacinXAllReduce1Drow(handle, orderC, Complex, (int64_t(N) * int64_t(N + 1)) / int64_t(2), tp, col_comm);
  Timer::register_kernel(handle.cudaStream, handle.timer);
  deq_dispatcher(handle.cudaStream, N, tp, orderC, G, ldg, vexp, Gtype);
  cudaFreeAsync(acc, handle.cudaStream);
  cudaFreeAsync(tp, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#endif
