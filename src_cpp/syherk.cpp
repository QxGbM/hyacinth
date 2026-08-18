
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <tuple>
#include <stdexcept>

inline std::tuple<int32_t, int32_t, int32_t, uint64_t> ext_params(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, hyacinAlgorithm_t& alg) {
  hyacinPrecision_t AtypeReal = Atype; hyacinXelem('R', &AtypeReal);
  int32_t Complex = int32_t(Atype != AtypeReal);
  int32_t bits = int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (Complex ? 2 : 0) + (umax << 1);
  int32_t orderA_limbs = int32_t(uint32_t(umax + 9) >> 3), orderA_crt = int32_t(uint32_t(bits + 11) >> 3);
  int32_t cost_limbs = int32_t(uint32_t(orderA_limbs * (orderA_limbs + 1)) >> 1), cost_crt = orderA_crt + int32_t(uint32_t(orderA_crt) >> 3);
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS || (alg == HYACIN_ALG_AUTO && (orderA_limbs <= 3 || cost_limbs <= cost_crt)));

  int32_t orderA = use_limbs ? orderA_limbs : orderA_crt;
  int32_t orderC = (use_limbs ? (bits + 62) : ((orderA_crt << 3) + 63)) / 63;
  uint64_t C_bytes = ((uint64_t(N) * uint64_t(N + 1)) / uint64_t(2)) * uint64_t(orderC) * uint64_t(Complex + 1) * sizeof(uint64_t);
  alg = use_limbs ? HYACIN_ALG_LIMBS : HYACIN_ALG_CRT;
  return std::tie(Complex, orderA, orderC, C_bytes);
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
  uint32_t umax, const int32_t* vexp, int32_t orderA, int32_t orderC, uint64_t* C, hyacinAlgorithm_t alg) {
  if (alg == HYACIN_ALG_LIMBS) switch(Atype) {
    case HYACIN_F64: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const double*)A, lda, vexp, 0, orderC, C); return;
    case HYACIN_F32: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const float*)A, lda, vexp, 0, orderC, C); return;
    case HYACIN_F16: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const __half*)A, lda, vexp, 0, orderC, C); return;
    case HYACIN_F64_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const cuDoubleComplex*)A, lda, vexp, 0, orderC, C); return;
    case HYACIN_F32_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const cuComplex*)A, lda, vexp, 0, orderC, C); return;
    case HYACIN_F16_COMPLEX: internal::int8::i63AHA_limbs(stream, handle, M, N, orderA, (const __half2*)A, lda, vexp, 0, orderC, C); return;
    default: return;
  }
  else if (alg == HYACIN_ALG_CRT) switch(Atype) {
    case HYACIN_F64: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const double*)A, lda, umax, vexp, 0, orderC, C); break;
    case HYACIN_F32: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const float*)A, lda, umax, vexp, 0, orderC, C); break;
    case HYACIN_F16: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const __half*)A, lda, umax, vexp, 0, orderC, C); break;
    case HYACIN_F64_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const cuDoubleComplex*)A, lda, umax, vexp, 0, orderC, C); break;
    case HYACIN_F32_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const cuComplex*)A, lda, umax, vexp, 0, orderC, C); break;
    case HYACIN_F16_COMPLEX: internal::int8::i63AHA_crt(stream, handle, M, N, orderA, (const __half2*)A, lda, umax, vexp, 0, orderC, C); break;
    default: break;
  }
}

inline void deq_dispatcher(cudaStream_t stream, int32_t N, const uint64_t* C, int32_t order, void* G, int32_t ldg, const int32_t* vexp, hyacinPrecision_t Gtype) {
  switch (Gtype) {
    case HYACIN_F64: internal::int8::dequantize(stream, N, order, C, vexp, (double*)G, ldg); return;
    case HYACIN_F32: internal::int8::dequantize(stream, N, order, C, vexp, (float*)G, ldg); return;
    case HYACIN_DD: internal::int8::dequantize(stream, N, order, C, vexp, (double2*)G, ldg); return;
    case HYACIN_QF: internal::int8::dequantize(stream, N, order, C, vexp, (float4*)G, ldg); return;
    case HYACIN_F64_COMPLEX: internal::int8::dequantize_complex(stream, N, order, C, vexp, (cuDoubleComplex*)G, ldg); return;
    case HYACIN_F32_COMPLEX: internal::int8::dequantize_complex(stream, N, order, C, vexp, (cuComplex*)G, ldg); return;
    case HYACIN_DD_COMPLEX: internal::int8::dequantize_complex(stream, N, order, C, vexp, (complex_double2*)G, ldg); return;
    case HYACIN_QF_COMPLEX: internal::int8::dequantize_complex(stream, N, order, C, vexp, (complex_float4*)G, ldg); return;
    default: return;
  }
}

extern "C" void hyacinXsyherk(hyacinHandle_t handle, int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, orderA, orderC; uint64_t C_bytes;
  std::tie(Complex, orderA, orderC, C_bytes) = ext_params(M, N, umax, Atype, alg);

  uint64_t* C = nullptr; int32_t* vexp = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&C, C_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  vexp_dispatcher(handle.cudaStream, M, N, Atype, A, lda, umax, vexp);
  igemm_dispatcher(handle.cudaStream, handle.cublasHandle, M, N, Atype, A, lda, uint32_t(umax), vexp, orderA, orderC, C, alg);

  deq_dispatcher(handle.cudaStream, N, C, orderC, G, ldg, vexp, Gtype);
  cudaFreeAsync(C, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#ifndef NO_NCCL

extern "C" void hyacinXsyherk1Drow(hyacinHandle_t handle, int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg, ncclComm_t col_comm) {
  if (globalM <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, orderA, orderC; uint64_t C_bytes;
  std::tie(Complex, orderA, orderC, C_bytes) = ext_params(globalM, N, umax, Atype, alg);

  uint64_t* C = nullptr; int32_t* vexp = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&C, C_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  vexp_dispatcher(handle.cudaStream, localM, N, Atype, A, lda, umax, vexp);

  Timer::register_comm(handle.cudaStream, handle.timer);
  ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMin, col_comm, handle.cudaStream);
  if (0 < localM) {
    Timer::register_kernel(handle.cudaStream, handle.timer);
    igemm_dispatcher(handle.cudaStream, handle.cublasHandle, localM, N, Atype, A, lda, uint32_t(umax), vexp, orderA, orderC, C, alg);
  }
  else { cudaMemsetAsync(C, 0, C_bytes, handle.cudaStream); }

  hyacinXAllReduce1Drow(handle, orderC, Complex, (int64_t(N) * int64_t(N + 1)) / int64_t(2), C, col_comm);
  Timer::register_kernel(handle.cudaStream, handle.timer);
  deq_dispatcher(handle.cudaStream, N, C, orderC, G, ldg, vexp, Gtype);
  cudaFreeAsync(C, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#endif
