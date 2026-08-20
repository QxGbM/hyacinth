
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <tuple>
#include <stdexcept>

inline std::tuple<int32_t, int32_t, uint64_t> ext_params(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype) {
  hyacinPrecision_t AtypeReal = Atype; hyacinXelem('R', &AtypeReal);
  int32_t Complex = int32_t(Atype != AtypeReal);
  int32_t orderC = (int32_t(std::ceil(std::log2(double(std::max(1, M))))) + (Complex ? 63 : 62) + (umax << 1)) / 63;
  uint64_t C_bytes = uint64_t(N) * uint64_t(N + 1) * uint64_t(orderC) * uint64_t(Complex + 1) * sizeof(uint32_t);
  return std::tie(Complex, orderC, C_bytes);
}

inline void vexp_dispatcher(cudaStream_t stream, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t umax, int32_t* vexp) {
  switch(Atype) {
    case HYACIN_F64: internal::int8::vector_exponents(stream, M, N, (const double*)A, lda, &umax, vexp); return;
    case HYACIN_F32: internal::int8::vector_exponents(stream, M, N, (const float*)A, lda, &umax, vexp); return;
    case HYACIN_F16: internal::int8::vector_exponents(stream, M, N, (const __half*)A, lda, &umax, vexp); return;
    case HYACIN_F64_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const cuDoubleComplex*)A, lda, &umax, vexp); return;
    case HYACIN_F32_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const cuComplex*)A, lda, &umax, vexp); return;
    case HYACIN_F16_COMPLEX: internal::int8::vector_exponents(stream, M, N, (const __half2*)A, lda, &umax, vexp); return;
    default: return;
  }
}

extern "C" void hyacinXsyherk(hyacinHandle_t handle, int32_t M, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg) {
  if (M <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, orderC; uint64_t C_bytes;
  std::tie(Complex, orderC, C_bytes) = ext_params(M, N, umax, Atype);

  uint64_t* C = nullptr; int32_t* vexp = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&C, C_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");

  vexp_dispatcher(handle.cudaStream, M, N, Atype, A, lda, umax, vexp);
  hyacinXherk(handle, M, N, Atype, A, lda, vexp, 0, orderC, C, alg);
  hyacinXdequantize(handle, N, orderC, C, vexp, Gtype, G, ldg);
  cudaFreeAsync(C, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#ifndef NO_NCCL

extern "C" void hyacinXsyherk1Drow(hyacinHandle_t handle, int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t Atype, const void* A, int32_t lda, hyacinPrecision_t Gtype, void* G, int32_t ldg, hyacinAlgorithm_t alg, ncclComm_t col_comm) {
  if (globalM <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);

  int32_t Complex, orderC; uint64_t C_bytes;
  std::tie(Complex, orderC, C_bytes) = ext_params(globalM, N, umax, Atype);

  uint64_t* C = nullptr; int32_t* vexp = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&C, C_bytes, handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at Integer SY/HERK.");

  vexp_dispatcher(handle.cudaStream, localM, N, Atype, A, lda, umax, vexp);
  Timer::register_comm(handle.cudaStream, handle.timer);
  ncclAllReduce(vexp, vexp, int64_t(N), ncclInt32, ncclMin, col_comm, handle.cudaStream);

  hyacinXherk(handle, localM, N, Atype, A, lda, vexp, 0, orderC, C, alg);
  hyacinXAllReduce1Drow(handle, orderC, Complex, (int64_t(N) * int64_t(N + 1)) / int64_t(2), C, col_comm);
  hyacinXdequantize(handle, N, orderC, C, vexp, Gtype, G, ldg);

  cudaFreeAsync(C, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#endif
