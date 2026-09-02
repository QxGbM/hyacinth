
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <tuple>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

int32_t device_sm = 0, device_f64_capable = 0;
const std::vector<int32_t> f64_capable_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100
int32_t internal::device_is_f64_capable() {
  if (device_sm == 0) {
    int32_t device, major, minor; cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    device_sm = 100 * major + minor;
    return device_f64_capable = int32_t(f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), device_sm));
  } else return device_f64_capable;
}

const int32_t u_practical_limit = 80; // u <= 80 to satisfy implementation assumptions
// mappings vector for datatypes
const std::vector<hyacinPrecision_t> real_type({ HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF, HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF });
const std::vector<hyacinPrecision_t> complex_type({ HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX, HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX });
const std::vector<int32_t> type_bytes({ sizeof(double), sizeof(float), sizeof(__half), sizeof(double2), sizeof(float4), sizeof(cuDoubleComplex), sizeof(cuComplex), sizeof(half2), sizeof(complex_double2), sizeof(complex_float4) });
const std::vector<int32_t> type_mantissa({ 52, 23, 10, 105, 95, 52, 23, 10, 105, 95 });

extern "C" int32_t hyacinXelem(char sel, hyacinPrecision_t* Atype) {
  int32_t type = int32_t(*Atype);
  *Atype = ((sel == 'R') || (sel == 'r')) ? real_type[type] : (((sel == 'C') || (sel == 'c')) ? complex_type[type] : hyacinPrecision_t(type));
  return type_bytes[int32_t(*Atype)];
}

extern "C" int32_t hyacinXquantizeScale(hyacinHandle_t handle, double epi, int32_t u_corr, int32_t globalM, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* vexp, int32_t* cPanels) {
  double epi_nrm = std::min(1., std::max(std::abs(epi), std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  int32_t u = std::min(u_practical_limit, u_corr + int32_t(std::ceil(-std::log2(epi_nrm))));
  if (cPanels) { *cPanels = (int32_t(std::ceil(std::log2(double(std::max(1, globalM))))) + ((Atype == real_type[int32_t(Atype)]) ? 62 : 63) + (u << 1)) / 63; }
  if (0 < u) switch(Atype) {
    case HYACIN_F64: internal::int8::vector_exponents(handle.cudaStream, M, N, (const double*)A, lda, &u, vexp); return u;
    case HYACIN_F32: internal::int8::vector_exponents(handle.cudaStream, M, N, (const float*)A, lda, &u, vexp); return u;
    case HYACIN_F16: internal::int8::vector_exponents(handle.cudaStream, M, N, (const __half*)A, lda, &u, vexp); return u;
    case HYACIN_F64_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, M, N, (const cuDoubleComplex*)A, lda, &u, vexp); return u;
    case HYACIN_F32_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, M, N, (const cuComplex*)A, lda, &u, vexp); return u;
    case HYACIN_F16_COMPLEX: internal::int8::vector_exponents(handle.cudaStream, M, N, (const __half2*)A, lda, &u, vexp); return u;
    default: return 0;
  } else return 0;
}

extern "C" hyacinPrecision_t hyacinXGautoType(int32_t g_corr, int32_t M, hyacinPrecision_t Atype, int32_t u, int32_t* gElemBytes) {
  hyacinPrecision_t AtypeReal = real_type[int32_t(Atype)];
  int32_t bits = (g_corr + 1) + int32_t(std::ceil(0.5 * std::log2(double(std::max(1, M))))) + (u << 1);
  hyacinPrecision_t GtypeReal = 
    (bits <= type_mantissa[int32_t(HYACIN_F32)] && (AtypeReal == HYACIN_F32 || AtypeReal == HYACIN_F16)) ? HYACIN_F32 : (
    bits <= type_mantissa[int32_t(HYACIN_F64)] ? HYACIN_F64 : (
    (bits <= type_mantissa[int32_t(HYACIN_QF)] && !internal::device_is_f64_capable()) ? HYACIN_QF : HYACIN_DD));
  hyacinPrecision_t Gtype = (Atype == AtypeReal) ? GtypeReal : complex_type[int32_t(GtypeReal)];
  if (gElemBytes) { *gElemBytes = type_bytes[Gtype]; }
  return Gtype;
}

extern "C" void hyacinXsyherk_autoTune(double epi, int32_t u_extra, int32_t* umax, hyacinPrecision_t Atype, hyacinPrecision_t* ComputeType) {
  int32_t f64_incapable = !internal::device_is_f64_capable();
  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  double epi_qf = std::pow(double(std::numeric_limits<float>::epsilon()), 2);

  hyacinPrecision_t ATypeReal = real_type[int32_t(Atype)];
  double epi_nrm = std::min(1., std::max(epi, std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  hyacinPrecision_t auto_prec =
    (epi_f32 <= epi_nrm && (ATypeReal == HYACIN_F32 || ATypeReal == HYACIN_F16)) ? HYACIN_F32 : (
    epi_f64 <= epi_nrm ? HYACIN_F64 : (
    (epi_qf <= epi_nrm && f64_incapable) ? HYACIN_QF : HYACIN_DD));

  int32_t u = std::min(u_practical_limit, u_extra + int32_t(std::ceil(-std::log2(epi_nrm))));
  if (umax)
    *umax = u;
  if (ComputeType)
    *ComputeType = (Atype != ATypeReal) ? complex_type[int32_t(auto_prec)] : auto_prec;
}


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
  hyacinXherk(handle, M, N, Atype, A, lda, 0, vexp, 0, orderC, C, alg);
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

  hyacinXherk(handle, localM, N, Atype, A, lda, 0, vexp, 0, orderC, C, alg);
  hyacinXAllReduce1Drow(handle, orderC, Complex, (int64_t(N) * int64_t(N + 1)) / int64_t(2), C, col_comm);
  hyacinXdequantize(handle, N, orderC, C, vexp, Gtype, G, ldg);

  cudaFreeAsync(C, handle.cudaStream);
  cudaFreeAsync(vexp, handle.cudaStream);
}

#endif
