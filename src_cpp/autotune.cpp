
#include <hyacin.h>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

int32_t device_sm = 0;
const int32_t umax_threshold = 30; // umax < 30 : Limbs, 30 <= umax : CRT
const int32_t umax_practical_limit = 80; // umax <= 80 to satisfy implementation assumptions
const std::vector<int32_t> f64_capable_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100

// mappings vector for datatypes
const std::vector<hyacinPrecision_t> real_type({ HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF, HYACIN_F64, HYACIN_F32, HYACIN_F16, HYACIN_DD, HYACIN_QF });
const std::vector<hyacinPrecision_t> complex_type({ HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX, HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_F16_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX });
const std::vector<int32_t> type_bytes({ sizeof(double), sizeof(float), sizeof(__half), sizeof(double2), sizeof(float4), sizeof(cuDoubleComplex), sizeof(cuComplex), sizeof(half2), sizeof(complex_double2), sizeof(complex_float4) });
const std::vector<int32_t> type_mantissa({ 52, 23, 10, 0, 0, 52, 23, 10, 0, 0 });

extern "C" int32_t hyacinXelem(char sel, hyacinPrecision_t* Atype) {
  int32_t type = int32_t(*Atype);
  *Atype = ((sel == 'R') || (sel == 'r')) ? real_type[type] : (((sel == 'C') || (sel == 'c')) ? complex_type[type] : hyacinPrecision_t(type));
  return type_bytes[int32_t(*Atype)];
}

extern "C" void hyacinXsyherk_autoTune(double epi, int32_t u_extra, int32_t* umax, hyacinPrecision_t Atype, hyacinPrecision_t* ComputeType, hyacinAlgorithm_t* alg) {
  if (device_sm == 0) {
    int32_t device, major, minor;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    device_sm = 100 * major + minor;
  }

  int32_t f64_incapable = int32_t(f64_capable_sm_list.end() == std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), device_sm));
  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  double epi_qf = std::pow(double(std::numeric_limits<float>::epsilon()), 2);

  hyacinPrecision_t ATypeReal = real_type[int32_t(Atype)];
  double epi_nrm = std::min(1., std::max(epi, std::ldexp(1., -type_mantissa[int32_t(Atype)])));
  hyacinPrecision_t auto_prec =
    (epi_f32 <= epi_nrm && (ATypeReal == HYACIN_F32 || ATypeReal == HYACIN_F16)) ? HYACIN_F32 : (
    epi_f64 <= epi_nrm ? HYACIN_F64 : (
    (epi_qf <= epi_nrm && f64_incapable) ? HYACIN_QF : HYACIN_DD));

  int32_t u = std::min(umax_practical_limit, u_extra + int32_t(std::ceil(-std::log2(epi_nrm))));
  if (umax)
    *umax = u;
  if (ComputeType)
    *ComputeType = (Atype != ATypeReal) ? complex_type[int32_t(auto_prec)] : auto_prec;
  if (alg)
    *alg = u < umax_threshold ? HYACIN_ALG_LIMBS : HYACIN_ALG_CRT;
}

extern "C" char hyacinXGevPcsvd_autoTune(int32_t N, int32_t K, hyacinPrecision_t Gtype) {
  if (device_sm == 0) {
    int32_t device, major, minor;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    device_sm = 100 * major + minor;
  }

  int32_t f64_capable = int32_t(f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), device_sm));
  hyacinPrecision_t GTypeReal = real_type[int32_t(Gtype)];
  int32_t pred = ((GTypeReal == HYACIN_F32) || ((GTypeReal == HYACIN_F64) && f64_capable)) && (N <= (K << 1));
  return pred ? 'Y' : 'N';
}
