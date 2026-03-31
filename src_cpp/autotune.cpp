
#include <hyacin.h>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

const int32_t umax_threshold = 30; // umax < 30 : Limbs, 30 <= umax : CRT
const int32_t umax_practical_limit = 80; // umax <= 80 to satisfy implementation assumptions
const std::vector<int32_t> f64_capable_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100

inline hyacinPrecision_t real_precision(hyacinPrecision_t prec) {
  switch(prec) {
    case HYACIN_F64: return HYACIN_F64; case HYACIN_F32: return HYACIN_F32;
    case HYACIN_DD: return HYACIN_DD; case HYACIN_QF: return HYACIN_QF;
    case HYACIN_F64_COMPLEX: return HYACIN_F64; case HYACIN_F32_COMPLEX: return HYACIN_F32;
    case HYACIN_DD_COMPLEX: return HYACIN_DD; case HYACIN_QF_COMPLEX: return HYACIN_QF;
    default: return hyacinPrecision_t(0);
  }
}

inline hyacinPrecision_t complex_precision(hyacinPrecision_t prec) {
  switch(prec) {
    case HYACIN_F64: return HYACIN_F64_COMPLEX; case HYACIN_F32: return HYACIN_F32_COMPLEX;
    case HYACIN_DD: return HYACIN_DD_COMPLEX; case HYACIN_QF: return HYACIN_QF_COMPLEX;
    case HYACIN_F64_COMPLEX: return HYACIN_F64_COMPLEX; case HYACIN_F32_COMPLEX: return HYACIN_F32_COMPLEX;
    case HYACIN_DD_COMPLEX: return HYACIN_DD_COMPLEX; case HYACIN_QF_COMPLEX: return HYACIN_QF_COMPLEX;
    default: return hyacinPrecision_t(0);
  }
}

extern "C" int32_t hyacinXelem_bytes(hyacinPrecision_t Atype) {
  switch(Atype) {
    case HYACIN_F64: return sizeof(double); case HYACIN_F32: return sizeof(float);
    case HYACIN_DD: return sizeof(double2); case HYACIN_QF: return sizeof(float4);
    case HYACIN_F64_COMPLEX: return sizeof(cuDoubleComplex); case HYACIN_F32_COMPLEX: return sizeof(cuComplex);
    case HYACIN_DD_COMPLEX: return sizeof(complex_double2); case HYACIN_QF_COMPLEX: return sizeof(complex_float4);
    default: return 0;
  }
}

extern "C" void hyacinXsyherk_autoTune(double epi, int32_t use_nd_allreduce, int32_t u_extra, int32_t* umax, hyacinPrecision_t Atype, hyacinPrecision_t* ComputeType, hyacinAlgorithm_t* alg) {
  int32_t device, major, minor;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device = 100 * major + minor;

  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  double epi_qf = std::pow(double(std::numeric_limits<float>::epsilon()), 2);
  double epi_dd = std::numeric_limits<double>::epsilon();
  int32_t f64_incapable = int32_t(f64_capable_sm_list.end() == std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), device));

  hyacinPrecision_t ATypeReal = real_precision(Atype);
  double machine_epi = (ATypeReal == HYACIN_F32) ? double(std::numeric_limits<float>::epsilon()) : epi_dd;
  double epi_nrm = std::min(1., std::max(epi, machine_epi));
  hyacinPrecision_t auto_prec =
    (epi_f32 <= epi_nrm && (f64_incapable || ATypeReal == HYACIN_F32)) ? HYACIN_F32 : (
    epi_f64 <= epi_nrm ? HYACIN_F64 : (
    (epi_qf <= epi_nrm && f64_incapable) ? HYACIN_QF : HYACIN_DD));

  int32_t u = u_extra + int32_t(std::ceil(-std::log2(epi_nrm)));
  int32_t Complex = int32_t(Atype != ATypeReal), use_limbs = int32_t(u < umax_threshold);
  use_nd_allreduce = use_nd_allreduce && (auto_prec == HYACIN_F32 || auto_prec == HYACIN_F64);

  *umax = std::min(umax_practical_limit, u);
  *ComputeType = Complex ? complex_precision(auto_prec) : auto_prec;
  *alg = use_nd_allreduce ? (use_limbs ? HYACIN_ALG_LIMBS_ND : HYACIN_ALG_CRT_ND) : (use_limbs ? HYACIN_ALG_LIMBS : HYACIN_ALG_CRT);
}
