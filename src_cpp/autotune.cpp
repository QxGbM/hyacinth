
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
const std::vector<hyacinPrecision_t> real_type({ HYACIN_F64, HYACIN_F32, HYACIN_DD, HYACIN_QF, HYACIN_F64, HYACIN_F32, HYACIN_DD, HYACIN_QF });
const std::vector<hyacinPrecision_t> complex_type({ HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX, HYACIN_F64_COMPLEX, HYACIN_F32_COMPLEX, HYACIN_DD_COMPLEX, HYACIN_QF_COMPLEX });
const std::vector<int32_t> type_bytes({ sizeof(double), sizeof(float), sizeof(double2), sizeof(float4), sizeof(cuDoubleComplex), sizeof(cuComplex), sizeof(complex_double2), sizeof(complex_float4) });
const std::vector<cudaDataType_t> cuda_type({ CUDA_R_64F, CUDA_R_32F, cudaDataType_t(), cudaDataType_t(), CUDA_C_64F, CUDA_C_32F, cudaDataType_t(), cudaDataType_t() });

extern "C" void hyacinXelem(char sel, hyacinPrecision_t Atype, hyacinPrecision_t* type, int32_t* bytes, cudaDataType_t* cutype) {
  hyacinPrecision_t prec = ((sel == 'R') || (sel == 'r')) ? real_type[int32_t(Atype)] : (((sel == 'C') || (sel == 'c')) ? complex_type[int32_t(Atype)] : Atype);
  if (type) { *type = prec; } if (bytes) { *bytes = type_bytes[int32_t(prec)]; } if (cutype) { *cutype = cuda_type[int32_t(prec)]; }
}

extern "C" void hyacinXsyherk_autoTune(double epi, int32_t use_nd_allreduce, int32_t u_extra, int32_t* umax, hyacinPrecision_t Atype, hyacinPrecision_t* ComputeType, hyacinAlgorithm_t* alg) {
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
  double epi_dd = std::numeric_limits<double>::epsilon();

  hyacinPrecision_t ATypeReal = real_type[int32_t(Atype)];
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
  *ComputeType = Complex ? complex_type[int32_t(auto_prec)] : auto_prec;
  *alg = use_nd_allreduce ? (use_limbs ? HYACIN_ALG_LIMBS_ND : HYACIN_ALG_CRT_ND) : (use_limbs ? HYACIN_ALG_LIMBS : HYACIN_ALG_CRT);
}

extern "C" void hyacinXGevPcsvd_autoTune(char* use_evd, int32_t N, int32_t K, hyacinPrecision_t Gtype) {
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
  *use_evd = pred ? 'Y' : 'N';
}
