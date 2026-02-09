
#include <hyacin.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

const int32_t umax_threshold = 30; // umax < 30 : Limbs, 30 <= umax : CRT
const std::vector<int32_t> dd_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100

inline hyacinPrecision_t real_precision(hyacinPrecision_t prec) { return hyacinPrecision_t(int32_t(prec) & 7); }
inline hyacinPrecision_t complex_precision(hyacinPrecision_t prec) { return hyacinPrecision_t(int32_t(prec) | 8); }
inline int32_t pad_u_limbs(int32_t umax) { return ((umax + 10) & (~7)) - 3; }
inline int32_t pad_u_crt(int32_t umax, int32_t extra) { int32_t b = ((umax * 2) + extra) | 7; return (b - extra) / 2; }

extern "C" void hyacinXcpqrk_autoTune(double epi, int64_t globalM, int32_t u_extra, int32_t* umax, hyacinPrecision_t Atype, hyacinPrecision_t* ComputeType, hyacinAlgorithm_t* alg) {
  int32_t device, major, minor;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device = 100 * major + minor;

  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  double epi_qf = std::pow(double(std::numeric_limits<float>::epsilon()), 2);
  double epi_dd = std::numeric_limits<double>::epsilon();
  int32_t use_qf = int32_t(dd_sm_list.end() == std::find(dd_sm_list.begin(), dd_sm_list.end(), device));

  hyacinPrecision_t ATypeReal = real_precision(Atype);
  double machine_epi = ATypeReal == HYACIN_F32 ? double(std::numeric_limits<float>::epsilon()) : epi_dd;
  double epi_nrm = std::min(1., std::max(epi, machine_epi));
  hyacinPrecision_t auto_prec =
    epi_f32 <= epi_nrm ? HYACIN_F32 : (
    epi_f64 <= epi_nrm ? HYACIN_F64 : (
    (epi_qf <= epi_nrm && use_qf) ? HYACIN_QF : HYACIN_DD));

  int32_t u = u_extra + int32_t(std::ceil(-std::log2(epi_nrm)));
  int32_t Complex = int32_t(Atype != ATypeReal), use_limbs = int32_t(u < umax_threshold);
  int32_t b_extra = int32_t(std::ceil(std::log2(double(globalM)))) + 2 + Complex;

  *umax = use_limbs ? pad_u_limbs(u) : pad_u_crt(u, b_extra);
  *ComputeType = Complex ? complex_precision(auto_prec) : auto_prec;
  *alg = use_limbs ? HYACIN_ALG_LIMBS : HYACIN_ALG_CRT;
}
