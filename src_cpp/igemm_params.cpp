
#include <hyacinth.hpp>
#include <limits>

inline device::Precision device_igemm_behavior() {
  int32_t device, major, minor;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

  device = 100 * major + minor;
  return (device == 800 || device == 900 || device == 1000) ? device::Precision::FP128_DD : device::Precision::FP128_QF;
}

void device::MixPrecAHA::rATA_params_query(gemm_params* param, double epi, int32_t M, int32_t N, Precision precA) {
  double machine_epi = precA == Precision::FP32 ? 
    double(std::numeric_limits<float>::epsilon()) : std::numeric_limits<double>::epsilon();
  epi = -std::log2(std::min(1., std::max(epi, machine_epi)));

  param->M = M; param->N = N; param->precA = precA;
  param->algnM = (M + 127) & (~127); param->algnN = (N + 63) & (~63);
  param->orderA = std::max(1, 1 + int32_t(std::ceil(epi / Config::exp_base)));

  int32_t acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  Precision f128 = device_igemm_behavior();
  param->iter_k = 131072 << (14 - 2 * Config::exp_base);
  param->precC = acc_bits <= 24 ? Precision::FP32 : (acc_bits <= 53 ? Precision::FP64 : f128);
  param->C_elem_bytes = acc_bits <= 24 ? 4 : (acc_bits <= 53 ? 8 : 16);

  constexpr int64_t algn_i8 = int64_t(64) * sizeof(int32_t) - int64_t(1);
  int64_t strideA = int64_t(param->algnM) * int64_t(N);
  int64_t strideC = int64_t(param->algnN) * int64_t(N);

  param->acc_bytes = int64_t(param->C_elem_bytes) * strideC;
  param->i8_bytes = int64_t(param->orderA) * strideA;
  param->i8_bytes = (param->i8_bytes + algn_i8) & (~algn_i8);
  param->exp_bytes = sizeof(int32_t) * int64_t(param->algnN);
  param->scratch_bytes = sizeof(int32_t) * int64_t(param->orderA + 1) * strideC;

  int64_t A_elem_bytes = precA == Precision::FP32 ? sizeof(float) : sizeof(double);
  int64_t spare_space = std::max(param->i8_bytes + param->scratch_bytes, A_elem_bytes * strideC);
  param->C_bytes = param->acc_bytes + param->exp_bytes + spare_space;
}

void device::MixPrecAHA::cAHA_params_query(gemm_params* param, double epi, int32_t M, int32_t N, Precision precA) {
  double machine_epi = precA == Precision::FP32 ? 
    double(std::numeric_limits<float>::epsilon()) : std::numeric_limits<double>::epsilon();
  epi = -std::log2(std::min(1., std::max(epi, machine_epi)));

  param->M = M; param->N = N; param->precA = precA;
  param->algnM = (M + 127) & (~127); param->algnN = (N + 63) & (~63);
  param->orderA = std::max(1, 1 + int32_t(std::ceil(epi / Config::exp_base)));

  int32_t acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  Precision f128 = device_igemm_behavior();
  param->iter_k = 131072 << (14 - 2 * Config::exp_base);
  param->precC = acc_bits <= 24 ? Precision::FP32 : (acc_bits <= 53 ? Precision::FP64 : f128);
  param->C_elem_bytes = acc_bits <= 24 ? 8 : (acc_bits <= 53 ? 16 : 32);

  constexpr int64_t algn_i8 = int64_t(64) * sizeof(int32_t) - int64_t(1);
  int64_t strideA = int64_t(2) * int64_t(param->algnM) * int64_t(N);
  int64_t strideC = int64_t(param->algnN) * int64_t(N);

  param->acc_bytes = int64_t(param->C_elem_bytes) * strideC;
  param->i8_bytes = int64_t(param->orderA) * strideA;
  param->i8_bytes = (param->i8_bytes + algn_i8) & (~algn_i8);
  param->exp_bytes = sizeof(int32_t) * int64_t(param->algnN);
  param->scratch_bytes = sizeof(int32_t) * int64_t(param->orderA + 1) * strideC;

  int64_t spare_space = std::max(param->i8_bytes + param->scratch_bytes, param->acc_bytes);
  param->C_bytes = param->acc_bytes + param->exp_bytes + spare_space;
}

