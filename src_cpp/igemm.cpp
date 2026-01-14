
#include <hyacin.hpp>
#include <internal.hpp>
#include <crt_selector.hpp>
#include <limits>

void device::MixPrecAHA::igemm_params(double* epi, int32_t N, int32_t* algnN, int32_t* umax, Precision precA, Precision* precC) {
  int32_t device, major, minor;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device = 100 * major + minor;

  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  std::pair<Precision, double> f128 = (device == 800 || device == 900 || device == 1000) ? 
    std::make_pair(device::Precision::FP128_DD, std::numeric_limits<double>::epsilon()) :
    std::make_pair(device::Precision::FP128_QF, std::pow(double(std::numeric_limits<float>::epsilon()), 2));

  double machine_epi = precA == Precision::FP32 ? double(std::numeric_limits<float>::epsilon()) : f128.second;
  *algnN = (N + 63) & (~63);
  *epi = std::min(1., std::max(std::abs(*epi), machine_epi));
  *precC = epi_f32 <= *epi ? Precision::FP32 : (epi_f64 <= *epi ? Precision::FP64 : f128.first);
  *umax = *umax + int32_t(std::ceil(-std::log2(*epi)));
}

inline std::tuple<int32_t, int32_t, int64_t, int64_t, int64_t, int64_t> i8gemm_ext_params(int32_t M, int32_t N, int32_t algnN, int32_t umax, int32_t Complex, device::Precision prec) {
  int32_t algnM = (M + 255) & (~255);
  int32_t orderA = umax <= 46 ? ((umax + 9) >> 3) : 8;
  int64_t elem_bytes = prec == device::Precision::FP32 ? sizeof(float) : (prec == device::Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = (int64_t(algnN) * int64_t(N) * elem_bytes) << Complex;
  int64_t i8_bytes = (int64_t(algnM) * int64_t(N) * int64_t(orderA)) << Complex;
  int64_t scratch_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderA) * sizeof(int32_t);
  scratch_bytes = std::max(scratch_bytes, C_bytes - i8_bytes);

  int32_t bits = int32_t(std::ceil(std::log2(double(algnM)))) + (umax << 1) + 2 + Complex;
  int32_t orderC = 1 + ((8 + (bits & (~7))) / 63);
  int64_t acc_bytes = (int64_t(algnN) * int64_t(N) * int64_t(orderC) * sizeof(uint64_t)) << Complex;
  int64_t vec_bytes = int64_t(algnN) * int64_t(Complex ? 5 : 3) * sizeof(uint64_t);
  return std::tie(orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes);
}

void device::MixPrecAHA::igemm_limbed_workspace(int32_t M, int32_t N, int32_t algnN, int32_t umax, int32_t Complex, Precision precC, int64_t* workspace) {
  int32_t orderA, orderC; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, Complex, precC);
  *workspace = i8_bytes + acc_bytes + scratch_bytes + vec_bytes;
}

void device::MixPrecAHA::rATA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC) {
  int32_t orderA, orderC; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, 0, precC);

  int8_t* iA = (int8_t*)(C), *workspace = &iA[i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes];

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp);
    internal::int8::vsum_f64(stream, M, N, (const double*)A, lda, (uint64_t*)v_exp, algnN);
    if (orderA < 8)
      internal::int8::i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp);
    internal::int8::vsum_f32(stream, M, N, (const float*)A, lda, (uint64_t*)v_exp, algnN);
    if (orderA < 8)
      internal::int8::i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
  }

  switch (precC) {
    case Precision::FP64:
      internal::int8::dequantize_f64(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double*)iA, algnN); break;
    case Precision::FP32:
      internal::int8::dequantize_f32(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float*)iA, algnN); break;
    case Precision::FP128_DD:
      internal::int8::dequantize_f128_dd(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double2*)iA, algnN); break;
    case Precision::FP128_QF:
      internal::int8::dequantize_f128_qf(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float4*)iA, algnN); break;
    default: break;
  }
}

void device::MixPrecAHA::cAHA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC) {
  int32_t orderA, orderC; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, 1, precC);

  int8_t* iA = (int8_t*)(C), *workspace = &iA[i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes];

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, 2 * M, N, (const double*)A, 2 * lda, umax, (uint64_t*)v_exp);
    internal::int8::vsum_cf64(stream, M, N, (const std::complex<double>*)A, lda, (uint64_t*)v_exp, algnN);
    if (orderA < 8)
      internal::int8::i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, 2 * M, N, (const float*)A, 2 * lda, umax, (uint64_t*)v_exp);
    internal::int8::vsum_cf32(stream, M, N, (const std::complex<float>*)A, lda, (uint64_t*)v_exp, algnN);
    if (orderA < 8)
      internal::int8::i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, (uint64_t*)acc, algnN, iA);
  }

  switch (precC) {
    case Precision::FP64:
      internal::int8::dequantize_cf64(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<double>*)iA, algnN); break;
    case Precision::FP32:
      internal::int8::dequantize_cf32(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<float>*)iA, algnN); break;
    case Precision::FP128_DD:
      internal::int8::dequantize_cf128_dd(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_double2*)iA, algnN); break;
    case Precision::FP128_QF:
      internal::int8::dequantize_cf128_qf(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_float4*)iA, algnN); break;
    default: break;
  }
}
