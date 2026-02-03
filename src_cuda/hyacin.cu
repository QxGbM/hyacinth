
#include <hyacin.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <tuple>

const int32_t umax_threshold = 30; // umax < 30 : Limbs, 30 <= umax : CRT
const std::vector<int32_t> dd_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100

__device__ __forceinline__ void conv(double a, double& b) { b = a; }
__device__ __forceinline__ void conv(float a, double& b) { b = double(a); }
__device__ __forceinline__ void conv(double2 a, double& b) { b = device::dd::dd2double(a); }
__device__ __forceinline__ void conv(float4 a, double& b) { b = device::qf::qf2double(a); }

__device__ __forceinline__ void conv(double a, float& b) { b = float(a); }
__device__ __forceinline__ void conv(float a, float& b) { b = a; }
__device__ __forceinline__ void conv(double2 a, float& b) { b = float(a.x); }
__device__ __forceinline__ void conv(float4 a, float& b) { b = a.x; }

__device__ __forceinline__ void conv(double a, double2& b) { b = device::dd::double2dd(a); }
__device__ __forceinline__ void conv(float a, double2& b) { b = make_double2(double(a), 0.); }
__device__ __forceinline__ void conv(double2 a, double2& b) { b = a; }
__device__ __forceinline__ void conv(float4 a, double2& b) { b = device::dd::qf2dd(a); }

__device__ __forceinline__ void conv(double a, float4& b) { b = device::qf::double2qf(a); }
__device__ __forceinline__ void conv(float a, float4& b) { b = make_float4(a, 0.f, 0.f, 0.f); }
__device__ __forceinline__ void conv(double2 a, float4& b) { b = device::dd::dd2qf(a); }
__device__ __forceinline__ void conv(float4 a, float4& b) { b = a; }

__device__ __forceinline__ void conv(cuComplex a, cuDoubleComplex& b) { conv(a.x, b.x); conv(a.y, b.y); }
__device__ __forceinline__ void conv(complex_double2 a, cuDoubleComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }
__device__ __forceinline__ void conv(complex_float4 a, cuDoubleComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }

__device__ __forceinline__ void conv(cuDoubleComplex a, cuComplex& b) { conv(a.x, b.x); conv(a.y, b.y); }
__device__ __forceinline__ void conv(cuComplex a, cuComplex& b) { b = a; }
__device__ __forceinline__ void conv(complex_double2 a, cuComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }
__device__ __forceinline__ void conv(complex_float4 a,cuComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }

__device__ __forceinline__ void conv(cuDoubleComplex a, complex_double2& b) { conv(a.x, b.real); conv(a.y, b.real); }
__device__ __forceinline__ void conv(cuComplex a, complex_double2& b) { conv(a.x, b.real); conv(a.y, b.real); }
__device__ __forceinline__ void conv(complex_double2 a, complex_double2& b) { b = a; }
__device__ __forceinline__ void conv(complex_float4 a, complex_double2& b) { conv(a.real, b.real); conv(a.imag, b.real); }

__device__ __forceinline__ void conv(cuDoubleComplex a, complex_float4& b) { conv(a.x, b.real); conv(a.y, b.real); }
__device__ __forceinline__ void conv(cuComplex a, complex_float4& b) { conv(a.x, b.real); conv(a.y, b.real); }
__device__ __forceinline__ void conv(complex_double2 a, complex_float4& b) { conv(a.real, b.real); conv(a.imag, b.real); }
__device__ __forceinline__ void conv(complex_float4 a, complex_float4& b) { b = a; }

template <int32_t blockSft, class AType, class constAptr, class Btype, class Bptr> 
__global__ void cvcpy_kernel(int64_t M, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << blockSft) + int64_t(threadIdx.x);
  if (y < M) {
    int64_t x = int64_t(blockIdx.y);
    if (y <= x) conv(A[y + x * lda], B[y + x * ldb]);
  }
};

template <int32_t complex, class Btype, class Bptr>
inline void conv_copy_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const void* A, int64_t lda, device::Precision precA, Bptr B, int64_t ldb) {
  constexpr int32_t block_threads = 512, blockSft = 9;
  dim3 grid((M + block_threads - 1) >> blockSft, N, 1);

  if constexpr(complex) switch (precA) {
    case device::Precision::FP64_COMPLEX:
      cvcpy_kernel <blockSft, cuDoubleComplex, const cuDoubleComplex* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const cuDoubleComplex*)A, lda, B, ldb); break;
    case device::Precision::FP32_COMPLEX:
      cvcpy_kernel <blockSft, cuComplex, const cuComplex* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const cuComplex*)A, lda, B, ldb); break;
    case device::Precision::FP128_DD_COMPLEX:
      cvcpy_kernel <blockSft, complex_double2, const complex_double2* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const complex_double2*)A, lda, B, ldb); break;
    case device::Precision::FP128_QF_COMPLEX:
      cvcpy_kernel <blockSft, complex_float4, const complex_float4* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const complex_float4*)A, lda, B, ldb); break;
    default: break;
  }
  else switch (precA) {
    case device::Precision::FP64:
      cvcpy_kernel <blockSft, double, const double* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const double*)A, lda, B, ldb); break;
    case device::Precision::FP32:
      cvcpy_kernel <blockSft, float, const float* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const float*)A, lda, B, ldb); break;
    case device::Precision::FP128_DD:
      cvcpy_kernel <blockSft, double2, const double2* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const double2*)A, lda, B, ldb); break;
    case device::Precision::FP128_QF:
      cvcpy_kernel <blockSft, float4, const float4* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const float4*)A, lda, B, ldb); break;
    default: break;
  }
}

inline device::Precision real_precision(device::Precision prec) {
  switch (prec) {
    case device::Precision::FP64_COMPLEX: return device::Precision::FP64;
    case device::Precision::FP32_COMPLEX: return device::Precision::FP32;
    case device::Precision::FP128_DD_COMPLEX: return device::Precision::FP128_DD;
    case device::Precision::FP128_QF_COMPLEX: return device::Precision::FP128_QF;
    default: return prec;
  }
}

inline device::Precision complex_precision(device::Precision prec) {
  switch (prec) {
    case device::Precision::FP64: return device::Precision::FP64_COMPLEX;
    case device::Precision::FP32: return device::Precision::FP32_COMPLEX;
    case device::Precision::FP128_DD: return device::Precision::FP128_DD_COMPLEX;
    case device::Precision::FP128_QF: return device::Precision::FP128_QF_COMPLEX;
    default: return prec;
  }
}

inline int32_t pad_u_limbs(int32_t umax) {
  return ((umax + 10) & (~7)) - 3;
}

inline int32_t pad_u_crt(int32_t umax, int32_t extra) {
  int32_t b = ((umax * 2) + extra) | 7;
  return (b - extra) / 2;
}

void device::MixPrecAHA::igemm_params(double epi, int32_t M, int32_t u_extra, int32_t* umax, Precision AType, Precision* ComputeType, Algorithm* alg) {
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

  Precision ATypeReal = real_precision(AType);
  double machine_epi = ATypeReal == Precision::FP32 ? double(std::numeric_limits<float>::epsilon()) : epi_dd;
  double epi_nrm = std::min(1., std::max(epi, machine_epi));
  Precision auto_prec =
    epi_f32 <= epi_nrm ? Precision::FP32 : (
    epi_f64 <= epi_nrm ? Precision::FP64 : (
    (epi_qf <= epi_nrm && use_qf) ? Precision::FP128_QF : Precision::FP128_DD));

  int32_t u = u_extra + int32_t(std::ceil(-std::log2(epi_nrm)));
  int32_t Complex = int32_t(AType != ATypeReal), use_limbs = int32_t(u < umax_threshold);
  int32_t b_extra = int32_t(std::ceil(std::log2(double(M)))) + 2 + Complex;

  *umax = use_limbs ? pad_u_limbs(u) : pad_u_crt(u, b_extra);
  *ComputeType = Complex ? complex_precision(auto_prec) : auto_prec;
  *alg = use_limbs ? Algorithm::Limbs : Algorithm::CRT;
}

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t, int64_t> i8gemm_ext_params(int32_t M, int32_t N, int32_t umax, device::Precision ComputeType, device::Algorithm alg) {
  device::Precision ComputeTypeReal = real_precision(ComputeType);
  int32_t Complex = int32_t(ComputeType != ComputeTypeReal);
  int32_t algnM = (M + 255) & (~255);
  int32_t algnN = (N + 63) & (~63);
  int32_t orderA = (alg == device::Algorithm::Limbs) ? ((umax + 10) >> 3) : 8;
  int64_t elem_bytes = ComputeTypeReal == device::Precision::FP32 ? sizeof(float) : (ComputeTypeReal == device::Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = (int64_t(algnN) * int64_t(N) * elem_bytes) << Complex;
  int64_t i8_bytes = (int64_t(algnM) * int64_t(N) * int64_t(orderA)) << Complex;
  int64_t scratch_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderA) * sizeof(int32_t);
  scratch_bytes = std::max(scratch_bytes, C_bytes - i8_bytes);

  int32_t bits = int32_t(std::ceil(std::log2(double(M)))) + (umax << 1) + 2 + Complex;
  int32_t n_moduli = (bits + 8) >> 3;
  int32_t orderC = (((alg == device::Algorithm::Limbs) ? bits : (n_moduli << 3)) + 63) / 63;
  int64_t acc_bytes = (int64_t(algnN) * int64_t(N) * int64_t(orderC) * sizeof(uint64_t)) << Complex;
  int64_t vec_bytes = int64_t(algnN) * int64_t(Complex ? 5 : 3) * sizeof(uint64_t);
  return std::tie(algnM, algnN, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes);
}

void device::MixPrecAHA::igemm_workspace(int32_t M, int32_t N, int32_t umax, Precision ComputeType, Algorithm alg, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  int32_t algnM, algnN, orderA, orderC, n_moduli; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, algnN, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, umax, ComputeType, alg);
  *dev_work_bytes = i8_bytes + acc_bytes + scratch_bytes + vec_bytes;
  *pinned_work_bytes = uint64_t(8192);
}

int32_t device::MixPrecAHA::iAHA(cublasHandle_t handle, char mode, double epi, int32_t M, int32_t N, int32_t K, int32_t p, int32_t umax, Precision AType, const void* A, int32_t lda, int32_t* jpiv, Precision RType, void* R, int32_t ldr, Precision ComputeType, void* dev_work, void* pinned_work, Algorithm alg) {
  int32_t algnM, algnN, orderA, orderC, n_moduli; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, algnN, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, umax, ComputeType, alg);

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int8_t* iA = (int8_t*)(dev_work), *workspace = &iA[i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes];

  if (AType == Precision::FP64) {
    internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_f64(stream, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      internal::int8::i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, (uint64_t*)acc, algnN, iA);
  }
  else if (AType == Precision::FP32) {
    internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_f32(stream, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      internal::int8::i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, (uint64_t*)acc, algnN, iA);
  }
  else if (AType == Precision::FP64_COMPLEX) {
    internal::int8::vexp_f64(stream, 2 * M, N, (const double*)A, 2 * lda, (uint64_t*)v_exp);
    internal::int8::vsum_cf64(stream, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      internal::int8::i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, orderC, (uint64_t*)acc, algnN, iA);
  }
  else if (AType == Precision::FP32_COMPLEX) {
    internal::int8::vexp_f32(stream, 2 * M, N, (const float*)A, 2 * lda, (uint64_t*)v_exp);
    internal::int8::vsum_cf32(stream, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      internal::int8::i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, orderC, (uint64_t*)acc, algnN, iA);
  }

  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  switch (ComputeType) {
    case Precision::FP64:
      internal::int8::dequantize_f64(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double*)iA, algnN);
      K = internal::Cholesky::potrfp_f64(stream, handle, epi, K, p, N, (double*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP32:
      internal::int8::dequantize_f32(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float*)iA, algnN);
      K = internal::Cholesky::potrfp_f32(stream, handle, epi, K, p, N, (float*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP128_DD:
      internal::int8::dequantize_f128_dd(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double2*)iA, algnN);
      K = internal::Cholesky::potrfp_f128_dd(stream, handle, epi, K, p, N, (double2*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP128_QF:
      internal::int8::dequantize_f128_qf(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float4*)iA, algnN);
      K = internal::Cholesky::potrfp_f128_qf(stream, handle, epi, K, p, N, (float4*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP64_COMPLEX:
      internal::int8::dequantize_cf64(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<double>*)iA, algnN);
      K = internal::Cholesky::potrfp_cf64(stream, handle, epi, K, p, N, (std::complex<double>*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP32_COMPLEX:
      internal::int8::dequantize_cf32(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<float>*)iA, algnN);
      K = internal::Cholesky::potrfp_cf32(stream, handle, epi, K, p, N, (std::complex<float>*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP128_DD_COMPLEX:
      internal::int8::dequantize_cf128_dd(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_double2*)iA, algnN);
      K = internal::Cholesky::potrfp_cf128_dd(stream, handle, epi, K, p, N, (complex_double2*)iA, algnN, &hpiv[0], pinned_work); break;
    case Precision::FP128_QF_COMPLEX:
      internal::int8::dequantize_cf128_qf(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_float4*)iA, algnN);
      K = internal::Cholesky::potrfp_cf128_qf(stream, handle, epi, K, p, N, (complex_float4*)iA, algnN, &hpiv[0], pinned_work); break;
    default: break;
  }

  if (mode == 'R' || mode == 'r') switch (RType) {
    case device::Precision::FP64:
      conv_copy_dispatcher<0, double, double* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (double*)R, int64_t(ldr)); break;
    case device::Precision::FP32:
      conv_copy_dispatcher<0, float, float* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (float*)R, int64_t(ldr)); break;
    case device::Precision::FP128_DD:
      conv_copy_dispatcher<0, double2, double2* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (double2*)R, int64_t(ldr)); break;
    case device::Precision::FP128_QF:
      conv_copy_dispatcher<0, float4, float4* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (float4*)R, int64_t(ldr)); break;
    case device::Precision::FP64_COMPLEX:
      conv_copy_dispatcher<1, cuDoubleComplex, cuDoubleComplex* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (cuDoubleComplex*)R, int64_t(ldr)); break;
    case device::Precision::FP32_COMPLEX:
      conv_copy_dispatcher<1, cuComplex, cuComplex* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (cuComplex*)R, int64_t(ldr)); break;
    case device::Precision::FP128_DD_COMPLEX:
      conv_copy_dispatcher<1, complex_double2, complex_double2* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (complex_double2*)R, int64_t(ldr)); break;
    case device::Precision::FP128_QF_COMPLEX:
      conv_copy_dispatcher<1, complex_float4, complex_float4* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (complex_float4*)R, int64_t(ldr)); break;
    default: break;
  }
  cudaMemcpyAsync(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  return K;
}
