
#include <hyacin.h>
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

template <int32_t blockSft, class Atype, class constAptr, class Btype, class Bptr> 
__global__ void cvcpy_kernel(int64_t M, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << blockSft) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M && y <= x)
    conv(A[y + x * lda], B[y + x * ldb]);
};

template <int32_t complex, class Btype, class Bptr>
inline void conv_copy_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const void* A, int64_t lda, hyacinPrecision_t precA, Bptr B, int64_t ldb) {
  constexpr int32_t block_threads = 512, blockSft = 9;
  dim3 grid((M + block_threads - 1) >> blockSft, N, 1);

  if constexpr(complex) switch (precA) {
    case HYACIN_F64_COMPLEX:
      cvcpy_kernel <blockSft, cuDoubleComplex, const cuDoubleComplex* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const cuDoubleComplex*)A, lda, B, ldb); break;
    case HYACIN_F32_COMPLEX:
      cvcpy_kernel <blockSft, cuComplex, const cuComplex* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const cuComplex*)A, lda, B, ldb); break;
    case HYACIN_DD_COMPLEX:
      cvcpy_kernel <blockSft, complex_double2, const complex_double2* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const complex_double2*)A, lda, B, ldb); break;
    case HYACIN_QF_COMPLEX:
      cvcpy_kernel <blockSft, complex_float4, const complex_float4* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const complex_float4*)A, lda, B, ldb); break;
    default: break;
  }
  else switch (precA) {
    case HYACIN_F64:
      cvcpy_kernel <blockSft, double, const double* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const double*)A, lda, B, ldb); break;
    case HYACIN_F32:
      cvcpy_kernel <blockSft, float, const float* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const float*)A, lda, B, ldb); break;
    case HYACIN_DD:
      cvcpy_kernel <blockSft, double2, const double2* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const double2*)A, lda, B, ldb); break;
    case HYACIN_QF:
      cvcpy_kernel <blockSft, float4, const float4* __restrict__, Btype, Bptr>
        <<< grid, block_threads, 0, stream >>> (int64_t(M), (const float4*)A, lda, B, ldb); break;
    default: break;
  }
}

inline hyacinPrecision_t real_precision(hyacinPrecision_t prec) { return hyacinPrecision_t(int32_t(prec) & 7); }
inline hyacinPrecision_t complex_precision(hyacinPrecision_t prec) { return hyacinPrecision_t(int32_t(prec) | 8); }
inline int32_t pad_u_limbs(int32_t umax) { return ((umax + 10) & (~7)) - 3; }
inline int32_t pad_u_crt(int32_t umax, int32_t extra) { int32_t b = ((umax * 2) + extra) | 7; return (b - extra) / 2; }

extern "C" void hyacinXcpqrk_autoTune(double epi, int32_t M, int32_t u_extra, int32_t* umax, hyacinPrecision_t Atype, hyacinPrecision_t* ComputeType, hyacinAlgorithm_t* alg) {
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
  int32_t b_extra = int32_t(std::ceil(std::log2(double(M)))) + 2 + Complex;

  *umax = use_limbs ? pad_u_limbs(u) : pad_u_crt(u, b_extra);
  *ComputeType = Complex ? complex_precision(auto_prec) : auto_prec;
  *alg = use_limbs ? HYACIN_ALG_LIMBS : HYACIN_ALG_CRT;
}

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t, int64_t, int64_t> i8gemm_ext_params(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t ComputeType, hyacinAlgorithm_t alg) {
  hyacinPrecision_t ComputeTypeReal = real_precision(ComputeType);
  int32_t Complex = int32_t(ComputeType != ComputeTypeReal);
  int32_t algnM = (M + 255) & (~255);
  int32_t algnN = (N + 63) & (~63);
  int32_t orderA = (alg == HYACIN_ALG_LIMBS) ? ((umax + 10) >> 3) : 8;
  int64_t elem_bytes = ComputeTypeReal == HYACIN_F32 ? sizeof(float) : (ComputeTypeReal == HYACIN_F64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = (int64_t(algnN) * int64_t(N) * elem_bytes) << Complex;
  int64_t i8_bytes = (int64_t(algnM) * int64_t(N) * int64_t(orderA)) << Complex;
  int64_t scratch_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderA) * sizeof(int32_t);
  scratch_bytes = std::max(scratch_bytes, C_bytes - i8_bytes);

  int32_t bits = int32_t(std::ceil(std::log2(double(M)))) + (umax << 1) + 2 + Complex;
  int32_t n_moduli = (bits + 8) >> 3;
  int32_t orderC = (((alg == HYACIN_ALG_LIMBS) ? bits : (n_moduli << 3)) + 63) / 63;
  int64_t acc_bytes = (int64_t(algnN) * int64_t(N) * int64_t(orderC) * sizeof(uint64_t)) << Complex;
  int64_t vec_bytes = int64_t(algnN) * int64_t(Complex ? 5 : 3) * sizeof(uint64_t);
  int64_t idx_bytes = elem_bytes << 9;
  return std::tie(algnM, algnN, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes, idx_bytes);
}

extern "C" void hyacinXcpqrk_bufferSize(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t ComputeType, hyacinAlgorithm_t alg, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  int32_t algnM, algnN, orderA, orderC, n_moduli; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes, idx_bytes;
  std::tie(algnM, algnN, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes, idx_bytes) = i8gemm_ext_params(M, N, umax, ComputeType, alg);
  *dev_work_bytes = uint64_t(i8_bytes + acc_bytes + scratch_bytes + vec_bytes);
  *pinned_work_bytes = uint64_t(int64_t(algnN) * sizeof(int32_t) + idx_bytes);
}

extern "C" int32_t hyacinXcpqrk(cublasHandle_t handle, char mode, double epi, int32_t M, int32_t N, int32_t K, int32_t p, int32_t umax,
  hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* jpiv, hyacinPrecision_t Rtype, void* R, int32_t ldr, hyacinPrecision_t ComputeType, void* dev_work, void* pinned_work, hyacinAlgorithm_t alg) {

  int32_t algnM, algnN, orderA, orderC, n_moduli; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes, idx_bytes;
  std::tie(algnM, algnN, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes, idx_bytes) = i8gemm_ext_params(M, N, umax, ComputeType, alg);

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes + scratch_bytes], *v_exp = &acc[acc_bytes];
  int32_t* hpiv = (int32_t*)(&((int8_t*)pinned_work)[idx_bytes]);
  std::iota(hpiv, &hpiv[N], 1);

  if (Atype == HYACIN_F64) {
    internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_f64(stream, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == HYACIN_ALG_LIMBS)
      internal::int8::i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, (uint64_t*)acc, algnN, iA);
  }
  else if (Atype == HYACIN_F32) {
    internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_f32(stream, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == HYACIN_ALG_LIMBS)
      internal::int8::i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, (uint64_t*)acc, algnN, iA);
  }
  else if (Atype == HYACIN_F64_COMPLEX) {
    internal::int8::vexp_cf64(stream, M, N, (const std::complex<double>*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_cf64(stream, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == HYACIN_ALG_LIMBS)
      internal::int8::i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, orderC, (uint64_t*)acc, algnN, iA);
  }
  else if (Atype == HYACIN_F32_COMPLEX) {
    internal::int8::vexp_cf32(stream, M, N, (const std::complex<float>*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_cf32(stream, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == HYACIN_ALG_LIMBS)
      internal::int8::i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      internal::int8::i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, orderC, (uint64_t*)acc, algnN, iA);
  }

  switch (ComputeType) {
    case HYACIN_F64:
      internal::int8::dequantize_i63_f64(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double*)iA, algnN);
      K = internal::Cholesky::potrfp_f64(stream, handle, epi, K, p, N, (double*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_F32:
      internal::int8::dequantize_i63_f32(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float*)iA, algnN);
      K = internal::Cholesky::potrfp_f32(stream, handle, epi, K, p, N, (float*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_DD:
      internal::int8::dequantize_i63_f128_dd(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double2*)iA, algnN);
      K = internal::Cholesky::potrfp_f128_dd(stream, handle, epi, K, p, N, (double2*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_QF:
      internal::int8::dequantize_i63_f128_qf(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float4*)iA, algnN);
      K = internal::Cholesky::potrfp_f128_qf(stream, handle, epi, K, p, N, (float4*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::dequantize_i63_cf64(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<double>*)iA, algnN);
      K = internal::Cholesky::potrfp_cf64(stream, handle, epi, K, p, N, (std::complex<double>*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::dequantize_i63_cf32(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<float>*)iA, algnN);
      K = internal::Cholesky::potrfp_cf32(stream, handle, epi, K, p, N, (std::complex<float>*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_DD_COMPLEX:
      internal::int8::dequantize_i63_cf128_dd(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_double2*)iA, algnN);
      K = internal::Cholesky::potrfp_cf128_dd(stream, handle, epi, K, p, N, (complex_double2*)iA, algnN, hpiv, pinned_work); break;
    case HYACIN_QF_COMPLEX:
      internal::int8::dequantize_i63_cf128_qf(stream, orderC, int64_t(M), N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_float4*)iA, algnN);
      K = internal::Cholesky::potrfp_cf128_qf(stream, handle, epi, K, p, N, (complex_float4*)iA, algnN, hpiv, pinned_work); break;
    default: break;
  }

  if ((mode == 'R' || mode == 'r') && (R != nullptr && K <= ldr)) switch (Rtype) {
    case HYACIN_F64:
      conv_copy_dispatcher<0, double, double* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (double*)R, int64_t(ldr)); break;
    case HYACIN_F32:
      conv_copy_dispatcher<0, float, float* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (float*)R, int64_t(ldr)); break;
    case HYACIN_DD:
      conv_copy_dispatcher<0, double2, double2* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (double2*)R, int64_t(ldr)); break;
    case HYACIN_QF:
      conv_copy_dispatcher<0, float4, float4* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (float4*)R, int64_t(ldr)); break;
    case HYACIN_F64_COMPLEX:
      conv_copy_dispatcher<1, cuDoubleComplex, cuDoubleComplex* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (cuDoubleComplex*)R, int64_t(ldr)); break;
    case HYACIN_F32_COMPLEX:
      conv_copy_dispatcher<1, cuComplex, cuComplex* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (cuComplex*)R, int64_t(ldr)); break;
    case HYACIN_DD_COMPLEX:
      conv_copy_dispatcher<1, complex_double2, complex_double2* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (complex_double2*)R, int64_t(ldr)); break;
    case HYACIN_QF_COMPLEX:
      conv_copy_dispatcher<1, complex_float4, complex_float4* __restrict__>(stream, K, N, iA, int64_t(algnN), ComputeType, (complex_float4*)R, int64_t(ldr)); break;
    default: break;
  }
  cudaMemcpyAsync(jpiv, hpiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  return K;
}
