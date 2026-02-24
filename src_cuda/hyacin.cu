
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <numeric>
#include <tuple>

__device__ __forceinline__ void conv(float a, double& b) { b = double(a); }
__device__ __forceinline__ void conv(double2 a, double& b) { b = device::dd::dd2double(a); }
__device__ __forceinline__ void conv(float4 a, double& b) { b = device::qf::qf2double(a); }

__device__ __forceinline__ void conv(double a, float& b) { b = float(a); }
__device__ __forceinline__ void conv(double2 a, float& b) { b = float(a.x); }
__device__ __forceinline__ void conv(float4 a, float& b) { b = a.x; }

__device__ __forceinline__ void conv(cuComplex a, cuDoubleComplex& b) { conv(a.x, b.x); conv(a.y, b.y); }
__device__ __forceinline__ void conv(complex_double2 a, cuDoubleComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }
__device__ __forceinline__ void conv(complex_float4 a, cuDoubleComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }

__device__ __forceinline__ void conv(cuDoubleComplex a, cuComplex& b) { conv(a.x, b.x); conv(a.y, b.y); }
__device__ __forceinline__ void conv(complex_double2 a, cuComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }
__device__ __forceinline__ void conv(complex_float4 a, cuComplex& b) { conv(a.real, b.x); conv(a.imag, b.y); }

template <int32_t mode, class BType, class constAptr, class Bptr> 
__global__ void cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) {
    A = &A[y + x * lda]; if constexpr(mode & 1) B = &B[y + int64_t(jpiv[x] - 1) * ldb]; else B = &B[y + x * ldb];
    if (y <= x) { if constexpr(mode & 2) conv(*A, *B); else *B = *A; }
      else *B = BType();
  }
};

template <int32_t mode>
inline void conv_copy_dispatcher(cudaStream_t stream, int64_t M, int32_t N, const int32_t* jpiv, const void* A, int64_t lda, hyacinPrecision_t precA, void* B, int64_t ldb, hyacinPrecision_t precB) {
  constexpr int32_t block_threads = 512, mode_c = mode | 2;
  dim3 grid(uint32_t(M + block_threads - 1) >> 9, uint32_t(N), 1);

  if (precB == HYACIN_F64) switch (precA) {
    case HYACIN_F64:
      cvcpy_kernel <mode, double, const double* __restrict__, double* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const double*)A, lda, (double*)B, ldb); break;
    case HYACIN_F32:
      cvcpy_kernel <mode_c, double, const float* __restrict__, double* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const float*)A, lda, (double*)B, ldb); break;
    case HYACIN_DD:
      cvcpy_kernel <mode_c, double, const double2* __restrict__, double* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const double2*)A, lda, (double*)B, ldb); break;
    case HYACIN_QF:
      cvcpy_kernel <mode_c, double, const float4* __restrict__, double* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const float4*)A, lda, (double*)B, ldb); break;
    default: break;
  }
  else if (precB == HYACIN_F32) switch (precA) {
    case HYACIN_F64:
      cvcpy_kernel <mode_c, float, const double* __restrict__, float* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const double*)A, lda, (float*)B, ldb); break;
    case HYACIN_F32:
      cvcpy_kernel <mode, float, const float* __restrict__, float* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const float*)A, lda, (float*)B, ldb); break;
    case HYACIN_DD:
      cvcpy_kernel <mode_c, float, const double2* __restrict__, float* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const double2*)A, lda, (float*)B, ldb); break;
    case HYACIN_QF:
      cvcpy_kernel <mode_c, float, const float4* __restrict__, float* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const float4*)A, lda, (float*)B, ldb); break;
    default: break;
  }
  else if (precB == HYACIN_F64_COMPLEX) switch (precA) {
    case HYACIN_F64_COMPLEX:
      cvcpy_kernel <mode, cuDoubleComplex, const cuDoubleComplex* __restrict__, cuDoubleComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, ldb); break;
    case HYACIN_F32_COMPLEX:
      cvcpy_kernel <mode_c, cuDoubleComplex, const cuComplex* __restrict__, cuDoubleComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const cuComplex*)A, lda, (cuDoubleComplex*)B, ldb); break;
    case HYACIN_DD_COMPLEX:
      cvcpy_kernel <mode_c, cuDoubleComplex, const complex_double2* __restrict__, cuDoubleComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const complex_double2*)A, lda, (cuDoubleComplex*)B, ldb); break;
    case HYACIN_QF_COMPLEX:
      cvcpy_kernel <mode_c, cuDoubleComplex, const complex_float4* __restrict__, cuDoubleComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const complex_float4*)A, lda, (cuDoubleComplex*)B, ldb); break;
    default: break;
  }
  else if (precB == HYACIN_F32_COMPLEX) switch (precA) {
    case HYACIN_F64_COMPLEX:
      cvcpy_kernel <mode_c, cuComplex, const cuDoubleComplex* __restrict__, cuComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const cuDoubleComplex*)A, lda, (cuComplex*)B, ldb); break;
    case HYACIN_F32_COMPLEX:
      cvcpy_kernel <mode, cuComplex, const cuComplex* __restrict__, cuComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const cuComplex*)A, lda, (cuComplex*)B, ldb); break;
    case HYACIN_DD_COMPLEX:
      cvcpy_kernel <mode_c, cuComplex, const complex_double2* __restrict__, cuComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const complex_double2*)A, lda, (cuComplex*)B, ldb); break;
    case HYACIN_QF_COMPLEX:
      cvcpy_kernel <mode_c, cuComplex, const complex_float4* __restrict__, cuComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (M, jpiv, (const complex_float4*)A, lda, (cuComplex*)B, ldb); break;
    default: break;
  }
}

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t, int64_t> ext_params(int32_t localM, int32_t globalM, int32_t N, int32_t umax, int32_t acc_bits, hyacinPrecision_t ComputeType, hyacinAlgorithm_t alg) {
  hyacinPrecision_t ComputeTypeReal = hyacinPrecision_t(int32_t(ComputeType) & 7);
  int32_t Complex = int32_t(ComputeType != ComputeTypeReal);
  int32_t algnM = (localM + 255) & (~255);
  int32_t algnN = (N + 63) & (~63);
  int32_t bits = int32_t(std::ceil(std::log2(double(globalM)))) + (umax << 1) + 2 + Complex;
  int32_t use_limbs = int32_t(alg == HYACIN_ALG_LIMBS);

  int32_t orderA = ((use_limbs ? umax : bits) + 8) >> 3;
  int64_t elem_bytes = ComputeTypeReal == HYACIN_F32 ? sizeof(float) : (ComputeTypeReal == HYACIN_F64 ? sizeof(double) : sizeof(double2));
  int64_t i8_bytes = int64_t(N) * int64_t(use_limbs ? orderA : 8) * ((int64_t(algnM) << Complex) + (int64_t(algnN) * sizeof(int32_t)));
  int64_t C_bytes = (int64_t(algnN) * int64_t(N) * elem_bytes) << Complex;
  i8_bytes = std::max(i8_bytes, C_bytes);

  int32_t orderC = ((use_limbs ? bits : (orderA << 3)) + 63) / 63;
  int32_t orderD = (orderC + int32_t(acc_bits < 63)) << Complex;
  int64_t acc_bytes = int64_t(algnN) * int64_t(N + 1) * int64_t(orderD) * sizeof(uint64_t);
  int64_t vec_bytes = int64_t(algnN) * sizeof(int32_t);
  int64_t idx_bytes = elem_bytes << 9;
  return std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes, idx_bytes);
}

inline void vexp_dispatcher(cudaStream_t stream, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* jpiv) {
  switch(Atype) {
    case HYACIN_F64:
      internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, jpiv); break;
    case HYACIN_F32:
      internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, jpiv); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::vexp_cf64(stream, M, N, (const std::complex<double>*)A, lda, jpiv); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::vexp_cf32(stream, M, N, (const std::complex<float>*)A, lda, jpiv); break;
    default: break;
  }
}

inline void igemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, 
  int32_t umax, const int32_t* jpiv, int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* acc, int32_t algnN, int8_t* iA, hyacinAlgorithm_t alg) {
  if (alg == HYACIN_ALG_LIMBS) switch(Atype) {
    case HYACIN_F64:
      internal::int8::i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32:
      internal::int8::i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    default: break;
  }
  else if (alg == HYACIN_ALG_CRT) switch(Atype) {
    case HYACIN_F64:
      internal::int8::i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32:
      internal::int8::i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, jpiv, algnM, orderA, orderC, acc, algnN, iA); break;
    default: break;
  }
}

inline int32_t rrf_dispatcher(cudaStream_t stream, cublasHandle_t handle, char mode, double epi, int32_t M, int32_t N, int32_t algnN, int32_t K, int32_t p, int32_t umax, 
  const uint64_t* acc, int32_t bits, int32_t order, void* A, int32_t* jpiv, int32_t* hpiv, hyacinPrecision_t Rtype, void* R, int32_t ldr, hyacinPrecision_t ComputeType, void* pinned_work) {

  std::iota(hpiv, &hpiv[N], 1);
  switch (ComputeType) {
    case HYACIN_F64:
      internal::int8::dequantize_i63_f64(stream, bits, order, M, N, acc, algnN, umax, jpiv, (double*)A, algnN);
      K = internal::Cholesky::potrfp_f64(stream, handle, epi, K, p, N, (double*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_F32:
      internal::int8::dequantize_i63_f32(stream, bits, order, M, N, acc, algnN, umax, jpiv, (float*)A, algnN);
      K = internal::Cholesky::potrfp_f32(stream, handle, epi, K, p, N, (float*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_DD:
      internal::int8::dequantize_i63_f128_dd(stream, bits, order, M, N, acc, algnN, umax, jpiv, (double2*)A, algnN);
      K = internal::Cholesky::potrfp_f128_dd(stream, handle, epi, K, p, N, (double2*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_QF:
      internal::int8::dequantize_i63_f128_qf(stream, bits, order, M, N, acc, algnN, umax, jpiv, (float4*)A, algnN);
      K = internal::Cholesky::potrfp_f128_qf(stream, handle, epi, K, p, N, (float4*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_F64_COMPLEX:
      internal::int8::dequantize_i63_cf64(stream, bits, order, M, N, acc, algnN, umax, jpiv, (std::complex<double>*)A, algnN);
      K = internal::Cholesky::potrfp_cf64(stream, handle, epi, K, p, N, (std::complex<double>*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_F32_COMPLEX:
      internal::int8::dequantize_i63_cf32(stream, bits, order, M, N, acc, algnN, umax, jpiv, (std::complex<float>*)A, algnN);
      K = internal::Cholesky::potrfp_cf32(stream, handle, epi, K, p, N, (std::complex<float>*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_DD_COMPLEX:
      internal::int8::dequantize_i63_cf128_dd(stream, bits, order, M, N, acc, algnN, umax, jpiv, (complex_double2*)A, algnN);
      K = internal::Cholesky::potrfp_cf128_dd(stream, handle, epi, K, p, N, (complex_double2*)A, algnN, hpiv, pinned_work); break;
    case HYACIN_QF_COMPLEX:
      internal::int8::dequantize_i63_cf128_qf(stream, bits, order, M, N, acc, algnN, umax, jpiv, (complex_float4*)A, algnN);
      K = internal::Cholesky::potrfp_cf128_qf(stream, handle, epi, K, p, N, (complex_float4*)A, algnN, hpiv, pinned_work); break;
    default: break;
  }

  cudaMemcpyAsync(jpiv, hpiv, int64_t(N) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  if (R != nullptr && 0 < K && K <= ldr) {
    if (mode == 'R' || mode == 'r')
      conv_copy_dispatcher<0>(stream, int64_t(K), N, nullptr, A, int64_t(algnN), ComputeType, R, int64_t(ldr), Rtype);
    else if (mode == 'J' || mode == 'j')
      conv_copy_dispatcher<1>(stream, int64_t(K), N, jpiv, A, int64_t(algnN), ComputeType, R, int64_t(ldr), Rtype);
  }
  return K;
}

extern "C" void hyacinXcpqrk_bufferSize(int32_t M, int32_t N, int32_t umax, hyacinPrecision_t ComputeType, hyacinAlgorithm_t alg, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  int32_t Complex, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes, idx_bytes;
  std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes, idx_bytes) = ext_params(M, int64_t(M), N, umax, 63, ComputeType, alg);
  *dev_work_bytes = uint64_t(i8_bytes + acc_bytes);
  *pinned_work_bytes = uint64_t(vec_bytes + idx_bytes);
}

extern "C" void hyacinXcpqrk1Dcol_bufferSize(int32_t localM, int32_t globalM, int32_t N, int32_t umax, hyacinPrecision_t ComputeType, hyacinAlgorithm_t alg, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  int32_t Complex, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes, idx_bytes;
  std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes, idx_bytes) = ext_params(localM, globalM, N, umax, 47, ComputeType, alg);
  *dev_work_bytes = uint64_t(i8_bytes + acc_bytes);
  *pinned_work_bytes = uint64_t(vec_bytes + idx_bytes);
}

extern "C" int32_t hyacinXcpqrk(cublasHandle_t handle, char mode, double epi, int32_t M, int32_t N, int32_t K, int32_t p, int32_t umax,
  hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* jpiv, hyacinPrecision_t Rtype, void* R, int32_t ldr, hyacinPrecision_t ComputeType, void* dev_work, void* pinned_work, hyacinAlgorithm_t alg) {

  int32_t Complex, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes, idx_bytes;
  std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes, idx_bytes) = ext_params(M, M, N, umax, 63, ComputeType, alg);

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes];
  int32_t* hpiv = (int32_t*)(&((int8_t*)pinned_work)[idx_bytes]);

  vexp_dispatcher(stream, M, N, Atype, A, lda, jpiv);
  igemm_dispatcher(stream, handle, M, N, Atype, A, lda, umax, jpiv, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);

  return rrf_dispatcher(stream, handle, mode, epi, M, N, algnN, K, p, umax, (uint64_t*)acc, 63, orderC, iA, jpiv, hpiv, Rtype, R, ldr, ComputeType, pinned_work);
}

#ifndef NO_NCCL

extern "C" int32_t hyacinXcpqrk1Dcol(cublasHandle_t handle, char mode, double epi, int32_t localM, int32_t globalM, int32_t N, int32_t K, int32_t p, int32_t umax,
  hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t* jpiv, hyacinPrecision_t Rtype, void* R, int32_t ldr, hyacinPrecision_t ComputeType, void* dev_work, void* pinned_work, hyacinAlgorithm_t alg, ncclComm_t col_comm) {

  int32_t Complex, algnM, algnN, orderA, orderC; int64_t i8_bytes, acc_bytes, vec_bytes, idx_bytes;
  std::tie(Complex, algnM, algnN, orderA, orderC, i8_bytes, acc_bytes, vec_bytes, idx_bytes) = ext_params(localM, globalM, N, umax, 47, ComputeType, alg);

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int8_t* iA = (int8_t*)(dev_work), *acc = &iA[i8_bytes];
  vexp_dispatcher(stream, localM, N, Atype, A, lda, (int32_t*)jpiv);
  ncclAllReduce(jpiv, jpiv, int64_t(N), ncclInt32, ncclMax, col_comm, stream);

  igemm_dispatcher(stream, handle, localM, N, Atype, A, lda, umax, (const int32_t*)jpiv, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA, alg);
  int64_t stride = int64_t(algnN) * int64_t(N) + int64_t(algnN), acc_len = stride * (int64_t(orderC + 1) << Complex);
  internal::int8::accumulate_conv_i63_u47(stream, orderC, Complex, stride, (uint64_t*)acc);
  ncclAllReduce(acc, acc, acc_len, ncclUint64, ncclSum, col_comm, stream);

  int32_t* hpiv = (int32_t*)(&((int8_t*)pinned_work)[idx_bytes]);
  return rrf_dispatcher(stream, handle, mode, epi, globalM, N, algnN, K, p, umax, (uint64_t*)acc, 47, orderC, iA, jpiv, hpiv, Rtype, R, ldr, ComputeType, pinned_work);
}

#endif
