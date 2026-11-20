#pragma once

#include <stdint.h>
#include <cublas_v2.h>

struct __align__(alignof(double2) + alignof(double2)) complex_double2 { double2 real, imag; };
struct __align__(alignof(float3) + alignof(float3)) complex_float3 { float3 real, imag; };
struct __align__(alignof(float4) + alignof(float4)) complex_float4 { float4 real, imag; };

namespace internal::Cholesky {

  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, float3* A, int32_t lda, int32_t* jpiv, float3* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, complex_float3* A, int32_t lda, int32_t* jpiv, float3* dev_work, int32_t* r, void* logger);
  int32_t potrfp(cudaStream_t stream, double eps, int32_t K, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* r, void* logger);

};

namespace internal::int8 {

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);

  void accumulate_limbs_diag(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, int32_t orderA, uint64_t* A);
  void accumulate_limbs_cross_real(cudaStream_t stream, int32_t beta, int32_t N, int32_t sft, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void accumulate_limbs_cross_imag(cudaStream_t stream, int32_t beta, int32_t N, int32_t sft, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);

  void accumulate_crt_diag_real(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, uint32_t* A);
  void accumulate_crt_rotate(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, uint32_t* A);
  void accumulate_crt_extra_real(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, uint32_t* A);
  void accumulate_crt_extra_imag(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, uint32_t* A);
  void reconstruct_real(cudaStream_t stream, int32_t beta, int64_t N, int32_t orderX, const uint32_t* X, int32_t orderA, uint64_t* A);
  void reconstruct_complex(cudaStream_t stream, int32_t beta, int64_t N, int32_t orderX, const uint32_t* X, int32_t orderA, uint64_t* A);

  void vector_range(cudaStream_t stream, int32_t batches, int32_t N, int32_t u, const int32_t* vexp, int32_t* segU);

};

namespace internal {

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void matmul_cf16(cudaStream_t stream, cublasHandle_t handle, cudaMemPool_t mempool, char opA, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, const __half2* B, int32_t ldb, __half2* C, int32_t ldc);

};

namespace internal::timer {
  void register_distribute_kernel(cudaStream_t stream, void* timer);
  void register_replicate_kernel(cudaStream_t stream, void* timer);
  void register_comm(cudaStream_t stream, void* timer);
  void register_untimed(cudaStream_t stream, void* timer);
};

#ifdef __CUDACC__
namespace device {

  template <uint32_t expon> __device__ __forceinline__ void add_shifted(uint64_t (&a)[1], int64_t i) noexcept {
    if constexpr(expon == uint32_t(0)) { a[0] += uint64_t(i); } else
    if constexpr(expon <= uint32_t(63)) { a[0] += uint64_t(i) << expon; }
  } 

  template <uint32_t expon> __device__ __forceinline__ void add_shifted(uint64_t (&a)[2], int64_t i) noexcept {
    constexpr uint64_t i63 = 0x7fffffffffffffffllu;
    if constexpr(expon == uint32_t(0)) {
      uint64_t sign = -(uint64_t(i) >> 63); a[0] += uint64_t(i) & i63; a[1] += sign + (a[0] >> 63); a[0] &= i63; 
    } else if constexpr(expon == uint32_t(63)) {
      a[1] += uint64_t(i); 
    } else if constexpr(expon < uint32_t(63)) {
      constexpr uint32_t left_sft = expon, sign_sft = uint32_t(1) + left_sft, right_sft = uint32_t(63) - expon;
      uint64_t q0 = uint64_t(i) << left_sft, sign = -(uint64_t(i) >> 63), q1 = (sign << sign_sft) | (uint64_t(i) >> right_sft);
      a[0] += q0 & i63; a[1] += q1 + (a[0] >> 63); a[0] &= i63;
    } else if constexpr(expon <= uint32_t(126)) {
      constexpr uint32_t left_sft = expon - uint32_t(63);
      a[1] += uint64_t(i) << left_sft;
    }
  }

  template <uint32_t expon> __device__ __forceinline__ void add_shifted(uint64_t (&a)[3], int64_t i) noexcept {
    constexpr uint64_t i63 = 0x7fffffffffffffffllu;
    if constexpr(expon == uint32_t(0)) {
      uint64_t sign = -(uint64_t(i) >> 63); a[0] += uint64_t(i) & i63;
      a[1] += (sign & i63) + (a[0] >> 63); a[0] &= i63; a[2] += sign + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon == uint32_t(63)) {
      uint64_t sign = -(uint64_t(i) >> 63); a[1] += uint64_t(i) & i63;
      a[2] += sign + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon == uint32_t(126)) {
      a[2] += uint64_t(i);
    } else if constexpr(expon < uint32_t(63)) {
      constexpr uint32_t left_sft = expon, sign_sft = uint32_t(1) + left_sft, right_sft = uint32_t(63) - expon;
      uint64_t q0 = uint64_t(i) << left_sft, sign = -(uint64_t(i) >> 63), q1 = (sign << sign_sft) | (uint64_t(i) >> right_sft);
      a[0] += q0 & i63; a[1] += (q1 & i63) + (a[0] >> 63); a[0] &= i63; a[2] += sign + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon < uint32_t(126)) {
      constexpr uint32_t left_sft = expon - uint32_t(63), sign_sft = uint32_t(1) + left_sft, right_sft = uint32_t(126) - expon;
      uint64_t q0 = uint64_t(i) << left_sft, sign = -(uint64_t(i) >> 63), q1 = (sign << sign_sft) | (uint64_t(i) >> right_sft);
      a[1] += q0 & i63; a[2] += q1 + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon <= uint32_t(189)) {
      constexpr uint32_t left_sft = expon - uint32_t(126);
      a[2] += uint64_t(i) << left_sft;
    }
  }

  template<uint32_t DIV>
  __device__ __forceinline__ uint32_t barrett_reduc(uint32_t x) noexcept {
    if constexpr(DIV) {
      constexpr uint32_t d = uint32_t((0x100000000llu / uint64_t(DIV))), minus_div = -DIV;
      x += __umulhi(x, d) * minus_div; return __viaddmin_u32(minus_div, x, x);
    } else return uint32_t(0);
  }

  template<int32_t sign, uint32_t MUL, uint32_t DIV>
  __device__ __forceinline__ uint32_t mulx_reduc(uint32_t x) noexcept {
    if constexpr(MUL && DIV) {
      static_assert(DIV <= 4096u, "Modular needs to be smaller than 4096");
      constexpr uint32_t mul = MUL % DIV, mul_r16 = (mul << 16) % DIV;
      uint32_t mul_x;
      if constexpr(mul == uint32_t(1)) { mul_x = (mul_r16 * (x >> 16)) + (x & uint32_t(0xffff)); } else
      { mul_x = (mul_r16 * (x >> 16)) + (mul * (x & uint32_t(0xffff))); }
      if constexpr(sign) { constexpr uint32_t mul_r32 = ((DIV - mul_r16) << 16) % DIV; return mul_x + ((-(x >> 31)) & mul_r32); } else { return mul_x; }
    } else return uint32_t(0);
  }

  __device__ __forceinline__ void two_sum(const double a, const double b, double& s, double& e) noexcept
  { s = a + b; double d = a - s, sd = s + d; e = (b + d) + (a - sd); }

  __device__ __forceinline__ void two_sum(const float a, const float b, float& s, float& e) noexcept
  { s = a + b; float d = a - s, sd = s + d; e = (b + d) + (a - sd); }

  __device__ __forceinline__ void two_sum(const float a0, const float a1, const float b0, const float b1, float& s0, float& s1, float& e0, float& e1) noexcept {
    s0 = a0 + b0; s1 = a1 + b1;
    float2 d = make_float2(a0 - s0, a1 - s1), sd = make_float2(s0 + d.x, s1 + d.y);
    e0 = (b0 + d.x) + (a0 - sd.x); e1 = (b1 + d.y) + (a1 - sd.y);
  }

  __device__ __forceinline__ void two_sum(const float a0, const float a1, const float a2, const float b0, const float b1, const float b2, float& s0, float& s1, float& s2, float& e0, float& e1, float& e2) noexcept {
    s0 = a0 + b0; s1 = a1 + b1; s2 = a2 + b2;
    float3 d = make_float3(a0 - s0, a1 - s1, a2 - s2), sd = make_float3(s0 + d.x, s1 + d.y, s2 + d.z);
    e0 = (b0 + d.x) + (a0 - sd.x); e1 = (b1 + d.y) + (a1 - sd.y); e2 = (b2 + d.z) + (a2 - sd.z);
  }

  __device__ __forceinline__ double2 renormalize(const double2& a) noexcept
  { double s = a.x + a.y, d = a.x - s; return make_double2(s, a.y + d); }

  __device__ __forceinline__ float3 renormalize(const float3& a) noexcept {
    float x = a.x + a.y, y = (a.x - x) + a.y, s0 = x + a.z, z = (x - s0) + a.z, s1 = y + z;
    return make_float3(s0, s1, (y - s1) + z);
  }

  __device__ __forceinline__ float4 renormalize(const float4& a) noexcept {
    float x = a.x + a.y, y = (a.x - x) + a.y, s0 = x + a.z, z = (x - s0) + a.z, s1 = y + z, s2 = (y - s1) + z;
    x = s0 + a.w; float w = (s0 - x) + a.w; y = s1 + w; z = s2 + (w += s1 - y);
    return make_float4(x, y, z, (s2 - z) + w); 
  }

  __device__ __forceinline__ double2 add(const double2& a, const double2& b) noexcept
  { double s, e; two_sum(a.x, b.x, s, e); return renormalize(make_double2(s, (a.y + b.y) + e)); }

  __device__ __forceinline__ float3 add(const float3& a, const float3& b) noexcept {
    float2 s, e; two_sum(a.x, a.y, b.x, b.y, s.x, s.y, e.x, e.y); two_sum(s.y, e.x, s.y, e.x);
    return renormalize(make_float3(s.x, s.y, (a.z + b.z) + (e.x + e.y)));
  }

  __device__ __forceinline__ float4 add(const float4& a, const float4& b) noexcept {
    float3 s, e;
    two_sum(a.x, a.y, a.z, b.x, b.y, b.z, s.x, s.y, s.z, e.x, e.y, e.z);
    two_sum(s.y, s.z, e.x, e.y, s.y, s.z, e.x, e.y);
    two_sum(s.z, e.x, s.z, e.x);
    return renormalize(make_float4(s.x, s.y, s.z, (a.w + b.w) + (e.x + e.y + e.z)));
  }

};
#endif

#ifndef NDEBUG
#include <cassert>
#define CUDA_CHECK(call) do { cudaError_t err = (call); assert(err == cudaSuccess); } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t err = (call); assert(err == CUBLAS_STATUS_SUCCESS); } while(0)
#define CUSOLVER_CHECK(call) do { cusolverStatus_t err = (call); assert(err == CUSOLVER_STATUS_SUCCESS); } while(0)
#else
#define CUDA_CHECK(call) (call)
#define CUBLAS_CHECK(call) (call)
#define CUSOLVER_CHECK(call) (call)
#endif
