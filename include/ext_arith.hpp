
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(32) complex_double2 { double2 real, imag; };
struct __align__(16) complex_float3 { float3 real, imag; };
struct __align__(32) complex_float4 { float4 real, imag; };

namespace device {

  __host__ __device__ __forceinline__ complex_double2 make_complex_double2(double2 real, double2 imag) { return complex_double2({ real, imag }); }
  __host__ __device__ __forceinline__ complex_float3 make_complex_float3(float3 real, float3 imag) { return complex_float3({ real, imag }); }
  __host__ __device__ __forceinline__ complex_float4 make_complex_float4(float4 real, float4 imag) { return complex_float4({ real, imag }); }

  template <uint32_t expon> __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[1], int64_t i) {
    if constexpr(expon == uint32_t(0)) { a[0] += uint64_t(i); } else
    if constexpr(expon <= uint32_t(63)) { a[0] += uint64_t(i) << expon; }
  } 

  template <uint32_t expon> __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[2], int64_t i) {
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

  template <uint32_t expon> __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[3], int64_t i) {
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
  __host__ __device__ __forceinline__ uint32_t barrett_reduc(uint32_t x) {
#ifdef __CUDA_ARCH__
    if constexpr(DIV) {
      constexpr uint32_t d = uint32_t((0x100000000llu / uint64_t(DIV))), minus_div = -DIV;
      x += __umulhi(x, d) * minus_div; return __viaddmin_u32(minus_div, x, x);
    } else return uint32_t(0);
#else
    if constexpr(DIV) return x % DIV; else return uint32_t(0);
#endif
  }

  template<uint32_t MUL, uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t mulx_reduc(uint32_t x) {
    if constexpr(MUL && DIV) {
      static_assert(DIV <= 4096u, "Modular needs to be smaller than 4096");
      constexpr uint32_t mul = MUL % DIV, mul_r16 = (mul << 16) % DIV;
      return (mul_r16 * (x >> 16)) + (mul * (x & uint32_t(0xffff)));
    } else return uint32_t(0);
  }

  __host__ __device__ __forceinline__ void two_sum(const double a, const double b, double& s, double& e)
  { s = a + b; double d = a - s, sd = s + d; e = (b + d) + (a - sd); }

  __host__ __device__ __forceinline__ void two_sum(const float a, const float b, float& s, float& e)
  { s = a + b; float d = a - s, sd = s + d; e = (b + d) + (a - sd); }

  __host__ __device__ __forceinline__ void two_sum(const float a0, const float a1, const float b0, const float b1, float& s0, float& s1, float& e0, float& e1) {
    s0 = a0 + b0; s1 = a1 + b1;
    float2 d = make_float2(a0 - s0, a1 - s1), sd = make_float2(s0 + d.x, s1 + d.y);
    e0 = (b0 + d.x) + (a0 - sd.x); e1 = (b1 + d.y) + (a1 - sd.y);
  }

  __host__ __device__ __forceinline__ void two_sum(const float a0, const float a1, const float a2, const float b0, const float b1, const float b2, float& s0, float& s1, float& s2, float& e0, float& e1, float& e2) {
    s0 = a0 + b0; s1 = a1 + b1; s2 = a2 + b2;
    float3 d = make_float3(a0 - s0, a1 - s1, a2 - s2), sd = make_float3(s0 + d.x, s1 + d.y, s2 + d.z);
    e0 = (b0 + d.x) + (a0 - sd.x); e1 = (b1 + d.y) + (a1 - sd.y); e2 = (b2 + d.z) + (a2 - sd.z);
  }

  __host__ __device__ __forceinline__ double2 renormalize(const double2& a)
  { double s = a.x + a.y, d = a.x - s; return make_double2(s, a.y + d); }

  __host__ __device__ __forceinline__ float3 renormalize(const float3& a) {
    float x = a.x + a.y, y = (a.x - x) + a.y, s0 = x + a.z, z = (x - s0) + a.z, s1 = y + z;
    return make_float3(s0, s1, z + (y - s1));
  }

  __host__ __device__ __forceinline__ float4 renormalize(const float4& a) {
    float x = a.x + a.y, y = (a.x - x) + a.y, s0 = x + a.z, s1 = y + a.w, z = (x - s0) + a.z, w = (y - s1) + a.w;
    x = s0; y = s1; s0 = x + w; s1 = y + z; w += x - s0; z += y - s1;
    float s2 = z + w; return make_float4(s0, s1, s2, w + (z - s2));
  }

  __host__ __device__ __forceinline__ double2 add(const double2& a, const double2& b)
  { double s, e; two_sum(a.x, b.x, s, e); return renormalize(make_double2(s, (a.y + b.y) + e)); }

  __host__ __device__ __forceinline__ float3 add(const float3& a, const float3& b) {
    float2 s, e; two_sum(a.x, a.y, b.x, b.y, s.x, s.y, e.x, e.y); two_sum(s.y, e.x, s.y, e.x);
    return renormalize(make_float3(s.x, s.y, (a.z + b.z) + (e.y + e.x)));
  }

  __host__ __device__ __forceinline__ float4 add(const float4& a, const float4& b) {
    float3 s, e;
    two_sum(a.x, a.y, a.z, b.x, b.y, b.z, s.x, s.y, s.z, e.x, e.y, e.z);
    two_sum(s.y, s.z, e.x, e.y, s.y, s.z, e.x, e.y); e.z += a.w + b.w;
    two_sum(s.z, e.x, s.z, e.x);
    return renormalize(make_float4(s.x, s.y, s.z, e.x + e.y + e.z));
  }

};
