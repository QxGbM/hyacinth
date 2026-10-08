
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(32) complex_double2 { double2 real, imag; };
struct __align__(32) complex_float4 { float4 real, imag; };

namespace device {

  __host__ __device__ __forceinline__ complex_double2 make_complex_double2(double2 real, double2 imag) { return complex_double2({ real, imag }); }
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

  __host__ __device__ __forceinline__ void two_sum(const double a, const double b, double& sum, double& err)
  { sum = a + b; double d = a - sum, sd = sum + d, bd = b + d; err = (a - sd) + bd; }

  __host__ __device__ __forceinline__ void two_sum(const float a, const float b, float& sum, float& err)
  { sum = a + b; float d = a - sum, sd = sum + d, bd = b + d; err = (a - sd) + bd; }

  __host__ __device__ __forceinline__ void two_sum(const float2 a, const float2 b, float2& sum, float2& err) {
    sum = make_float2(a.x + b.x, a.y + b.y);
    float2 d = make_float2(a.x - sum.x, a.y - sum.y), sd = make_float2(sum.x + d.x, sum.y + d.y), bd = make_float2(b.x + d.x, b.y + d.y);
    err = make_float2((a.x - sd.x) + bd.x, (a.y - sd.y) + bd.y);
  }

  __host__ __device__ __forceinline__ double2 renormalize(const double2& a)
  { double sum = a.x + a.y, d = a.x - sum; return make_double2(sum, a.y + d); }

  __host__ __device__ __forceinline__ float4 renormalize(const float4& a) {
    float s0, s1, d0, d1, x, y, z, w;
    s0 = a.x + a.y; d0 = a.x - s0; x = s0; y = a.y + d0;
    s0 = x + a.z; d0 = x - s0; x = s0; z = a.z + d0;
    s1 = y + a.w; d1 = y - s1; y = s1; w = a.w + d1;
    s0 = x + w; d0 = x - s0; x = s0; w += d0;
    s1 = y + z; d1 = y - s1; y = s1; z += d1;
    s1 = z + w; d1 = z - s1; z = s1; w += d1;
    return make_float4(x, y, z, w);
  }

  __host__ __device__ __forceinline__ double2 add(const double2& a, const double2& b)
  { double sum, err, a2 = a.y + b.y; two_sum(a.x, b.x, sum, err); return renormalize(make_double2(sum, a2 + err)); }

  __host__ __device__ __forceinline__ float4 add(const float4& a, const float4& b) {
    float2 a0, a1, a2;
    two_sum(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1); // 1122 - 1223
    two_sum(a.z, b.z, a2.x, a2.y); // 33 - 34, 4@a2.y

    float r0 = a0.x; a2.y += a.w + b.w;
    two_sum(make_float2(a0.y, a2.x), a1, a0, a1); // 2233 - 2334, 4@a1.y
    two_sum(a0.y, a1.x, a0.y, a1.x); // 33 - 34, 4@a1.x
    return renormalize(make_float4(r0, a0.x, a0.y, a2.y + (a1.y + a1.x)));
  }

};
