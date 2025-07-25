#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace device::int8 {

  __host__ __device__ __forceinline__ int32_t get_double_top_exp(double value) {
    constexpr uint32_t i11 = (uint32_t(1) << 11) - 1;
    union { double d; uint64_t u; } v {value};
    return (int32_t(v.u >> 52) & i11) - 1023;
  }

  __host__ __device__ __forceinline__ int32_t get_float_top_exp(float value) {
    constexpr uint32_t i8 = (uint32_t(1) << 8) - 1;
    union { float d; uint32_t u; } v {value};
    return (int32_t(v.u >> 23) & i8) - 127;
  }

  __host__ __device__ __forceinline__ void fast_div7_i32x(int32_t x, int32_t& quo, int32_t& rem) {
    int32_t m_num = 0x24924924 + (uint32_t(~x) >> 31); // x-:floor(2^32 / 7) x+: ceil(2^32 / 7)
#ifdef __CUDA_ARCH__
    quo = __mulhi(x, m_num);
#else
    quo = int32_t((int64_t(x) * int64_t(m_num)) >> 32);
#endif
    rem = x - quo * 7;
  }

  __host__ __device__ __forceinline__ uint32_t vadd4(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __vadd4(a, b);
#else
    union { uint8_t bytes[4]; uint32_t i; } v;
    v.bytes[0] = uint8_t(a) + uint8_t(b);
    v.bytes[1] = uint8_t(a >> 8) + uint8_t(b >> 8);
    v.bytes[2] = uint8_t(a >> 16) + uint8_t(b >> 16);
    v.bytes[3] = uint8_t(a >> 24) + uint8_t(b >> 24);
    return v.i;
#endif
  }

  __host__ __device__ __forceinline__ void encode_double_exp7_9xi8(double value, int32_t& e, uint32_t (&code)[3]) {
    constexpr uint32_t i7 = (uint32_t(1) << 7) - 1;
    constexpr uint32_t i11 = (uint32_t(1) << 11) - 1;
    constexpr uint64_t i52 = (uint64_t(1) << 52) - 1;

    union { double d; int64_t u; } v {value};

    int32_t sign = int32_t(v.u >> 63);
    int32_t exp = (int32_t(v.u >> 52) & i11) - 1075, rem; // bias1023 + frac52
    fast_div7_i32x(exp, e, rem);

    int64_t impl_one = -int64_t(-1075 < exp) & (i52 + 1);
    uint64_t frac = ((v.u & i52) | impl_one) << rem;
    uint64_t sign_frac = sign ? ~frac : frac;

    uint32_t cmpl_i8 = sign & 0x81818181;
    uint32_t f32 = uint32_t(sign_frac);

    uint32_t a0 = f32 & i7;
    uint32_t a1 = (f32 << 1) & (i7 << 8);
    uint32_t a2 = (f32 << 2) & (i7 << 16);
    uint32_t a3 = (f32 << 3) & (i7 << 24);
    code[0] = vadd4(a0 | a1 | a2 | a3, cmpl_i8);

    f32 = uint32_t(sign_frac >> 28);
    a0 = f32 & i7;
    a1 = (f32 << 1) & (i7 << 8);
    a2 = (f32 << 2) & (i7 << 16);
    a3 = (f32 << 3) & (i7 << 24);
    code[1] = vadd4(a0 | a1 | a2 | a3, cmpl_i8);
    code[2] = uint8_t((uint32_t(sign_frac >> 56) & i7) + cmpl_i8);
  }

  __host__ __device__ __forceinline__ void encode_float_exp7_5xi8(float value, int32_t& e, uint32_t (&code)[2]) {
    constexpr uint32_t i7 = (uint32_t(1) << 7) - 1;
    constexpr uint32_t i8 = (uint32_t(1) << 8) - 1;
    constexpr uint32_t i23 = (uint32_t(1) << 23) - 1;

    union { float d; int32_t u; } v {value};

    int32_t sign = v.u >> 31;
    int32_t exp = ((v.u >> 23) & i8) - 150, rem; // bias127 + frac23
    fast_div7_i32x(exp, e, rem);

    int32_t impl_one = -int32_t(-150 < exp) & (i23 + 1);
    uint32_t frac = ((v.u & i23) | impl_one) << rem;
    uint32_t sign_frac = sign ? ~frac : frac;

    uint32_t cmpl_i8 = sign & 0x81818181;

    uint32_t a0 = sign_frac & i7;
    uint32_t a1 = (sign_frac << 1) & (i7 << 8);
    uint32_t a2 = (sign_frac << 2) & (i7 << 16);
    uint32_t a3 = (sign_frac << 3) & (i7 << 24);
    code[0] = vadd4(a0 | a1 | a2 | a3, cmpl_i8);
    code[1] = uint8_t((uint32_t(sign_frac >> 28) & i7) + cmpl_i8);
  }

  template <int32_t order>
  __host__ __device__ __forceinline__ void align_expon(uint32_t (&a)[order], int32_t exp7_diff) {
    uint32_t rsft = 32 - ((exp7_diff & 3) << 3);
    int32_t psft = (exp7_diff - (exp7_diff & 3)) >> 2;
    uint32_t b[order + 1];
    b[0] = a[0] << (32 - rsft);
    b[order] = a[order - 1] >> rsft;

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (int32_t i = 1; i < order; ++i)
      b[i] = uint32_t(((uint64_t(a[i]) << 32) | uint64_t(a[i - 1])) >> rsft);

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (int32_t i = 0; i < order; ++i) {
      int32_t j = i - psft;
      a[i] = (0 <= j && j <= order) ? b[j] : 0;
    }
  }

  __host__ __device__ __forceinline__ int32_t decode_scaled_3xi32(int32_t const (&a)[3], int32_t& c) {
    constexpr uint32_t i21 = (uint32_t(1) << 21) - 1;

    int32_t lo = (uint32_t(c) & i21) + (uint32_t(a[0]) & i21) + (uint32_t(a[1] << 7) & i21) + (uint32_t(a[2] << 14) & i21);
    int32_t hi = (c >> 21) + (a[0] >> 21) + (a[1] >> 14) + (a[2] >> 7) + (lo >> 21);
    int32_t sign = hi >> 31;

    c = hi - sign;
    return (lo & i21) | (sign & ~i21);
  }

  __host__ __device__ __forceinline__ int64_t decode_scaled_7xi32(int32_t const (&a)[7], int32_t& c) {
    constexpr uint32_t i21 = (uint32_t(1) << 21) - 1;
    constexpr uint32_t i28 = (uint32_t(1) << 28) - 1;
    constexpr uint64_t i49 = (uint64_t(1) << 49) - 1;

    int32_t lo4 = (uint32_t(c) & i28) + (uint32_t(a[0]) & i28) + (uint32_t(a[1] << 7) & i28) + (uint32_t(a[2] << 14) & i28) + (uint32_t(a[3] << 21) & i28);
    int32_t hi4 = (c >> 28) + (a[0] >> 28) + (a[1] >> 21) + (a[2] >> 14) + (a[3] >> 7) + (lo4 >> 28);

    int32_t lo3 = (uint32_t(hi4) & i21) + (uint32_t(a[4]) & i21) + (uint32_t(a[5] << 7) & i21) + (uint32_t(a[6] << 14) & i21);
    int32_t hi3 = (hi4 >> 21) + (a[4] >> 21) + (a[5] >> 14) + (a[6] >> 7) + (lo3 >> 21);
    int32_t sign = hi3 >> 31;

    c = hi3 - sign;
    return int64_t(lo4 & i28) | (int64_t(lo3 & i21) << 28) | (int64_t(sign) & ~i49);
  }

};
