#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace device::int8 {
  constexpr uint32_t i7 = (uint32_t(1) << 7) - 1;
  constexpr uint32_t i14 = (uint32_t(1) << 14) - 1;
  constexpr uint32_t i21 = (uint32_t(1) << 21) - 1;
  constexpr uint32_t i28 = (uint32_t(1) << 28) - 1;

  constexpr uint32_t i9 = (uint32_t(1) << 9) - 1;
  constexpr uint32_t i16 = (uint32_t(1) << 16) - 1;
  constexpr uint32_t i23 = (uint32_t(1) << 23) - 1;

  constexpr uint32_t i8 = (uint32_t(1) << 8) - 1;
  constexpr uint32_t i11 = (uint32_t(1) << 11) - 1;
  constexpr uint64_t i52 = (uint64_t(1) << 52) - 1;

  __host__ __device__ __forceinline__ int32_t get_double_top_exp(double value) {
    union { double d; uint64_t u; } v {value};
    return (int32_t(v.u >> 52) & i11) - 1023;
  }

  __host__ __device__ __forceinline__ int32_t get_float_top_exp(float value) {
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

  __host__ __device__ __forceinline__ uint32_t vcond_negate4(uint32_t a, int32_t pred) {
#ifdef __CUDA_ARCH__
    return pred ? __vneg4(a) : a;
#else
    if (pred) {
      int8_t a0 = int8_t(a & i8);
      int8_t a1 = int8_t((a >> 8) & i8);
      int8_t a2 = int8_t((a >> 16) & i8);
      int8_t a3 = int8_t((a >> 24) & i8);

      uint32_t r0 = uint8_t(-a0);
      uint32_t r1 = uint8_t(-a1);
      uint32_t r2 = uint8_t(-a2);
      uint32_t r3 = uint8_t(-a3);

      return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
    }
    return a;
#endif
  }

  __host__ __device__ __forceinline__ void encode_double_exp7_9xi8(double value, int32_t& e, uint32_t (&code)[3], int32_t em4 = 0) {
    union { double d; int64_t u; } v {value};

    int32_t sign = int32_t(v.u >> 63);
    int32_t exp = (int32_t(v.u >> 52) & i11) - 1075, rem; // bias1023 + frac52
    fast_div7_i32x(exp, e, rem);

    uint64_t frac = ((v.u & i52) | (i52 + 1)) << rem;
    frac = -1075 < exp ? frac : int64_t(0);

    uint32_t f32 = uint32_t(frac);
    uint32_t a0 = f32 & i7;
    uint32_t a1 = (f32 << 1) & (i7 << 8);
    uint32_t a2 = (f32 << 2) & (i7 << 16);
    uint32_t a3 = (f32 << 3) & (i7 << 24);
    code[0] = vcond_negate4(a0 | a1 | a2 | a3, sign);

    f32 = uint32_t(frac >> 28);
    a0 = f32 & i7;
    a1 = (f32 << 1) & (i7 << 8);
    a2 = (f32 << 2) & (i7 << 16);
    a3 = (f32 << 3) & (i7 << 24);
    code[1] = vcond_negate4(a0 | a1 | a2 | a3, sign);
    code[2] = vcond_negate4(uint32_t(frac >> 56) & i7, sign);

    rem = (e - em4) & 3;
    e += -rem; // e = em4 mod 4, right shift code to align

    int32_t rsft = 32 - (rem << 3);
    code[2] = uint32_t(((uint64_t(code[2]) << 32) | uint64_t(code[1])) >> rsft);
    code[1] = uint32_t(((uint64_t(code[1]) << 32) | uint64_t(code[0])) >> rsft);
    code[0] = code[0] << (32 - rsft);
  }

  __host__ __device__ __forceinline__ void encode_float_exp7_5xi8(float value, int32_t& e, uint32_t (&code)[2], int32_t em4 = 0) {
    union { float d; int32_t u; } v {value};

    int32_t sign = int32_t(v.u >> 31);
    int32_t exp = (int32_t(v.u >> 23) & i8) - 150, rem; // bias127 + frac23
    fast_div7_i32x(exp, e, rem);

    uint32_t frac = ((v.u & i23) | (i23 + 1)) << rem;
    frac = -150 < exp ? frac : int32_t(0);

    uint32_t a0 = frac & i7;
    uint32_t a1 = (frac << 1) & (i7 << 8);
    uint32_t a2 = (frac << 2) & (i7 << 16);
    uint32_t a3 = (frac << 3) & (i7 << 24);
    code[0] = vcond_negate4(a0 | a1 | a2 | a3, sign);
    code[1] = vcond_negate4((frac >> 28) & i7, sign);

    rem = (e - em4) & 3;
    e += -rem;

    int32_t rsft = 32 - (rem << 3);
    code[1] = uint32_t(((uint64_t(code[1]) << 32) | uint64_t(code[0])) >> rsft);
    code[0] = code[0] << (32 - rsft);
  }

  __host__ __device__ __forceinline__ void decode_scaled_int4_double(int32_t const (&code)[4], double& value) {
    uint4 code_lo = make_uint4(uint32_t(code[0]) & i28, uint32_t(code[1]) & i21, uint32_t(code[2]) & i14, uint32_t(code[3]) & i7);
    uint32_t lo = code_lo.x + (code_lo.y << 7) + (code_lo.z << 14) + (code_lo.w << 21);
    int32_t hi = (code[0] >> 28) + (code[1] >> 21) + (code[2] >> 14) + (code[3] >> 7);
    value = (double)(((int64_t)hi << 28) + lo);
  }

  __host__ __device__ __forceinline__ void decode_scaled_int3_float2(int32_t const (&code)[3], float (&value)[2]) {
    uint3 code_lo = make_uint3(uint32_t(code[0]) & i23, uint32_t(code[1]) & i16, uint32_t(code[2]) & i9);
    uint32_t lo = code_lo.x + (code_lo.y << 7) + (code_lo.z << 14);
    int32_t hi = (lo >> 23) + (code[0] >> 23) + (code[1] >> 16) + (code[2] >> 9);
    value[0] = (float)((int64_t)hi << 23);
    value[1] = (float)(lo & i23);
  }

};
