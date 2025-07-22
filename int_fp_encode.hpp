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

  __host__ __device__ __forceinline__ int32_t get_double_exp(double value) {
    union { double d; uint64_t u; } v {value};
    return (int32_t(v.u >> 52) & i11) - 1075; // bias1023 + frac52
  }

  __host__ __device__ __forceinline__ int32_t get_float_exp(float value) {
    union { float d; uint32_t u; } v {value};
    return (int32_t(v.u >> 23) & i8) - 150;  // bias127 + frac23
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

  __host__ __device__ __forceinline__ void encode_double_exp7_9xi8(double value, int32_t& e, int3& code) {
    union { double d; uint64_t u; } v {value};

    int32_t sign = int32_t(v.u >> 63);
    int32_t exp = (int32_t(v.u >> 52) & i11) - 1075, rem;
    fast_div7_i32x(exp, e, rem);

    v.u = ((v.u & i52) | (i52 + 1)) << rem;
    int64_t frac = (-1075 < exp ? (sign ? -int64_t(v.u) : int64_t(v.u)) : int64_t(0));

    int32_t f32 = int32_t(frac);
    int32_t a0 = f32 & i7;
    int32_t a1 = (f32 << 1) & (i7 << 8);
    int32_t a2 = (f32 << 2) & (i7 << 16);
    int32_t a3 = (f32 << 3) & (i7 << 24);
    
    int32_t lo = a0 | a1 | a2 | a3;
    f32 = int32_t(frac >> 28);
    a0 = f32 & i7;
    a1 = (f32 << 1) & (i7 << 8);
    a2 = (f32 << 2) & (i7 << 16);
    a3 = (f32 << 3) & (i7 << 24);
    code = make_int3(lo, a0 | a1 | a2 | a3, int32_t(frac >> 56) & i8);
  }

  __host__ __device__ __forceinline__ void encode_float_exp7_5xi8(float value, int32_t& e, int2& code) {
    union { float d; uint32_t u; } v {value};

    int32_t sign = int32_t(v.u >> 31);
    int32_t exp = (int32_t(v.u >> 23) & i8) - 150, rem;
    fast_div7_i32x(exp, e, rem);

    v.u = ((v.u & i23) | (i23 + 1)) << rem;
    int32_t frac = (-150 < exp ? (sign ? -int32_t(v.u) : int32_t(v.u)) : int32_t(0));

    int32_t a0 = frac & i7;
    int32_t a1 = (frac << 1) & (i7 << 8);
    int32_t a2 = (frac << 2) & (i7 << 16);
    int32_t a3 = (frac << 3) & (i7 << 24);
    code = make_int2(a0 | a1 | a2 | a3, (frac >> 28) & i8);
  }

  __host__ __device__ __forceinline__ double decode_scaled_int4_double(int4 code) {
    uint4 code_lo = make_uint4(uint32_t(code.x) & i28, uint32_t(code.y) & i21, uint32_t(code.z) & i14, uint32_t(code.w) & i7);
    uint32_t lo = code_lo.x + (code_lo.y << 7) + (code_lo.z << 14) + (code_lo.w << 21);
    int32_t hi = (code.x >> 28) + (code.y >> 21) + (code.z >> 14) + (code.w >> 7);
    return (double)(((int64_t)hi << 28) + lo);
  }

  __host__ __device__ __forceinline__ float2 decode_scaled_int3_float2(int3 code) {
    uint3 code_lo = make_uint3(uint32_t(code.x) & i23, uint32_t(code.y) & i16, uint32_t(code.z) & i9);
    uint32_t lo = code_lo.x + (code_lo.y << 7) + (code_lo.z << 14);
    int32_t hi = (lo >> 23) + (code.x >> 23) + (code.y >> 16) + (code.z >> 9);
    return make_float2((float)((int64_t)hi << 23), (float)(lo & i23));
  }

};
