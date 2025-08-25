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

  template <uint32_t div>
  __host__ __device__ __forceinline__ void fast_division_i32(int32_t x, int32_t& quo, int32_t& rem) {
    constexpr uint32_t m_num = uint32_t((uint64_t(1) << 32) / uint64_t(div));
    int32_t sign_m_num = m_num + (uint32_t(~x) >> 31); // x-:floor(2^32 / div) x+: ceil(2^32 / div)
#ifdef __CUDA_ARCH__
    quo = __mulhi(x, sign_m_num);
#else
    quo = int32_t(uint64_t(int64_t(x) * int64_t(sign_m_num)) >> 32);
#endif
    rem = x - quo * div;
  }

  template <uint32_t BASE>
  __host__ __device__ __forceinline__ uint32_t pack_4x_int(uint32_t a, uint32_t sign) {
    constexpr uint32_t iBASE = (uint32_t(1) << BASE) - 1;
    constexpr uint32_t lsft_1x = 8 - BASE;
    constexpr uint32_t lsft_2x = 16 - 2 * BASE;
    constexpr uint32_t lsft_3x = 24 - 3 * BASE;
    
    uint32_t a0 = a & iBASE;
    uint32_t a1 = (a << lsft_1x) & (iBASE << 8);
    uint32_t a2 = (a << lsft_2x) & (iBASE << 16);
    uint32_t a3 = (a << lsft_3x) & (iBASE << 24);
    a = (a0 | a1) | (a2 | a3);

#ifdef __CUDA_ARCH__
    return sign ? __vneg4(a) : a;
#else
    union { uint32_t i; int8_t bytes[4]; } v{a};
    v.bytes[0] = -v.bytes[0];
    v.bytes[1] = -v.bytes[1];
    v.bytes[2] = -v.bytes[2];
    v.bytes[3] = -v.bytes[3];
    return sign ? v.i : a;
#endif
  }

  template <uint32_t BASE>
  __host__ __device__ __forceinline__ void encode_double_align(double value, int32_t expon, uint32_t (&code)[4]) {
    static_assert(4 <= BASE && BASE <= 7, "Integer quantization base need to be in [2^4,2^7].");
    constexpr uint32_t BASE4x = 4 * BASE;

#ifdef __CUDA_ARCH__
    uint32_t sign = uint32_t(uint64_t(__double_as_longlong(value)) >> 63);
#else
    using std::fabs, std::scalbn, std::floor;
    union { double d; uint64_t u; } v {value};
    uint32_t sign = uint32_t(v.u >> 63);
#endif
    value = scalbn(fabs(value), -expon * BASE);
    double fr_hi = floor(scalbn(value, -2 * int32_t(BASE4x)));
    double fr_lo = value - scalbn(fr_hi, 2 * BASE4x);
    uint64_t ir_hi = uint64_t(fr_hi);
    uint64_t ir_lo = uint64_t(fr_lo);

    code[0] = pack_4x_int<BASE>(uint32_t(ir_lo), sign);
    code[1] = pack_4x_int<BASE>(uint32_t(ir_lo >> BASE4x), sign);
    code[2] = pack_4x_int<BASE>(uint32_t(ir_hi), sign);
    code[3] = pack_4x_int<BASE>(uint32_t(ir_hi >> BASE4x), sign);
  }

  template <uint32_t BASE>
  __host__ __device__ __forceinline__ void encode_float_align(float value, int32_t expon, uint32_t (&code)[2]) {
    static_assert(4 <= BASE && BASE <= 7, "Integer quantization base need to be in [2^4,2^7].");
    constexpr uint32_t BASE4x = 4 * BASE;

#ifdef __CUDA_ARCH__
    uint32_t sign = uint32_t(__float_as_uint(value)) >> 31;
    value = scalbnf(fabsf(value), -expon * BASE);
    float fr_hi = floorf(scalbnf(value, -int32_t(BASE4x)));
    float fr_lo = value - scalbnf(fr_hi, BASE4x);
#else
    union { float f; uint32_t u; } v {value};
    uint32_t sign = v.u >> 31;
    value = std::scalbnf(std::fabs(value), -expon * BASE);
    float fr_hi = std::floor(std::scalbnf(value, -int32_t(BASE4x)));
    float fr_lo = value - std::scalbnf(fr_hi, BASE4x);
#endif

    code[0] = pack_4x_int<BASE>(uint32_t(fr_lo), sign);
    code[1] = pack_4x_int<BASE>(uint32_t(fr_hi), sign);
  }

  __host__ __device__ __forceinline__ int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) 
  { return x < lo ? lo : (x < hi ? x : hi); };

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint32_t (&a)[ORDER], int32_t i, int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 6, "Integer accumulation order must be in [1,6]");

    constexpr uint32_t i31 = ~(uint32_t(1) << 31);
    int32_t quo, rem;
    fast_division_i32<31>(expon, quo, rem);
    uint32_t b[4]{ 0, (uint32_t(i) << rem) & i31, uint32_t(i >> (31 - rem)) & i31, -(uint32_t(i) >> 31) & i31 };
    
    a[0] += b[clamp_i32(1 - quo, 0, 3)];
    if constexpr(1 < ORDER) a[1] += b[clamp_i32(2 - quo, 0, 3)] + (a[0] >> 31);
    if constexpr(2 < ORDER) a[2] += b[clamp_i32(3 - quo, 0, 3)] + (a[1] >> 31);
    if constexpr(3 < ORDER) a[3] += b[clamp_i32(4 - quo, 0, 3)] + (a[2] >> 31);
    if constexpr(4 < ORDER) a[4] += b[clamp_i32(5 - quo, 0, 3)] + (a[3] >> 31);
    if constexpr(5 < ORDER) a[5] += b[clamp_i32(6 - quo, 0, 3)] + (a[4] >> 31);
    
    a[0] = a[0] & i31;
    if constexpr(1 < ORDER) a[1] = a[1] & i31;
    if constexpr(2 < ORDER) a[2] = a[2] & i31;
    if constexpr(3 < ORDER) a[3] = a[3] & i31;
    if constexpr(4 < ORDER) a[4] = a[4] & i31;
    if constexpr(5 < ORDER) a[5] = a[5] & i31;
  }

};
