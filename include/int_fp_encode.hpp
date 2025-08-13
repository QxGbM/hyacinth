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
    quo = int32_t((int64_t(x) * int64_t(sign_m_num)) >> 32);
#endif
    rem = x - quo * div;
  }

  template <uint32_t BASE>
  __host__ __device__ __forceinline__ uint32_t pack_4x_int(uint32_t a, int32_t sign) {
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
  __host__ __device__ __forceinline__ void encode_double(double value, int32_t& e, uint32_t (&code)[4]) {
    static_assert(4 <= BASE && BASE <= 7, "Integer quantization base need to be in [2^4, 2^7].");
    constexpr uint32_t i11 = (uint32_t(1) << 11) - 1;
    constexpr uint64_t i52 = (uint64_t(1) << 52) - 1;

    constexpr uint32_t rsft_1x = 4 * BASE;
    constexpr uint32_t rsft_2x = 8 * BASE;
    constexpr uint32_t rsft_3x = 12 * BASE;

    union { double d; int64_t u; } v {value};

    int32_t sign = int32_t(v.u >> 63);
    int32_t exp = (int32_t(v.u >> 52) & i11) - 1075, rem; // bias1023 + frac52
    fast_division_i32<BASE>(exp, e, rem);

    int64_t impl_one = -int64_t(-1075 < exp) & (i52 + 1);
    uint64_t frac = ((v.u & i52) | impl_one) << rem;

    code[0] = pack_4x_int<BASE>(uint32_t(frac), sign);
    code[1] = pack_4x_int<BASE>(uint32_t(frac >> rsft_1x), sign);
    code[2] = pack_4x_int<BASE>(uint32_t(frac >> rsft_2x), sign);
    if constexpr (rsft_3x < 64)
      code[3] = pack_4x_int<BASE>(uint32_t(frac >> rsft_3x), sign);
    else
      code[3] = 0;
  }

  template <uint32_t BASE>
  __host__ __device__ __forceinline__ void encode_float(float value, int32_t& e, uint32_t (&code)[2]) {
    static_assert(4 <= BASE && BASE <= 7, "Integer quantization base need to be in [2^4, 2^7].");
    constexpr uint32_t i8 = (uint32_t(1) << 8) - 1;
    constexpr uint32_t i23 = (uint32_t(1) << 23) - 1;
    constexpr uint32_t rsft_1x = 4 * BASE;

    union { float d; int32_t u; } v {value};

    int32_t sign = v.u >> 31;
    int32_t exp = ((v.u >> 23) & i8) - 150, rem; // bias127 + frac23
    fast_division_i32<BASE>(exp, e, rem);

    int32_t impl_one = -int32_t(-150 < exp) & (i23 + 1);
    uint32_t frac = ((v.u & i23) | impl_one) << rem;

    code[0] = pack_4x_int<BASE>(frac, sign);
    code[1] = pack_4x_int<BASE>(frac >> rsft_1x, sign);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void align_expon(uint32_t (&a)[ORDER], int32_t exp_diff) {
    uint32_t rsft = 32 - ((exp_diff & 3) << 3);
    uint32_t psft = uint32_t(((exp_diff & 3) - exp_diff) >> 2);
    uint32_t b[ORDER + 2];

    auto joint_a = [](uint32_t hi, uint32_t lo, uint32_t rsft)
      { return uint32_t(((uint64_t(hi) << 32) | uint64_t(lo)) >> rsft); };
    
    b[0] = a[0] << (32 - rsft);
    if constexpr(1 < ORDER) b[1] = joint_a(a[1], a[0], rsft);
    if constexpr(2 < ORDER) b[2] = joint_a(a[2], a[1], rsft);
    if constexpr(3 < ORDER) b[3] = joint_a(a[3], a[2], rsft);
    b[ORDER] = a[ORDER - 1] >> rsft;
    b[ORDER + 1] = 0;

    auto min_u = [](uint32_t x, uint32_t hi) { return x < hi ? x : hi; };
    a[0] = b[min_u(ORDER + 1, psft)];
    if constexpr(1 < ORDER) a[1] = b[min_u(ORDER + 1, psft + 1)];
    if constexpr(2 < ORDER) a[2] = b[min_u(ORDER + 1, psft + 2)];
    if constexpr(3 < ORDER) a[3] = b[min_u(ORDER + 1, psft + 3)];
  }

  template <uint32_t BASE, uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint32_t (&a)[ORDER], int32_t i, uint32_t expon) {
    static_assert(4 <= BASE && BASE <= 7, "BASE must be in [4,7]");
    static_assert(1 <= ORDER && ORDER <= 5, "ORDER must be in [1,5]");

    constexpr uint32_t base_4x = 4 * BASE;
    constexpr uint32_t iBASE = (uint32_t(1) << base_4x) - 1;
    constexpr int32_t small_base = (BASE <= 5);
    constexpr uint32_t len_b = 5 + small_base;

    uint32_t sign = uint32_t(i >> 31) & iBASE;
    int64_t frac = int64_t(i) << uint32_t(BASE * (expon & 3));
    int32_t expon_b = 1 - int32_t(expon >> 2);

    uint32_t b[len_b];
    b[0] = 0;
    b[1] = uint32_t(frac) & iBASE;
    b[2] = uint32_t(frac >> base_4x) & iBASE;
    b[3] = uint32_t(frac >> (2 * base_4x)) & iBASE;

    if constexpr(small_base) {
      b[4] = uint32_t(frac >> (3 * base_4x)) & iBASE;
      b[5] = sign;
    }
    else
      b[4] = sign;
    
    auto clamp = [](int32_t x, int32_t lo, int32_t hi) { return x < lo ? lo : (x < hi ? x : hi); };
    a[0] += b[clamp(expon_b, 0, len_b - 1)];
    if constexpr(1 < ORDER) a[1] += b[clamp(expon_b + 1, 0, len_b - 1)] + (a[0] >> base_4x);
    if constexpr(2 < ORDER) a[2] += b[clamp(expon_b + 2, 0, len_b - 1)] + (a[1] >> base_4x);
    if constexpr(3 < ORDER) a[3] += b[clamp(expon_b + 3, 0, len_b - 1)] + (a[2] >> base_4x);
    if constexpr(4 < ORDER) a[4] += b[clamp(expon_b + 4, 0, len_b - 1)] + (a[3] >> base_4x);
    
    a[0] = a[0] & iBASE;
    if constexpr(1 < ORDER) a[1] = a[1] & iBASE;
    if constexpr(2 < ORDER) a[2] = a[2] & iBASE;
    if constexpr(3 < ORDER) a[3] = a[3] & iBASE;
    if constexpr(4 < ORDER) a[4] = a[4] & iBASE;
  }

};
