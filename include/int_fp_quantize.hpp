#pragma once

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

namespace device::int8 {

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

  template <uint32_t BASE, uint32_t ORDER>
  __host__ __device__ __forceinline__ void quantize_double_align(double value, int32_t expon, uint32_t (&code)[ORDER]) {
    static_assert(4 <= BASE && BASE <= 7, "Integer quantization base need to be in [2^4,2^7].");
    static_assert(1 <= ORDER && ORDER <= 4, "Integer quantization order need to be in [1,4] for FP64.");
    constexpr int32_t BASE4x = 4 * BASE;

#ifndef __CUDA_ARCH__
    using std::signbit, std::fabs, std::scalbn, std::floor;
#endif
    uint32_t sign = signbit(value);
    value = scalbn(fabs(value), -expon);

    if constexpr(1 <= ORDER && ORDER <= 2) {
      uint64_t ir_hi = uint64_t(value);
      code[0] = pack_4x_int<BASE>(uint32_t(ir_hi), sign);
      if constexpr(2 == ORDER)
        code[1] = pack_4x_int<BASE>(uint32_t(ir_hi >> BASE4x), sign);
    }
    else {
      double fr_hi = floor(scalbn(value, -2 * BASE4x));
      double fr_lo = value - scalbn(fr_hi, 2 * BASE4x);
      uint64_t ir_hi = uint64_t(fr_hi);
      uint64_t ir_lo = uint64_t(fr_lo);

      code[0] = pack_4x_int<BASE>(uint32_t(ir_lo), sign);
      code[1] = pack_4x_int<BASE>(uint32_t(ir_lo >> BASE4x), sign);
      code[2] = pack_4x_int<BASE>(uint32_t(ir_hi), sign);
      if constexpr(4 == ORDER)
        code[3] = pack_4x_int<BASE>(uint32_t(ir_hi >> BASE4x), sign);
    }
  }

  template <uint32_t BASE, uint32_t ORDER>
  __host__ __device__ __forceinline__ void quantize_float_align(float value, int32_t expon, uint32_t (&code)[ORDER]) {
    static_assert(4 <= BASE && BASE <= 7, "Integer quantization base need to be in [2^4,2^7].");
    static_assert(1 <= ORDER && ORDER <= 2, "Integer quantization order need to be in [1,2] for FP32.");
    constexpr int32_t BASE4x = 4 * BASE;

#ifdef __CUDA_ARCH__
    uint32_t sign = uint32_t(signbit(value));
    value = scalbnf(fabsf(value), -expon);
#else
    uint32_t sign = uint32_t(std::signbit(value));
    value = std::scalbnf(std::fabs(value), -expon);
#endif

    if constexpr(ORDER == 1)
      code[0] = pack_4x_int<BASE>(uint32_t(value), sign);
    else {
#ifdef __CUDA_ARCH__
      float fr_hi = floorf(scalbnf(value, -BASE4x));
      float fr_lo = value - scalbnf(fr_hi, BASE4x);
#else
      float fr_hi = std::floor(std::scalbnf(value, -BASE4x));
      float fr_lo = value - std::scalbnf(fr_hi, BASE4x);
#endif
      code[0] = pack_4x_int<BASE>(uint32_t(fr_lo), sign);
      code[1] = pack_4x_int<BASE>(uint32_t(fr_hi), sign);
    }
  }

  template<int32_t i>
  __host__ __device__ __forceinline__ uint64_t u64_selector(int32_t x, const uint64_t (&b)[3]) {
    constexpr int32_t i1 = i - 1, i2 = i - 2;
    uint64_t sel0 = b[0] & -(uint64_t)(i == x);
    uint64_t sel1 = b[1] & -(uint64_t)(i1 == x);
    uint64_t sel2 = b[2] & -(uint64_t)(x <= i2);
    return sel0 | sel1 | sel2;
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[ORDER], int64_t i, uint32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 3, "64-bit integer accumulation order must be in [1,3]");

    constexpr uint64_t i63 = ~(uint64_t(1) << 63);
    constexpr uint32_t m_num = uint32_t((uint64_t(1) << 32) / uint64_t(63));
#ifdef __CUDA_ARCH__
    uint32_t quo = __umulhi(expon, m_num);
#else
    uint32_t quo = (uint64_t(expon) * uint64_t(m_num)) >> 32;
#endif
    uint32_t rem = expon - quo * uint32_t(63);
    uint64_t b[3]{ (uint64_t(i) << rem) & i63, uint64_t(i >> (63 - rem)) & i63, -(uint64_t(i) >> 63) & i63 };
    a[0] += u64_selector<0>(quo, b);
    if constexpr(1 < ORDER) a[1] += u64_selector<1>(quo, b) + (a[0] >> 63);
    if constexpr(2 < ORDER) a[2] += u64_selector<2>(quo, b) + (a[1] >> 63);
    
    a[0] = a[0] & i63;
    if constexpr(1 < ORDER) a[1] = a[1] & i63;
    if constexpr(2 < ORDER) a[2] = a[2] & i63;
  }

  __host__ __device__ __forceinline__ void quant_bounds(double xmin, double xmax, uint64_t umax, uint32_t& scale, int64_t& z) {
#ifndef __CUDA_ARCH__
    using std::fabs, std::fmax, std::nextafter, std::fma, std::scalbn, std::frexp;
#endif
    uint32_t sgn = fabs(xmax) < fabs(xmin);
    if (sgn) // swap and negate
    { double t = -xmin; xmin = -xmax; xmax = t; }

    double umax_d = double(umax);
    if (0. <= xmin) { // ulp checks, clamp to avoid subnormal ulp
      double ulp = fmax(nextafter(xmin, xmax) - xmin, DBL_MIN);
      xmax = fmax(xmax, fma(umax_d, ulp, xmin));
    }

    // accepting clamps for upto epi * umax
    double diff = scalbn(xmax - xmin, 1);
    int32_t exp; frexp(umax_d / diff, &exp);
    scale = (sgn << 31) | (exp & 0x7fffffff);
    z = int64_t(scalbn(xmin, exp));
  }

};
