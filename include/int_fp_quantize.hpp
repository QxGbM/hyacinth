#pragma once

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

namespace device::int8 {

  constexpr uint64_t i63 = 0x7fffffffffffffffllu;
  constexpr uint64_t u63 = 0x8000000000000000llu;
  constexpr uint32_t i31 = 0x7fffffffu;
  constexpr uint32_t u31 = 0x80000000u;

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
    value = scalbn(fabs(value), expon);

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

  template<int32_t i>
  __host__ __device__ __forceinline__ uint64_t u64_selector(int32_t x, const uint64_t (&b)[3]) {
    constexpr int32_t i1 = i - 1, i2 = i - 2;
    uint64_t sel0 = b[0] & -(uint64_t)(x == i);
    uint64_t sel1 = b[1] & -(uint64_t)(x == i1);
    uint64_t sel2 = b[2] & -(uint64_t)(x <= i2);
    return sel0 | sel1 | sel2;
  }

  __host__ __device__ __forceinline__ uint64_t copy_bit_i63(uint64_t i) {
    return ((i << 1) & u63) | (i & i63);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[ORDER], int64_t i, uint32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 4, "Integer 64 accumulation order must be in [1,4]");

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
    if constexpr(3 < ORDER) a[3] += u64_selector<3>(quo, b) + (a[2] >> 63);

    if constexpr(1 < ORDER) a[0] = a[0] & i63;
    if constexpr(2 < ORDER) a[1] = a[1] & i63;
    if constexpr(3 < ORDER) a[2] = a[2] & i63;
    a[ORDER - 1] = copy_bit_i63(a[ORDER - 1]);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void negate_shifted(uint64_t (&a)[ORDER]) {
    static_assert(1 <= ORDER && ORDER <= 4, "Integer 64 accumulation order must be in [1,4]");

    a[0] += i63;
    if constexpr(1 < ORDER) a[1] += i63 + (a[0] >> 63);
    if constexpr(2 < ORDER) a[2] += i63 + (a[1] >> 63);
    if constexpr(3 < ORDER) a[3] += i63 + (a[2] >> 63);
    
    if constexpr(1 < ORDER) a[0] = (~a[0]) & i63;
    if constexpr(2 < ORDER) a[1] = (~a[1]) & i63;
    if constexpr(3 < ORDER) a[2] = (~a[2]) & i63;
    a[ORDER - 1] = ~copy_bit_i63(a[ORDER - 1]);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void ima_shifted(uint64_t (&a)[ORDER], int64_t x, int64_t y, uint32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 4, "Integer 64 accumulation order must be in [1,4]");

#ifdef __CUDA_ARCH__
    int64_t prod = x * y;
    add_shifted(a, prod & i63, expon);
    add_shifted(a, ((prod & u63) >> 63) | (__mul64hi(x, y) << 1), expon + uint32_t(63));
    add_shifted(a, int64_t((uint64_t(x) == u63) && (uint64_t(y) == u63)), expon + uint32_t(126));
#else
    constexpr int32_t i24 = (int32_t(1) << 24) - 1;
    int32_t x_limbs[3]{ int32_t(x) & i24, int32_t(uint64_t(x) >> 24) & i24, int32_t(x >> 48) };
    int32_t y_hi = int32_t(y >> 32); uint32_t y_lo = uint32_t(y);

    for (int32_t i = 0; i < 3; ++i) {
      add_shifted(a, int64_t(x_limbs[i]) * int64_t(y_hi), expon + uint32_t(24 * i) + uint32_t(32));
      add_shifted(a, int64_t(x_limbs[i]) * int64_t(y_lo), expon + uint32_t(24 * i));
    }
#endif
  }

  __host__ __device__ __forceinline__ void quantize_f64(double x, int32_t expon, int32_t& hi, uint64_t& lo) {
#ifndef __CUDA_ARCH__
    using std::ilogb, std::max, std::scalbn, std::llrint;
#endif
    int32_t e = max(ilogb(x) + expon - 62, 0);
    int64_t q = llrint(scalbn(x, expon - e));
    lo += (uint64_t(q) << e) & i63;
    hi += int32_t(q >> (63 - e)) + int32_t(lo >> 63);
    lo &= i63;
  }

  __host__ __device__ __forceinline__ void quant_bounds(double xmin, double xmax, uint32_t umax, uint32_t c, uint64_t& s_lo, uint64_t& s_hi) {
#ifndef __CUDA_ARCH__
    using std::fabs, std::fmin, std::fmax, std::nextafter, std::scalbn, std::frexp, std::nearbyint;
#endif
    constexpr double inf = INFINITY;
    uint32_t sgn = (fabs(xmax) < fabs(xmin)) || (xmin == xmax && xmin < 0.);
    if (sgn)
    { double t = -xmin; xmin = -xmax; xmax = t; }

    double ulp = fmax(nextafter(xmin, inf) - xmin, DBL_MIN);
    double diff = xmax - xmin;
    double diff_pos = fmax(scalbn(ulp, int32_t(umax)), diff);
    double diff_neg = (diff == xmax) ? nextafter(diff, inf) : diff;
    diff = fmin((0. <= xmin) ? diff_pos : diff_neg, DBL_MAX);

    int32_t expon, cel = int32_t(frexp(diff, &expon) == 0.5);
    expon = int32_t(umax) - expon + cel;

    uint32_t c_lo = uint32_t(63) < c ? uint32_t(63) : c; c -= c_lo;
    uint64_t z_lo = 0x0080808080808080llu & ((uint64_t(1) << c_lo) - uint64_t(1));
    int32_t z_hi = 0x01010101 & ((1 << c) - 1);
    quantize_f64(xmin, expon, z_hi, z_lo);

    int32_t neg_z_hi = -(z_hi + int32_t(z_lo != 0));
    uint64_t neg_z_lo = -z_lo & i63;
    s_hi = sgn ? neg_z_lo : z_lo;
    s_lo = (uint64_t(sgn ? neg_z_hi : z_hi) << 32) | uint64_t(sgn << 31) | uint64_t(expon & i31);
  }

  __host__ __device__ __forceinline__ void extract_scale(uint32_t scale, int32_t& sgn, int32_t& expon) {
    sgn = sgn ^ int32_t(scale >> 31); expon = expon + int32_t(((scale << 1) & u31) | (scale & i31));
  }

};
