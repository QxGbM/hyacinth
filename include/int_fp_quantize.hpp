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

  __host__ __device__ __forceinline__ void quant_bounds(double xmin, double xmax, uint32_t umax, uint64_t& s_lo, uint64_t& s_hi) {
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

    int32_t expon, z_hi = 0, cel = int32_t(frexp(diff, &expon) == 0.5);
    expon = int32_t(umax) - expon + cel; s_hi = uint64_t(0);
    quantize_f64(sgn ? -xmin : xmin, expon, z_hi, s_hi);
    s_lo = (uint64_t(z_hi) << 32) | uint64_t(sgn << 31) | uint64_t(expon & i31);
  }

  __host__ __device__ __forceinline__ void extract_scale(uint32_t scale, int32_t& sgn, int32_t& expon) {
    sgn = sgn ^ int32_t(scale >> 31); expon = expon + int32_t(((scale << 1) & u31) | (scale & i31));
  }

  __host__ __device__ __forceinline__ void conv_u8i8(uint32_t& code, uint32_t& carry) {
    uint8_t* b = (uint8_t*)&code;
    uint16_t a = uint16_t(carry) + uint16_t(b[0]);
    b[0] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint16_t(1)) + uint16_t(b[1]);
    b[1] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint16_t(1)) + uint16_t(b[2]);
    b[2] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint16_t(1)) + uint16_t(b[3]);
    b[3] = uint8_t(a); carry = uint32_t((a >> 8) + ((a >> 7) & uint16_t(1)));
  }

  __host__ __device__ __forceinline__ void quantize_double_align(double value, int32_t expon, uint32_t (&code)[3]) {
    int32_t hi = 0; uint64_t lo = uint64_t(0);
    quantize_f64(value, expon, hi, lo);
    code[0] = uint32_t(lo);
    code[1] = uint32_t(lo >> 32) | (uint32_t(hi) << 31);
    code[2] = uint32_t(hi >> 1);

    uint32_t carry = 0;
    conv_u8i8(code[0], carry);
    conv_u8i8(code[1], carry);
    conv_u8i8(code[2], carry);
  }

};
