#pragma once

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

namespace device::int8 {

  const uint64_t i63 = 0x7fffffffffffffffllu;
  const uint64_t u63 = 0x8000000000000000llu;
  const uint32_t i31 = 0x7fffffffu;
  const uint32_t u31 = 0x80000000u;

  template<int32_t i>
  __host__ __device__ __forceinline__ uint64_t u64_selector(int32_t x, const uint64_t (&b)[3]) {
    constexpr int32_t i1 = i - 1, i2 = i - 2;
    uint64_t sel0 = b[0] & -(uint64_t)(x == i);
    uint64_t sel1 = b[1] & -(uint64_t)(x == i1);
    uint64_t sel2 = b[2] & -(uint64_t)(x <= i2);
    return sel0 | sel1 | sel2;
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
    a[ORDER - 1] = ((a[ORDER - 1] << 1) & u63) | (a[ORDER - 1] & i63);
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
    a[ORDER - 1] = ~(((a[ORDER - 1] << 1) & u63) | (a[ORDER - 1] & i63));
  }

  __host__ __device__ __forceinline__ void round_f64(double x, int32_t expon, int64_t& q, int32_t& e) {
#ifndef __CUDA_ARCH__
    using std::ilogb, std::max, std::scalbn, std::llrint;
#endif
    e = max(ilogb(x) + expon - 62, 0);
    q = llrint(scalbn(x, expon - e));
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void quantize_f64_u32limbs(double value, int32_t expon, int32_t umax, uint32_t (&code)[ORDER]) {
    static_assert(1 <= ORDER && ORDER <= 3, "Quantized Integer order must be in [1,3]");

    int64_t q; round_f64(value, expon, q, expon);
    uint64_t lo = ((-uint64_t(umax < 64)) & (uint64_t(1) << umax)) + ((uint64_t(q) << expon) & i63);
    code[0] = uint32_t(lo);

    if constexpr(1 < ORDER) {
      uint32_t hi = ((-uint32_t(63 < umax)) & (uint32_t(1) << (umax - 63))) + uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
      code[1] = (hi << 31) | (uint32_t(lo >> 32) & i31);
      if constexpr(2 < ORDER) code[2] = hi >> 1;
    }
  }

  __host__ __device__ __forceinline__ void conv_u8i8(uint32_t& code, uint32_t& carry) {
    uint8_t* b = (uint8_t*)&code;
    uint16_t a = uint16_t(carry) + uint16_t(b[0]);
    b[0] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint16_t(1)) + uint16_t(b[1]);
    b[1] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint16_t(1)) + uint16_t(b[2]);
    b[2] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint16_t(1)) + uint16_t(b[3]);
    b[3] = uint8_t(a); carry = uint32_t((a >> 8) + ((a >> 7) & uint16_t(1)));
  }

};
