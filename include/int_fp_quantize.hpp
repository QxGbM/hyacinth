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

#ifdef __CUDA_ARCH__
    uint32_t quo = __umulhi(expon, 0x82082082) >> 5;
#else
    uint32_t quo = uint32_t((uint64_t(expon) * 0x82082082llu) >> 37);
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

  template<uint16_t DIV>
  __host__ __device__ __forceinline__ uint32_t fast_rem_u32(uint32_t x) {
    constexpr uint32_t m_num = DIV ? uint32_t((0x800000000000llu / uint64_t(DIV))) : uint32_t(0);
#ifdef __CUDA_ARCH__
    return x - (__umulhi(x, m_num) >> 15) * uint32_t(DIV);
#else
    return x - uint32_t((uint64_t(x) * uint64_t(m_num)) >> 47) * uint32_t(DIV);
#endif
  }

  __host__ __device__ __forceinline__ uint32_t fast_rem_u32_255(uint32_t x) {
#ifdef __CUDA_ARCH__
    return x - (__umulhi(x, 0x80808080) >> 7) * uint32_t(255);
#else
    return x - uint32_t((uint64_t(x) * 0x80808080llu) >> 39) * uint32_t(255);
#endif
  }

  __host__ __device__ __forceinline__ uint32_t conv_u16i8r(uint32_t x) {
    uint32_t r255 = fast_rem_u32_255(x);
    return (x & uint32_t(255)) | ((r255 + (r255 >> 7)) << 8);
  }

  template<uint64_t MO, uint64_t R32, uint32_t ORDER>
  __host__ __device__ __forceinline__ uint64_t conv_u32i8_modular(const uint32_t (&code)[ORDER]) {
    static_assert(1 <= ORDER && ORDER <= 3, "Quantized Integer order must be in [1,3]");
    constexpr uint16_t m0 = uint16_t(MO), m1 = uint16_t(MO >> 16), m2 = uint16_t(MO >> 32), m3 = uint16_t(MO >> 48);

    uint32_t r65[4] = { code[0], code[0], code[0], code[0] };
    if constexpr(1 < ORDER) {
      constexpr uint16_t r0 = uint16_t(R32), r1 = uint16_t(R32 >> 16), r2 = uint16_t(R32 >> 32), r3 = uint16_t(R32 >> 48);
      r65[0] = fast_rem_u32<m0>(r65[0]) + fast_rem_u32<m0>(code[1]) * uint32_t(r0);
      r65[1] = fast_rem_u32<m1>(r65[1]) + fast_rem_u32<m1>(code[1]) * uint32_t(r1);
      r65[2] = fast_rem_u32<m2>(r65[2]) + fast_rem_u32<m2>(code[1]) * uint32_t(r2);
      r65[3] = fast_rem_u32<m3>(r65[3]) + fast_rem_u32<m3>(code[1]) * uint32_t(r3);

      if constexpr(2 < ORDER) {
        constexpr uint32_t R0 = (uint32_t(r0) * uint32_t(r0)) % uint32_t(m0);
        constexpr uint32_t R1 = (uint32_t(r1) * uint32_t(r1)) % uint32_t(m1);
        constexpr uint32_t R2 = (uint32_t(r2) * uint32_t(r2)) % uint32_t(m2);
        constexpr uint32_t R3 = (uint32_t(r3) * uint32_t(r3)) % uint32_t(m3);
        r65[0] = fast_rem_u32<m0>(r65[0]) + fast_rem_u32<m0>(code[2]) * R0;
        r65[1] = fast_rem_u32<m1>(r65[1]) + fast_rem_u32<m1>(code[2]) * R1;
        r65[2] = fast_rem_u32<m2>(r65[2]) + fast_rem_u32<m2>(code[2]) * R2;
        r65[3] = fast_rem_u32<m3>(r65[3]) + fast_rem_u32<m3>(code[2]) * R3;
      }
    }

    uint32_t lo = conv_u16i8r(fast_rem_u32<m0>(r65[0])) | (conv_u16i8r(fast_rem_u32<m1>(r65[1])) << 16);
    uint32_t hi = conv_u16i8r(fast_rem_u32<m2>(r65[2])) | (conv_u16i8r(fast_rem_u32<m3>(r65[3])) << 16);
    return uint64_t(lo) | (uint64_t(hi) << 32);
  }

  template<uint64_t MO, uint64_t MINV>
  __host__ __device__ __forceinline__ void crt_recover(int32_t (&r)[8]) {
    uint32_t rem[4];
    rem[0] = (uint32_t(r[1]) >> 16) + uint32_t(uint16_t(r[1])) + ((-uint32_t(r[1] < 0)) & uint32_t(254));
    rem[1] = (uint32_t(r[3]) >> 16) + uint32_t(uint16_t(r[3])) + ((-uint32_t(r[3] < 0)) & uint32_t(254));
    rem[2] = (uint32_t(r[5]) >> 16) + uint32_t(uint16_t(r[5])) + ((-uint32_t(r[5] < 0)) & uint32_t(254));
    rem[3] = (uint32_t(r[7]) >> 16) + uint32_t(uint16_t(r[7])) + ((-uint32_t(r[7] < 0)) & uint32_t(254));

    rem[0] = (uint32_t(uint8_t(-uint32_t(r[0]))) * uint32_t(255)) + (fast_rem_u32_255(rem[0]) << 8);
    rem[1] = (uint32_t(uint8_t(-uint32_t(r[2]))) * uint32_t(255)) + (fast_rem_u32_255(rem[1]) << 8);
    rem[2] = (uint32_t(uint8_t(-uint32_t(r[4]))) * uint32_t(255)) + (fast_rem_u32_255(rem[2]) << 8);
    rem[3] = (uint32_t(uint8_t(-uint32_t(r[6]))) * uint32_t(255)) + (fast_rem_u32_255(rem[3]) << 8);

    rem[0] += (-uint32_t(uint32_t(65280) < rem[0])) & uint32_t(-65280);
    rem[1] += (-uint32_t(uint32_t(65280) < rem[1])) & uint32_t(-65280);
    rem[2] += (-uint32_t(uint32_t(65280) < rem[2])) & uint32_t(-65280);
    rem[3] += (-uint32_t(uint32_t(65280) < rem[3])) & uint32_t(-65280);

    constexpr uint16_t m0 = uint16_t(MO), m1 = uint16_t(MO >> 16), m2 = uint16_t(MO >> 32), m3 = uint16_t(MO >> 48);
    constexpr uint16_t i0 = uint16_t(MINV), i1 = uint16_t(MINV >> 16), i2 = uint16_t(MINV >> 32), i3 = uint16_t(MINV >> 48);
    r[0] = int32_t(fast_rem_u32<m0>(rem[0] * uint32_t(i0))); r[1] = r[0] - int32_t(m0);
    r[2] = int32_t(fast_rem_u32<m1>(rem[1] * uint32_t(i1))); r[3] = r[2] - int32_t(m1);
    r[4] = int32_t(fast_rem_u32<m2>(rem[2] * uint32_t(i2))); r[5] = r[4] - int32_t(m2);
    r[6] = int32_t(fast_rem_u32<m3>(rem[3] * uint32_t(i3))); r[7] = r[6] - int32_t(m3);
  }

};
