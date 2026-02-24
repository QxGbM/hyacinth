#pragma once

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cuda_runtime.h>

namespace device::int8 {

  const uint64_t i63 = 0x7fffffffffffffffllu;
  const uint64_t u63 = 0x8000000000000000llu;
  const uint32_t i31 = 0x7fffffffu;
  const uint32_t u31 = 0x80000000u;

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[ORDER], int64_t i, uint32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    constexpr uint32_t bits = uint32_t(63);
    int32_t p0 = int32_t(expon < bits);
    int32_t p1 = int32_t((expon - uint32_t(63)) < bits);
    int32_t p2 = int32_t((expon - uint32_t(126)) < bits);
    uint32_t rem = (expon - uint32_t(p1 * 63 + p2 * 126)) & bits;

    uint64_t q0 = (uint64_t(i) << rem) & i63, m0 = -uint64_t(p0);
    a[0] += q0 & m0;

    if constexpr(1 < ORDER) {
      uint64_t q1 = uint64_t(i >> (bits - rem)) & i63, m1 = -uint64_t(p1);
      a[1] += ((q1 & m0) | (q0 & m1)) + (a[0] >> bits);
      a[0] = a[0] & i63;

      if constexpr(2 < ORDER) {
        uint64_t q2 = (-(uint64_t(i) >> bits)) & i63, m2 = -uint64_t(p2);
        a[2] += ((q2 & m0) | (q1 & m1) | (q0 & m2)) + (a[1] >> bits);
        a[1] = a[1] & i63;
      }
    }

    a[ORDER - 1] = ((a[ORDER - 1] << 1) & u63) | (a[ORDER - 1] & i63);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void negate_shifted(uint64_t (&a)[ORDER]) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    a[0] += i63;
    if constexpr(1 < ORDER) { a[1] += i63 + (a[0] >> 63); a[0] = (~a[0]) & i63; }
    if constexpr(2 < ORDER) { a[2] += i63 + (a[1] >> 63); a[1] = (~a[1]) & i63; }
    a[ORDER - 1] = ~(((a[ORDER - 1] << 1) & u63) | (a[ORDER - 1] & i63));
  }
  
  __host__ __device__ __forceinline__ int64_t round_f64(double x, int32_t expon, int32_t& e) {
#ifndef __CUDA_ARCH__
    using std::scalbn, std::llrint;
    e = std::max(std::ilogb(x) + expon - 62, 0);
#else
    e = __viaddmax_s32(ilogb(x), expon - 62, 0);
#endif
    return llrint(scalbn(x, expon - e));
  }

  __host__ __device__ __forceinline__ uint32_t conv_u8i8(uint32_t code, uint32_t& carry) {
    uint8_t* b = (uint8_t*)&code;
    uint32_t a = uint32_t(carry) + uint32_t(b[0]);
    b[0] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint32_t(b[1]);
    b[1] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint32_t(b[2]);
    b[2] = uint8_t(a); a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint32_t(b[3]);
    b[3] = uint8_t(a); carry = uint32_t((a >> 8) + ((a >> 7) & uint32_t(1)));
    return code;
  }

  template<uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t barrett_reduc(uint32_t x) {
    if constexpr(DIV) {
      constexpr uint32_t m_num = uint32_t((0x8000000000llu / uint64_t(DIV)));
#ifdef __CUDA_ARCH__
      return x - (__umulhi(x, m_num) >> 7) * DIV;
#else
      return x - uint32_t((uint64_t(x) * uint64_t(m_num)) >> 39) * DIV;
#endif
    }
    return uint32_t(0);
  }

  template<uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t u8u21_reduc(uint32_t x) {
    if constexpr(DIV) {
      constexpr uint32_t i20 = uint32_t(0xfffff), m20 = uint32_t(0x100000) % DIV;
      return m20 * (x >> 20) + (x & i20);
    }
    return uint32_t(0);
  }

  template<uint32_t MO, uint32_t R32, uint32_t R63>
  __host__ __device__ __forceinline__ uint32_t conv_u32i8_modular(uint32_t lo, uint32_t mi, uint32_t hi) {
    constexpr uint32_t m[4]{ uint8_t(MO) ? uint32_t(uint8_t(MO)) : uint32_t(256), uint8_t(MO >> 8), uint8_t(MO >> 16), uint8_t(MO >> 24) };
    constexpr uint32_t r32[4]{ uint8_t(R32), uint8_t(R32 >> 8), uint8_t(R32 >> 16), uint8_t(R32 >> 24) };
    constexpr uint32_t r63[4]{ uint8_t(R63), uint8_t(R63 >> 8), uint8_t(R63 >> 16), uint8_t(R63 >> 24) };
    uint32_t r[4];

    r[0] = barrett_reduc<m[0]>(u8u21_reduc<m[0]>(lo) + u8u21_reduc<m[0]>(mi) * r32[0] + u8u21_reduc<m[0]>(hi) * r63[0]);
    r[1] = barrett_reduc<m[1]>(u8u21_reduc<m[1]>(lo) + u8u21_reduc<m[1]>(mi) * r32[1] + u8u21_reduc<m[1]>(hi) * r63[1]);
    r[2] = barrett_reduc<m[2]>(u8u21_reduc<m[2]>(lo) + u8u21_reduc<m[2]>(mi) * r32[2] + u8u21_reduc<m[2]>(hi) * r63[2]);
    r[3] = barrett_reduc<m[3]>(u8u21_reduc<m[3]>(lo) + u8u21_reduc<m[3]>(mi) * r32[3] + u8u21_reduc<m[3]>(hi) * r63[3]);

    r[0] = uint8_t(r[0] + ((-uint32_t(uint32_t(127) < r[0])) & (-m[0])));
    r[1] = uint8_t(r[1] + ((-uint32_t(uint32_t(127) < r[1])) & (-m[1])));
    r[2] = uint8_t(r[2] + ((-uint32_t(uint32_t(127) < r[2])) & (-m[2])));
    r[3] = uint8_t(r[3] + ((-uint32_t(uint32_t(127) < r[3])) & (-m[3])));
    return r[0] | (r[1] << 8) | (r[2] << 16) | (r[3] << 24);
  }

  template<uint32_t MO, uint32_t MINV, uint32_t R32>
  __host__ __device__ __forceinline__ void crt_recover(int32_t (&r)[4]) {
    constexpr uint32_t m[4]{ uint8_t(MO) ? uint32_t(uint8_t(MO)) : uint32_t(256), uint8_t(MO >> 8), uint8_t(MO >> 16), uint8_t(MO >> 24) };
    constexpr uint32_t i[4]{ uint8_t(MINV), uint8_t(MINV >> 8), uint8_t(MINV >> 16), uint8_t(MINV >> 24) };
    constexpr uint32_t r32[4]{ uint8_t(R32), uint8_t(R32 >> 8), uint8_t(R32 >> 16), uint8_t(R32 >> 24) };

    uint32_t rem[4];
    rem[0] = barrett_reduc<m[0]>(u8u21_reduc<m[0]>(uint32_t(r[0])) * i[0] + ((-uint32_t(r[0] < 0)) & r32[0]));
    rem[1] = barrett_reduc<m[1]>(u8u21_reduc<m[1]>(uint32_t(r[1])) * i[1] + ((-uint32_t(r[1] < 0)) & r32[1]));
    rem[2] = barrett_reduc<m[2]>(u8u21_reduc<m[2]>(uint32_t(r[2])) * i[2] + ((-uint32_t(r[2] < 0)) & r32[2]));
    rem[3] = barrett_reduc<m[3]>(u8u21_reduc<m[3]>(uint32_t(r[3])) * i[3] + ((-uint32_t(r[3] < 0)) & r32[3]));

    r[0] = int32_t(__viaddmin_u32(rem[0], -m[0], rem[0]));
    r[1] = int32_t(__viaddmin_u32(rem[1], -m[1], rem[1]));
    r[2] = int32_t(__viaddmin_u32(rem[2], -m[2], rem[2]));
    r[3] = int32_t(__viaddmin_u32(rem[3], -m[3], rem[3]));
  }

};
