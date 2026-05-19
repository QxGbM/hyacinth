#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace device::int8 {

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[ORDER], int64_t i, uint32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    constexpr uint64_t i63 = 0x7fffffffffffffffllu;
    constexpr uint32_t bits = uint32_t(63);
    int32_t p0 = int32_t(expon < bits);
    int32_t p1 = int32_t((expon - uint32_t(63)) < bits);
    int32_t p2 = int32_t((expon - uint32_t(126)) < bits);
    uint32_t rem = (expon - uint32_t(p1 * 63 + p2 * 126)) & bits;
    uint64_t q0 = (uint64_t(i) << rem), m0 = -uint64_t(p0);

    if constexpr(1 < ORDER) {
      a[0] += q0 & m0 & i63;
      uint64_t q1 = uint64_t(i >> (bits - rem)), m1 = -uint64_t(p1);
      if constexpr(2 < ORDER) {
        a[1] += (((q1 & m0) | (q0 & m1)) & i63) + (a[0] >> bits); a[0] &= i63; 
        uint64_t q2 = (-(uint64_t(i) >> bits)), m2 = -uint64_t(p2);
        a[2] += ((q2 & m0) | (q1 & m1) | (q0 & m2)) + (a[1] >> bits); a[1] &= i63;
      } else { a[1] += ((q1 & m0) | (q0 & m1)) + (a[0] >> bits); a[0] &= i63; }
    } else { a[0] += q0 & m0; }
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
    constexpr uint32_t i8 = uint32_t(0xff), i7 = uint32_t(0x7f), r7 = uint32_t(0x80);
    uint32_t a = carry + (code & i8), s = (a >> 7) & uint32_t(1), c = (r7 & (-s)) | (i7 & a);
    a = (a >> 8) + s + ((code >> 8) & i8); s = (a >> 7) & uint32_t(1); c |= ((r7 & (-s)) | (i7 & a)) << 8;
    a = (a >> 8) + s + ((code >> 16) & i8); s = (a >> 7) & uint32_t(1); c |= ((r7 & (-s)) | (i7 & a)) << 16;
    a = (a >> 8) + s + ((code >> 24) & i8); s = (a >> 7) & uint32_t(1); c |= ((r7 & (-s)) | (i7 & a)) << 24;
    carry = (a >> 8) + s;
    return c;
  }

  template<uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t barrett_reduc(uint32_t x) {
#ifdef __CUDA_ARCH__
    if constexpr(DIV) {
      constexpr uint32_t d = uint32_t((0x100000000llu / uint64_t(DIV))), minus_div = -DIV;
      x = x - (__umulhi(x, d) * DIV); return __viaddmin_u32(minus_div, x, x);
    } else return uint32_t(0);
#else
    if constexpr(DIV) return x % DIV; else return uint32_t(0);
#endif
  }

  template<uint32_t MUL, uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t mulx_reduc(uint32_t x) {
    if constexpr(MUL && DIV) {
      static_assert(DIV <= 4096u, "Modular needs to be smaller than 4096");
      constexpr uint32_t mul = MUL % DIV, mul_r16 = (mul << 16) % DIV;
      return (mul_r16 * (x >> 16)) + (mul * (x & uint32_t(0xffff)));
    } else return uint32_t(0);
  }

  template<uint64_t MO, uint64_t R32, uint64_t R63>
  __host__ __device__ __forceinline__ uint32_t conv_u32i8_modular(uint32_t lo, uint32_t mi, uint32_t hi) {
    constexpr uint32_t m[4]{ uint16_t(MO), uint16_t(MO >> 16), uint16_t(MO >> 32), uint16_t(MO >> 48) };
    constexpr uint32_t r32[4]{ uint16_t(R32), uint16_t(R32 >> 16), uint16_t(R32 >> 32), uint16_t(R32 >> 48) };
    constexpr uint32_t r63[4]{ uint16_t(R63), uint16_t(R63 >> 16), uint16_t(R63 >> 32), uint16_t(R63 >> 48) };
    union { uint2 u; short4 h; } x, cx;

    x.h.x = int16_t(barrett_reduc<m[0]>(mulx_reduc<uint32_t(1), m[0]>(lo) + mulx_reduc<r32[0], m[0]>(mi) + mulx_reduc<r63[0], m[0]>(hi)));
    x.h.y = int16_t(barrett_reduc<m[1]>(mulx_reduc<uint32_t(1), m[1]>(lo) + mulx_reduc<r32[1], m[1]>(mi) + mulx_reduc<r63[1], m[1]>(hi)));
    x.h.z = int16_t(barrett_reduc<m[2]>(mulx_reduc<uint32_t(1), m[2]>(lo) + mulx_reduc<r32[2], m[2]>(mi) + mulx_reduc<r63[2], m[2]>(hi)));
    x.h.w = int16_t(barrett_reduc<m[3]>(mulx_reduc<uint32_t(1), m[3]>(lo) + mulx_reduc<r32[3], m[3]>(mi) + mulx_reduc<r63[3], m[3]>(hi)));

#ifdef __CUDA_ARCH__
    constexpr uint32_t m12 = uint32_t(MO), m34 = uint32_t(MO >> 32);
    cx.u.x = __vsub2(x.u.x, m12); cx.u.y = __vsub2(x.u.y, m34);
    uint32_t mask_x = __vcmplts2(__vadd2(x.u.x, cx.u.x), uint32_t(0));
    uint32_t mask_y = __vcmplts2(__vadd2(x.u.y, cx.u.y), uint32_t(0));
    x.u.x = (mask_x & x.u.x) | ((~mask_x) & cx.u.x);
    x.u.y = (mask_y & x.u.y) | ((~mask_y) & cx.u.y);
#else
    cx.h.x = int16_t(uint32_t(x.h.x) - m[0]);
    cx.h.y = int16_t(uint32_t(x.h.y) - m[1]);
    cx.h.z = int16_t(uint32_t(x.h.z) - m[2]);
    cx.h.w = int16_t(uint32_t(x.h.w) - m[3]);

    x.h.x = (int32_t(x.h.x) + int32_t(cx.h.x) < 0) ? x.h.x : cx.h.x;
    x.h.y = (int32_t(x.h.y) + int32_t(cx.h.y) < 0) ? x.h.y : cx.h.y;
    x.h.z = (int32_t(x.h.z) + int32_t(cx.h.z) < 0) ? x.h.z : cx.h.z;
    x.h.w = (int32_t(x.h.w) + int32_t(cx.h.w) < 0) ? x.h.w : cx.h.w;
#endif
    return uint32_t(uint8_t(x.h.x)) | (uint32_t(uint8_t(x.h.y)) << 8) | (uint32_t(uint8_t(x.h.z)) << 16) | (uint32_t(uint8_t(x.h.w)) << 24);
  }

  template<uint64_t MO, uint64_t MINV, uint64_t R32>
  __host__ __device__ __forceinline__ void crt_recover(int32_t (&r)[4]) {
    constexpr uint32_t m[4]{ uint16_t(MO), uint16_t(MO >> 16), uint16_t(MO >> 32), uint16_t(MO >> 48) };
    constexpr uint32_t i[4]{ uint16_t(MINV), uint16_t(MINV >> 16), uint16_t(MINV >> 32), uint16_t(MINV >> 48) };
    constexpr uint32_t r32[4]{ uint16_t(R32), uint16_t(R32 >> 16), uint16_t(R32 >> 32), uint16_t(R32 >> 48) };

    r[0] = int32_t(barrett_reduc<m[0]>(mulx_reduc<i[0], m[0]>(uint32_t(r[0])) + ((-(uint32_t(r[0]) >> 31)) & r32[0])));
    r[1] = int32_t(barrett_reduc<m[1]>(mulx_reduc<i[1], m[1]>(uint32_t(r[1])) + ((-(uint32_t(r[1]) >> 31)) & r32[1])));
    r[2] = int32_t(barrett_reduc<m[2]>(mulx_reduc<i[2], m[2]>(uint32_t(r[2])) + ((-(uint32_t(r[2]) >> 31)) & r32[2])));
    r[3] = int32_t(barrett_reduc<m[3]>(mulx_reduc<i[3], m[3]>(uint32_t(r[3])) + ((-(uint32_t(r[3]) >> 31)) & r32[3])));
  }

};
