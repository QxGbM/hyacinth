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

  template<int32_t i>
  __host__ __device__ __forceinline__ uint64_t u64_selector(int32_t x, uint64_t x0, uint64_t x1, uint64_t x2) {
    constexpr int32_t i1 = i - 1, i2 = i - 2;
    uint64_t sel0 = x0 & -(uint64_t)(x == i);
    uint64_t sel1 = x1 & -(uint64_t)(x == i1);
    uint64_t sel2 = x2 & -(uint64_t)(x <= i2);
    return sel0 | sel1 | sel2;
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[ORDER], int64_t i, uint32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    int32_t quo = int32_t(uint32_t(63) <= expon) + int32_t(uint32_t(126) <= expon) + int32_t(uint32_t(189) <= expon);
    uint32_t rem = (expon - uint32_t(quo * 63)) & uint32_t(63);
    uint64_t q0 = (uint64_t(i) << rem) & i63, q1 = uint64_t(i >> (uint32_t(63) - rem)) & i63, q2 = -(uint64_t(i) >> 63) & i63;
    a[0] += u64_selector<0>(quo, q0, q1, q2);
    if constexpr(1 < ORDER) a[1] += u64_selector<1>(quo, q0, q1, q2) + (a[0] >> 63);
    if constexpr(2 < ORDER) a[2] += u64_selector<2>(quo, q0, q1, q2) + (a[1] >> 63);

    if constexpr(1 < ORDER) a[0] = a[0] & i63;
    if constexpr(2 < ORDER) a[1] = a[1] & i63;
    a[ORDER - 1] = ((a[ORDER - 1] << 1) & u63) | (a[ORDER - 1] & i63);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ void negate_shifted(uint64_t (&a)[ORDER]) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    a[0] += i63;
    if constexpr(1 < ORDER) a[1] += i63 + (a[0] >> 63);
    if constexpr(2 < ORDER) a[2] += i63 + (a[1] >> 63);
    
    if constexpr(1 < ORDER) a[0] = (~a[0]) & i63;
    if constexpr(2 < ORDER) a[1] = (~a[1]) & i63;
    a[ORDER - 1] = ~(((a[ORDER - 1] << 1) & u63) | (a[ORDER - 1] & i63));
  }
  
  __host__ __device__ __forceinline__ int64_t round_f64(double x, uint64_t r, int32_t expon, int32_t& e) {
    if (x == 0.) { e = 0; return int64_t(0); }
#ifndef __CUDA_ARCH__
    using std::max, std::signbit;
    uint64_t p; std::memcpy(&p, &x, sizeof(uint64_t));
#else
    uint64_t p = __double_as_longlong(x);
#endif
    int32_t sign = signbit(x), exp_bias = (int32_t(p >> 52) & 0x7ff) - 1021;
    uint64_t sn = -uint64_t(-1021 < exp_bias);
    int64_t q = int64_t((sn & 0x4000000000000000llu) | ((p << 10) & 0x3ffffffffffffc00llu));

    e = max((expon += exp_bias) - 64, 0);
    expon -= e;
    int32_t quo = int32_t(64 <= expon) - int32_t(expon < 0) - int32_t(expon < -64);
    int32_t rem = (expon - (quo << 6)) & 63, rem2 = (64 - rem) & 63;
    q = sign ? -q : q;

    uint64_t q_sign = -uint64_t(sign), q_lo = uint64_t(q) << rem, q_hi = rem ? uint64_t(q >> rem2) : q_sign;
    uint64_t lo = u64_selector<0>(quo, q_lo, q_hi, q_sign);
    int64_t hi = int64_t(u64_selector<1>(quo, q_lo, q_hi, q_sign));
    return hi + int64_t(r < lo);
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

  template<uint32_t MO, uint32_t R32, uint32_t R63>
  __host__ __device__ __forceinline__ uint32_t conv_u32i8_modular(uint32_t lo, uint32_t mi, uint32_t hi) {
    constexpr uint32_t m[4]{ uint8_t(MO) ? uint32_t(uint8_t(MO)) : uint32_t(256), uint8_t(MO >> 8), uint8_t(MO >> 16), uint8_t(MO >> 24) };
    constexpr uint32_t r32[4]{ uint8_t(R32), uint8_t(R32 >> 8), uint8_t(R32 >> 16), uint8_t(R32 >> 24) };
    constexpr uint32_t r63[4]{ uint8_t(R63), uint8_t(R63 >> 8), uint8_t(R63 >> 16), uint8_t(R63 >> 24) };
    uint32_t r[4];

    r[0] = barrett_reduc<m[0]>(barrett_reduc<m[0]>(lo) + barrett_reduc<m[0]>(mi) * r32[0] + barrett_reduc<m[0]>(hi) * r63[0]);
    r[1] = barrett_reduc<m[1]>(barrett_reduc<m[1]>(lo) + barrett_reduc<m[1]>(mi) * r32[1] + barrett_reduc<m[1]>(hi) * r63[1]);
    r[2] = barrett_reduc<m[2]>(barrett_reduc<m[2]>(lo) + barrett_reduc<m[2]>(mi) * r32[2] + barrett_reduc<m[2]>(hi) * r63[2]);
    r[3] = barrett_reduc<m[3]>(barrett_reduc<m[3]>(lo) + barrett_reduc<m[3]>(mi) * r32[3] + barrett_reduc<m[3]>(hi) * r63[3]);

    r[0] = uint8_t(r[0] + ((-uint32_t(uint32_t(127) < r[0])) & (-m[0])));
    r[1] = uint8_t(r[1] + ((-uint32_t(uint32_t(127) < r[1])) & (-m[1])));
    r[2] = uint8_t(r[2] + ((-uint32_t(uint32_t(127) < r[2])) & (-m[2])));
    r[3] = uint8_t(r[3] + ((-uint32_t(uint32_t(127) < r[3])) & (-m[3])));
    return r[0] | (r[1] << 8) | (r[2] << 16) | (r[3] << 24);
  }

  template<uint32_t MO, uint32_t R32, uint32_t MINV>
  __host__ __device__ __forceinline__ void crt_recover(int32_t (&r)[4]) {
    constexpr uint32_t m[4]{ uint8_t(MO) ? uint32_t(uint8_t(MO)) : uint32_t(256), uint8_t(MO >> 8), uint8_t(MO >> 16), uint8_t(MO >> 24) };
    constexpr uint32_t r32[4]{ m[0] - uint8_t(R32), m[1] - uint8_t(R32 >> 8), m[2] - uint8_t(R32 >> 16), m[3] - uint8_t(R32 >> 24) };
    constexpr uint32_t i[4]{ uint8_t(MINV), uint8_t(MINV >> 8), uint8_t(MINV >> 16), uint8_t(MINV >> 24) };

    uint32_t rem[4];
    rem[0] = barrett_reduc<m[0]>(uint32_t(r[0])) + ((-uint32_t(r[0] < 0)) & r32[0]);
    rem[1] = barrett_reduc<m[1]>(uint32_t(r[1])) + ((-uint32_t(r[1] < 0)) & r32[1]);
    rem[2] = barrett_reduc<m[2]>(uint32_t(r[2])) + ((-uint32_t(r[2] < 0)) & r32[2]);
    rem[3] = barrett_reduc<m[3]>(uint32_t(r[3])) + ((-uint32_t(r[3] < 0)) & r32[3]);

    r[0] = int32_t(barrett_reduc<m[0]>(rem[0] * i[0]));
    r[1] = int32_t(barrett_reduc<m[1]>(rem[1] * i[1]));
    r[2] = int32_t(barrett_reduc<m[2]>(rem[2] * i[2]));
    r[3] = int32_t(barrett_reduc<m[3]>(rem[3] * i[3]));

    r[0] += int32_t((-uint32_t(int32_t(m[0]) < r[0])) & (-m[0]));
    r[1] += int32_t((-uint32_t(int32_t(m[1]) < r[1])) & (-m[1]));
    r[2] += int32_t((-uint32_t(int32_t(m[2]) < r[2])) & (-m[2]));
    r[3] += int32_t((-uint32_t(int32_t(m[3]) < r[3])) & (-m[3]));
  }

};
