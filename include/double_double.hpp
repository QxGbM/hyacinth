#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(32) complex_double2 {
  double2 real;
  double2 imag;
};

namespace device::dd {

  __host__ __device__ __forceinline__ complex_double2 make_complex_double2(double2 real, double2 imag) {
    return complex_double2({ real, imag });
  }

  __host__ __device__ __forceinline__ double2 negate(double2 a) {
    return make_double2(-a.x, -a.y);
  }

  __host__ __device__ __forceinline__ void fadd_err(double a, double b, double& sum, double& err) {
    sum = a + b;
    double delta = a - sum;
    double s_delta = sum + delta;
    double b_delta = b + delta;
    err = (a - s_delta) + b_delta;
  }

  __host__ __device__ __forceinline__ double2 normalize(double2 a) {
    fadd_err(a.x, a.y, a.x, a.y);
    return a;
  }

  __host__ __device__ __forceinline__ double2 add(double2 a, double2 b) {
    fadd_err(a.x, b.x, a.x, b.x);
    a.y += b.x + b.y;
    b.x = a.x + a.y;
    b.y = a.y + (a.x - b.x);
    return b;
  }

  __host__ __device__ __forceinline__ double2 mul(double2 a, double2 b) {
#ifndef __CUDA_ARCH__
    using std::fma;
#endif
    double s = fma(a.x, b.y, a.y * b.x);
    a.y = a.x * b.x;
    s += fma(a.x, b.x, -a.y);
    b.x = a.y + s;
    b.y = s + (a.y - b.x);
    return b;
  }

  __host__ __device__ __forceinline__ double2 square(double2 a) {
#ifndef __CUDA_ARCH__
    using std::fma;
#endif
    double b = (a.x + a.x) * a.y;
    a.y = a.x * a.x;
    b += fma(a.x, a.x, -a.y);
    a.x = a.y + b;
    a.y = b + (a.y - a.x);
    return a;
  }

  __host__ __device__ __forceinline__ double2 fscalbn(double2 a, int32_t e) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    return make_double2(scalbn(a.x, e), scalbn(a.y, e));
  }

  __host__ __device__ __forceinline__ double2 add_double(double2 a, double b) {
    fadd_err(a.x, b, a.x, b);
    a.y += b;
    double s = a.x + a.y;
    b = a.x - s;
    return make_double2(s, a.y + b);
  }

  __host__ __device__ __forceinline__ double2 frsqrt(double2 a) {
    double rsq; int32_t p;
#ifndef __CUDA_ARCH__
    using std::frexp;
    rsq = 1. / std::sqrt(a.x);
#else
    rsq = rsqrt(a.x);
#endif
    double2 x = make_double2(frexp(rsq, &p), 0.);
    a = fscalbn(negate(a), (p << 1) - 1);

    x = mul(x, add_double(mul(x, mul(a, x)), 1.5));
    x = mul(x, add_double(mul(x, mul(a, x)), 1.5));
    return fscalbn(x, p);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ double conv_a63_f64(uint64_t const (&a)[ORDER], uint32_t e) {
    static_assert(1 <= ORDER && ORDER <= 4, "Integer 64 accumulation order must be in [1,4]");
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    constexpr uint32_t u31 = uint32_t(1) << 31, i31 = u31 - 1;
    constexpr uint64_t u63 = uint64_t(1) << 63;
    double res = double(int64_t(a[ORDER - 1] | ((a[ORDER - 1] << 1) & u63)));
    if constexpr(3 < ORDER) res = scalbn(res, 63) + double(a[2]);
    if constexpr(2 < ORDER) res = scalbn(res, 63) + double(a[1]);
    if constexpr(1 < ORDER) res = scalbn(res, 63) + double(a[0]);
    return scalbn((e & u31) ? -res : res, int32_t(((e << 1) & u31) | (e & i31)));
  }

  __host__ __device__ __forceinline__ double2 conv_i64_dd(int64_t i) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    constexpr uint64_t i53 = (uint64_t(1) << 53) - uint64_t(1);
    double x = double(i >> 53), y = double(uint64_t(i) & i53);
    return normalize(make_double2(scalbn(x, 53), y));
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ double2 conv_a63_dd(uint64_t const (&a)[ORDER], uint32_t e) {
    static_assert(1 <= ORDER && ORDER <= 4, "Integer 64 accumulation order must be in [1,4]");

    constexpr uint32_t u31 = uint32_t(1) << 31, i31 = u31 - 1;
    constexpr uint64_t u63 = uint64_t(1) << 63;
    double2 res = conv_i64_dd(a[ORDER - 1] | ((a[ORDER - 1] << 1) & u63));
    if constexpr(3 < ORDER) res = add(fscalbn(res, 63), conv_i64_dd(a[2]));
    if constexpr(2 < ORDER) res = add(fscalbn(res, 63), conv_i64_dd(a[1]));
    if constexpr(1 < ORDER) res = add(fscalbn(res, 63), conv_i64_dd(a[0]));
    return fscalbn((e & u31) ? negate(res) : res, int32_t(((e << 1) & u31) | (e & i31)));
  }

  __host__ __device__ __forceinline__ double dd2double(double2 a) {
    return a.x + a.y;
  }

  __host__ __device__ __forceinline__ double2 double2dd(double a) {
    return make_double2(a, 0.);
  }

  __host__ __device__ __forceinline__ float4 dd2qf(double2 a) {
    float a0 = float(a.x); a.x = a.x - double(a0);
    float a1 = float(a.x); a.y = a.y + (a.x - double(a1));
    float a2 = float(a.y);
    return make_float4(a0, a1, a2, a.y - double(a2));
  }
  
  __host__ __device__ __forceinline__ double2 qf2dd(float4 a) {
    double2 c = make_double2(a.x, a.y);
    double2 d = make_double2(a.z, a.w);
    double2 s0 = make_double2(c.x + c.y, d.x + d.y);
    double delta0 = c.y + (c.x - s0.x);

    double s1 = s0.x + s0.y;
    double delta1 = s0.y + (s0.x - s1);
    return make_double2(s1, delta0 + delta1);
  }

};

