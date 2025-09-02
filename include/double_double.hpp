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

  __host__ __device__ __forceinline__ void fadd2_err(double2 a, double2 b, double2& sum, double2& err) {
    sum = make_double2(a.x + b.x, a.y + b.y);
    double2 delta = make_double2(a.x - sum.x, a.y - sum.y);
    double2 s_delta = make_double2(sum.x + delta.x, sum.y + delta.y);
    double2 b_delta = make_double2(b.x + delta.x, b.y + delta.y);
    err = make_double2((a.x - s_delta.x) + b_delta.x, (a.y - s_delta.y) + b_delta.y);
  }

  __host__ __device__ __forceinline__ double2 normalize(double2 a) {
    fadd_err(a.x, a.y, a.x, a.y);
    return a;
  }

  __host__ __device__ __forceinline__ double2 add(double2 a, double2 b) {
    fadd2_err(a, b, a, b);
    a.y += b.x + b.y;

    double s = a.x + a.y;
    double delta = a.x - s;
    return make_double2(s, a.y + delta);
  }

  __host__ __device__ __forceinline__ double2 mul(double2 a, double2 b) {
#ifndef __CUDA_ARCH__
    using std::fma;
#endif
    double s = fma(a.x, b.y, a.y * b.x);
    double2 d;
    d.x = a.x * b.x;
    d.y = s + fma(a.x, b.x, -d.x);

    s = d.x + d.y;
    double delta = d.x - s;
    return make_double2(s, d.y + delta);
  }

  __host__ __device__ __forceinline__ double2 fscalbn(double2 a, int32_t exp) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    return make_double2(scalbn(a.x, exp), scalbn(a.y, exp));
  }

  __host__ __device__ __forceinline__ double2 add_double(double2 a, double b) {
    double2 c = normalize(make_double2(a.x, b));
    c.y += a.y;

    double s = c.x + c.y;
    double delta = c.x - s;
    return make_double2(s, c.y + delta);
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

  __host__ __device__ __forceinline__ double conv_i31_f64(uint32_t i, int32_t expon) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    int32_t sign_i = int32_t(i) | (int32_t(i << 1) & (1 << 31));
    return scalbn(double(sign_i), expon);
  }

  __host__ __device__ __forceinline__ double conv_u31_f64(uint32_t i, int32_t expon) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    return scalbn(double(i), expon);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ double conv_a31_f64(uint32_t const (&a)[ORDER], int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 6, "Integer 32 accumulation order must be in [1,6]");
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    double res = conv_i31_f64(a[ORDER - 1], expon);
    if constexpr(5 < ORDER) res = scalbn(res, 31) + conv_u31_f64(a[4], expon);
    if constexpr(4 < ORDER) res = scalbn(res, 31) + conv_u31_f64(a[3], expon);
    if constexpr(3 < ORDER) res = scalbn(res, 31) + conv_u31_f64(a[2], expon);
    if constexpr(2 < ORDER) res = scalbn(res, 31) + conv_u31_f64(a[1], expon);
    if constexpr(1 < ORDER) res = scalbn(res, 31) + conv_u31_f64(a[0], expon);
    return res;
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ double2 conv_a31_dd(uint32_t const (&a)[ORDER], int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 6, "Integer 32 accumulation order must be in [1,6]");

    double2 res = make_double2(conv_i31_f64(a[ORDER - 1], expon), 0.);
    if constexpr(5 < ORDER) res = add_double(fscalbn(res, 31), conv_u31_f64(a[4], expon));
    if constexpr(4 < ORDER) res = add_double(fscalbn(res, 31), conv_u31_f64(a[3], expon));
    if constexpr(3 < ORDER) res = add_double(fscalbn(res, 31), conv_u31_f64(a[2], expon));
    if constexpr(2 < ORDER) res = add_double(fscalbn(res, 31), conv_u31_f64(a[1], expon));
    if constexpr(1 < ORDER) res = add_double(fscalbn(res, 31), conv_u31_f64(a[0], expon));
    return res;
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

