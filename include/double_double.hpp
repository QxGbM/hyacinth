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

  __host__ __device__ __forceinline__ double2 renormalize(double2 a) {
    double sum = a.x + a.y, delta = a.x - sum;
    return make_double2(sum, a.y + delta);
  }

  __host__ __device__ __forceinline__ double2 add(double2 a, double2 b) {
    fadd_err(a.x, b.x, a.x, b.x);
    return renormalize(make_double2(a.x, a.y + b.x + b.y));
  }

  __host__ __device__ __forceinline__ double2 mul(double2 a, double2 b) {
#ifndef __CUDA_ARCH__
    using std::fma;
#endif
    double p = a.x * b.x;
    return renormalize(make_double2(p, fma(a.x, b.y, a.y * b.x) + fma(a.x, b.x, -p)));
  }

  __host__ __device__ __forceinline__ double2 square(double2 a) {
#ifndef __CUDA_ARCH__
    using std::fma;
#endif
    double p = a.x * a.x;
    return renormalize(make_double2(p, fma(a.x + a.x, a.y, fma(a.x, a.x, -p))));
  }

  __host__ __device__ __forceinline__ double2 fldexp(double2 a, int32_t e) {
#ifndef __CUDA_ARCH__
    using std::ldexp;
#endif
    return make_double2(ldexp(a.x, e), ldexp(a.y, e));
  }

  __host__ __device__ __forceinline__ double2 frsqrt(double2 a) {
    int32_t p;
#ifndef __CUDA_ARCH__
    using std::frexp;
    double rsq = 1. / std::sqrt(a.x);
#else
    double rsq = rsqrt(a.x);
#endif
    double2 x = make_double2(frexp(rsq, &p), 0.);
    double2 c = make_double2(1.5, 0.);
    a = fldexp(negate(a), (p << 1) - 1);

    x = mul(x, add(mul(x, mul(a, x)), c));
    x = mul(x, add(mul(x, mul(a, x)), c));
    return fldexp(x, p);
  }

  __host__ __device__ __forceinline__ double2 conv_i64_dd_m1022(uint64_t i) {
#ifndef __CUDA_ARCH__
    using std::ldexp;
#endif
    return make_double2(ldexp(double(int32_t(i >> 32)), -990), ldexp(double(uint32_t(i)), -1022));
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ double conv_a63_f64(uint64_t const (&a)[ORDER], int32_t e) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");
#ifndef __CUDA_ARCH__
    using std::ldexp;
#endif
    double res = ldexp(double(int64_t(a[ORDER - 1])), -1022);
    if constexpr(2 < ORDER) { double2 i = conv_i64_dd_m1022(a[1]); res = (ldexp(res, 63) + i.x) + i.y; }
    if constexpr(1 < ORDER) { double2 i = conv_i64_dd_m1022(a[0]); res = (ldexp(res, 63) + i.x) + i.y; }
    return ldexp(res, 1022 + e);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ double2 conv_a63_dd(uint64_t const (&a)[ORDER], int32_t e) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    double2 res = renormalize(conv_i64_dd_m1022(a[ORDER - 1]));
    if constexpr(2 < ORDER) res = add(fldexp(res, 63), renormalize(conv_i64_dd_m1022(a[1])));
    if constexpr(1 < ORDER) res = add(fldexp(res, 63), renormalize(conv_i64_dd_m1022(a[0])));
    return fldexp(res, 1022 + e);
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
    double2 a0 = renormalize(make_double2(a.x, a.y));
    double2 a1 = renormalize(make_double2(a0.x, double(a.z) + double(a.w)));
    return make_double2(a1.x, a0.y + a1.y);
  }

};

