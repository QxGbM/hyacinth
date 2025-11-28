#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(32) complex_float4 {
  float4 real;
  float4 imag;
};

namespace device::qf {

  __host__ __device__ __forceinline__ complex_float4 make_complex_float4(float4 real, float4 imag) {
    return complex_float4({ real, imag });
  }

  __host__ __device__ __forceinline__ float4 negate(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
  }

  __host__ __device__ __forceinline__ void fadd_err(float a, float b, float& sum, float& err) {
    sum = a + b;
    float delta = a - sum;
    float s_delta = sum + delta;
    float b_delta = b + delta;
    err = (a - s_delta) + b_delta;
  }

  __host__ __device__ __forceinline__ void fadd2_err(float2 a, float2 b, float2& sum, float2& err) {
    sum = make_float2(a.x + b.x, a.y + b.y);
    float2 delta = make_float2(a.x - sum.x, a.y - sum.y);
    float2 s_delta = make_float2(sum.x + delta.x, sum.y + delta.y);
    float2 b_delta = make_float2(b.x + delta.x, b.y + delta.y);
    err = make_float2((a.x - s_delta.x) + b_delta.x, (a.y - s_delta.y) + b_delta.y);
  }

  __host__ __device__ __forceinline__ float4 normalize(float4 a) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(a.z, a.w), a0, a1);
    fadd2_err(make_float2(a0.x, a1.x), make_float2(a0.y, a1.y), a0, a1);
    fadd_err(a0.y, a1.x, a0.y, a1.x);
    fadd_err(a1.x, a1.y, a1.x, a1.y);
    return make_float4(a0.x, a0.y, a1.x, a1.y);
  }

  __host__ __device__ __forceinline__ float4 add(float4 a, float4 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1); // 1122 - 1223
    fadd_err(a.z, b.z, a.z, b.z); // 33 - 34, 4@b.z

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1); // 2233 - 2334, 4@a1.y
    fadd_err(a0.y, a1.x, a0.y, a1.x); // 33 - 34, 4@a1.x
    return normalize(make_float4(r0, a0.x, a0.y, a.w + (b.w + b.z) + (a1.y + a1.x)));
  }

  __host__ __device__ __forceinline__ void fmul2_err(float2 a, float2 b, float2& prod, float2& err) {
#ifndef __CUDA_ARCH__
    using std::fmaf;
#endif
    prod = make_float2(a.x * b.x, a.y * b.y);
    err = make_float2(fmaf(a.x, b.x, -prod.x), fmaf(a.y, b.y, -prod.y));
  }

  __host__ __device__ __forceinline__ float4 mul(float4 a, float4 b) {
#ifndef __CUDA_ARCH__
    using std::fmaf;
#endif
    float2 prod, err;
    fmul2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), prod, err);
    float c1 = prod.x;
    float c4 = fmaf(a.w, b.x, a.z * b.y) + fmaf(a.y, b.z, fmaf(a.x, b.w, err.y));
    float2 c23 = make_float2(err.x, prod.y);

    fmul2_err(make_float2(a.y, a.z), make_float2(b.x, b.x), prod, err);
    fadd2_err(c23, prod, c23, prod);
    fadd_err(prod.x, err.x, prod.x, err.x);
    fadd_err(c23.y, prod.x, c23.y, prod.x);
    c4 += (prod.x + prod.y) + (err.x + err.y);

    fmul2_err(make_float2(a.x, a.x), make_float2(b.y, b.z), prod, err);
    fadd2_err(c23, prod, c23, prod);
    fadd_err(prod.x, err.x, prod.x, err.x);
    fadd_err(c23.y, prod.x, c23.y, prod.x);
    c4 += (prod.x + prod.y) + (err.x + err.y);

    return normalize(make_float4(c1, c23.x, c23.y, c4));
  }

  __host__ __device__ __forceinline__ float4 square(float4 a) {
#ifndef __CUDA_ARCH__
    using std::fmaf;
#endif
    float2 prod, err;
    fmul2_err(make_float2(a.x, a.y), make_float2(a.x, a.y), prod, err);
    float c1 = prod.x; a.x = a.x + a.x;
    float c4 = fmaf(a.w, a.x, a.z * (a.y + a.y)) + err.y;
    float2 c23 = make_float2(err.x, prod.y);

    fmul2_err(make_float2(a.y, a.z), make_float2(a.x, a.x), prod, err);
    fadd2_err(c23, prod, c23, prod);
    fadd_err(prod.x, err.x, prod.x, err.x);
    fadd_err(c23.y, prod.x, c23.y, prod.x);
    c4 += (prod.x + prod.y) + (err.x + err.y);

    return normalize(make_float4(c1, c23.x, c23.y, c4));
  }

  __host__ __device__ __forceinline__ float4 fscalbn(float4 a, int32_t exp) {
#ifndef __CUDA_ARCH__
    using std::scalbnf;
#endif
    return make_float4(scalbnf(a.x, exp), scalbnf(a.y, exp), scalbnf(a.z, exp), scalbnf(a.w, exp));
  }

  __host__ __device__ __forceinline__ float4 add_float2(float4 a, float2 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1);

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1);
    fadd_err(a0.y, a1.x, a0.y, a1.x);
    return normalize(make_float4(r0, a0.x, a0.y, a.w + a1.y + a1.x));
  }

  __host__ __device__ __forceinline__ float4 frsqrt(float4 a) {
    float rsq; int32_t p;
#ifndef __CUDA_ARCH__
    rsq = 1. / std::sqrt(a.x);
    float4 x = make_float4(std::frexp(rsq, &p), 0.f, 0.f, 0.f);
#else
    rsq = rsqrtf(a.x);
    float4 x = make_float4(frexpf(rsq, &p), 0.f, 0.f, 0.f);
#endif
    float2 c = make_float2(1.5f, 0.f);
    a = fscalbn(negate(a), (p << 1) - 1);

    x = mul(x, add_float2(mul(x, mul(a, x)), c));
    x = mul(x, add_float2(mul(x, mul(a, x)), c));
    x = mul(x, add_float2(mul(x, mul(a, x)), c));
    return fscalbn(x, p);
  }

  __host__ __device__ __forceinline__ float4 conv_i64_qf(int64_t i) {
#ifndef __CUDA_ARCH__
    using std::scalbnf;
#endif
    constexpr uint32_t i24 = (uint32_t(1) << 24) - uint32_t(1);
    float x = float(int16_t(i >> 48)), y = float(uint32_t(i >> 24) & i24), z = float(uint32_t(i) & i24);
    return normalize(make_float4(scalbnf(x, 48), scalbnf(y, 24), z, 0.f));
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ float4 conv_a63_qf(uint64_t const (&a)[ORDER], int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 4, "Integer 64 accumulation order must be in [1,4]");

    constexpr uint64_t u63 = uint64_t(1) << 63;
    float4 res = conv_i64_qf(a[ORDER - 1] | ((a[ORDER - 1] << 1) & u63));
    if constexpr(3 < ORDER) res = add(fscalbn(res, 63), conv_i64_qf(a[2]));
    if constexpr(2 < ORDER) res = add(fscalbn(res, 63), conv_i64_qf(a[1]));
    if constexpr(1 < ORDER) res = add(fscalbn(res, 63), conv_i64_qf(a[0]));
    return fscalbn(res, expon);
  }

  __host__ __device__ __forceinline__ double qf2double(float4 a) {
    return (double(a.x) + double(a.y)) + (double(a.z) + double(a.w));
  }

  __host__ __device__ __forceinline__ float4 double2qf(double a) {
    float a0 = float(a); a = a - double(a0);
    float a1 = float(a);
    return make_float4(a0, a1, a - double(a1), 0.f);
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
