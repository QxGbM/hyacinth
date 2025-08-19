#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct complex_float4 {
  float4 real;
  float4 imag;
};

namespace device::qf {
  constexpr int32_t use_pred_fast_sum = 0;

  __host__ __device__ __forceinline__ complex_float4 make_complex_float4(float4 real, float4 imag) {
    return complex_float4({ real, imag });
  }

  __host__ __device__ __forceinline__ float4 negate(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
  }

  __host__ __device__ __forceinline__ void fadd_err(float a, float b, float& sum, float& err) {
    sum = a + b;
    if constexpr(use_pred_fast_sum) {
      union { float fp; uint32_t in; } va{a}, vb{b};
      int32_t pred = (va.in << 1) < (vb.in << 1);

      a = pred ? vb.fp : va.fp;
      b = pred ? va.fp : vb.fp;
      err = b + (a - sum);
    }
    else {
      float delta = a - sum;
      float s_delta = sum + delta;
      float b_delta = b + delta;
      err = (a - s_delta) + b_delta;
    }
  }

  __host__ __device__ __forceinline__ void fadd2_err(float2 a, float2 b, float2& sum, float2& err) {
    sum = make_float2(a.x + b.x, a.y + b.y);
    if constexpr(use_pred_fast_sum) {
      union { float2 fp; uint32_t in[2]; } va{a}, vb{b};
      int32_t pred_x = (va.in[0] << 1) < (vb.in[0] << 1);
      int32_t pred_y = (va.in[1] << 1) < (vb.in[1] << 1);

      a = make_float2(pred_x ? vb.fp.x : va.fp.x, pred_y ? vb.fp.y : va.fp.y);
      b = make_float2(pred_x ? va.fp.x : vb.fp.x, pred_y ? va.fp.y : vb.fp.y);
      err = make_float2(b.x + (a.x - sum.x), b.y + (a.y - sum.y));
    }
    else {
      float2 delta = make_float2(a.x - sum.x, a.y - sum.y);
      float2 s_delta = make_float2(sum.x + delta.x, sum.y + delta.y);
      float2 b_delta = make_float2(b.x + delta.x, b.y + delta.y);
      err = make_float2((a.x - s_delta.x) + b_delta.x, (a.y - s_delta.y) + b_delta.y);
    }
  }

  __host__ __device__ __forceinline__ float4 normalize(float4 a) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(a.z, a.w), a0, a1);
    fadd2_err(make_float2(a0.x, a1.x), make_float2(a0.y, a1.y), a0, a1);
    fadd_err(a0.y, a1.x, a0.y, a1.x);
    fadd_err(a1.x, a1.y, a1.x, a1.y);
    return make_float4(a0.x, a0.y, a1.x, a1.y);
  }

  __host__ __device__ __forceinline__ float2 add(float2 a, float2 b) {
    fadd2_err(a, b, a, b);
    a.y += b.x + b.y;

    float s = a.x + a.y;
    float delta = a.x - s;
    return make_float2(s, a.y + delta);
  }

  __host__ __device__ __forceinline__ float4 add(float4 a, float4 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1); // 1122 - 1223
    fadd_err(a.z, b.z, a.z, b.z); // 33 - 34, 4@b.z

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1); // 2233 - 2334, 4@a1.y
    fadd_err(a0.y, a1.x, a0.y, a1.x); // 33 - 34, 4@a1.x
    return normalize(make_float4(r0, a0.x, a0.y, a.w + b.w + b.z + a1.y + a1.x));
  }

  __host__ __device__ __forceinline__ void fmul2_err(float2 a, float2 b, float2& prod, float2& err) {
#ifdef __CUDA_ARCH__
    prod = make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
    err = make_float2(__fmaf_rn(a.x, b.x, -prod.x), __fmaf_rn(a.y, b.y, -prod.y));
#else
    prod = make_float2(a.x * b.x, a.y * b.y);
    err = make_float2(std::fmaf(a.x, b.x, -prod.x), std::fmaf(a.y, b.y, -prod.y));
#endif
  }

  __host__ __device__ __forceinline__ float4 mul(float4 a, float4 b) {
    float2 prod, err;
    fmul2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), prod, err);
    float c1 = prod.x;
    float c4 = err.y + (a.w * b.x + a.z * b.y) + (a.y * b.z + a.x * b.w);
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

  __host__ __device__ __forceinline__ float4 fma(float4 a, float4 b, float4 c) {
    return add(c, mul(a, b));
  }

  __host__ __device__ __forceinline__ float4 fscalbn(float4 a, int32_t exp) {
#ifdef __CUDA_ARCH__
    return make_float4(scalbnf(a.x, exp), scalbnf(a.y, exp), scalbnf(a.z, exp), scalbnf(a.w, exp));
#else
    return make_float4(std::scalbnf(a.x, exp), std::scalbnf(a.y, exp), std::scalbnf(a.z, exp), std::scalbnf(a.w, exp));
#endif
  }

  __host__ __device__ __forceinline__ float4 add_float2(float4 a, float2 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1);

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1);
    fadd_err(a0.y, a1.x, a0.y, a1.x);
    return normalize(make_float4(r0, a0.x, a0.y, a.w + a1.y + a1.x));
  }

  __host__ __device__ __forceinline__ float2 conv_i31_f32(uint32_t i, int32_t expon) {
#ifndef __CUDA_ARCH__
    using std::scalbnf;
#endif
    int32_t sign_i = int32_t(i) | (int32_t(i << 1) & (1 << 31));
    float c1 = float(sign_i);
    float c2 = float(sign_i + int32_t(-c1));
    return make_float2(scalbnf(c1, expon), scalbnf(c2, expon));
  }

  __host__ __device__ __forceinline__ float2 conv_u31_f32(uint32_t i, int32_t expon) {
#ifndef __CUDA_ARCH__
    using std::scalbnf;
#endif
    float c1 = float(i);
    float c2 = float(int32_t(i) + int32_t(-c1));
    return make_float2(scalbnf(c1, expon), scalbnf(c2, expon));
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ float conv_a31_f32(uint32_t const (&a)[ORDER], int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 6, "Integer 32 accumulation order must be in [1,6]");
#ifndef __CUDA_ARCH__
    using std::scalbnf;
#endif
    float res = conv_i31_f32(a[ORDER - 1], expon).x;
    if constexpr(5 < ORDER) res = scalbnf(res, 31) + conv_u31_f32(a[4], expon).x;
    if constexpr(4 < ORDER) res = scalbnf(res, 31) + conv_u31_f32(a[3], expon).x;
    if constexpr(3 < ORDER) res = scalbnf(res, 31) + conv_u31_f32(a[2], expon).x;
    if constexpr(2 < ORDER) res = scalbnf(res, 31) + conv_u31_f32(a[1], expon).x;
    if constexpr(1 < ORDER) res = scalbnf(res, 31) + conv_u31_f32(a[0], expon).x;
    return res;
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ float4 conv_a31_qf(uint32_t const (&a)[ORDER], int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 6, "Integer 32 accumulation order must be in [1,6]");

    float2 conv = conv_i31_f32(a[ORDER - 1], expon);
    float4 res = make_float4(conv.x, conv.y, 0.f, 0.f);
    if constexpr(5 < ORDER) res = add_float2(fscalbn(res, 31), conv_u31_f32(a[4], expon));
    if constexpr(4 < ORDER) res = add_float2(fscalbn(res, 31), conv_u31_f32(a[3], expon));
    if constexpr(3 < ORDER) res = add_float2(fscalbn(res, 31), conv_u31_f32(a[2], expon));
    if constexpr(2 < ORDER) res = add_float2(fscalbn(res, 31), conv_u31_f32(a[1], expon));
    if constexpr(1 < ORDER) res = add_float2(fscalbn(res, 31), conv_u31_f32(a[0], expon));
    return res;
  }

  __host__ __device__ __forceinline__ float4 dd2qf(double2 a) {
    float a0 = float(a.x); a.x = a.x - double(a0);
    float a1 = float(a.x); a.y = a.y + (a.x - double(a1));
    float a2 = float(a.y);
    return make_float4(a0, a1, a2, a.y - double(a2));
  }
  
  __host__ __device__ __forceinline__ double2 qf2dd(float4 a) {
    double2 c = make_double2(double(a.x) + double(a.y), double(a.z) + double(a.w));
    double s = c.x + c.y;
    double delta = c.x - s;
    return make_double2(s, c.y + delta);
  }

  __host__ __device__ __forceinline__ complex_float4 conj(complex_float4 a) {
    return make_complex_float4(a.real, negate(a.imag));
  }

  __host__ __device__ __forceinline__ complex_float4 add(complex_float4 a, complex_float4 b) {
    return make_complex_float4(add(a.real, b.real), add(a.imag, b.imag));
  }

  __host__ __device__ __forceinline__ complex_float4 mul(complex_float4 a, complex_float4 b) {
    return make_complex_float4(fma(a.real, b.real, mul(negate(a.imag), b.imag)),
      fma(a.real, b.imag, mul(a.imag, b.real)));
  }

  __host__ __device__ __forceinline__ complex_float4 fma(complex_float4 a, complex_float4 b, complex_float4 c) {
    return make_complex_float4(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
