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

  __host__ __device__ __forceinline__ float4 add(float4 a, float4 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1); // 1122 - 1223
    fadd_err(a.z, b.z, a.z, b.z); // 33 - 34, 4@b.z

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1); // 2233 - 2334, 4@a1.y
    fadd_err(a0.y, a1.x, a0.y, a1.x); // 33 - 34, 4@a1.x
    return normalize(make_float4(r0, a0.x, a0.y, a.w + b.w + b.z + a1.y + a1.x));
  }

  __host__ __device__ __forceinline__ float4 add_float2(float4 a, float2 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1);

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1);
    fadd_err(a0.y, a1.x, a0.y, a1.x);
    return normalize(make_float4(r0, a0.x, a0.y, a.w + a1.y + a1.x));
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

  __host__ __device__ __forceinline__ float4 frsqrt(float4 a) {
#ifdef __CUDA_ARCH__
    int32_t p = int32_t(scalbnf(-log2f(a.x), -1));
    float4 x = make_float4(scalbnf(rsqrtf(a.x), -p), 0.f, 0.f, 0.f);
#else
    int32_t p = int32_t(-0.5f * std::log2f(a.x));
    float4 x = make_float4(std::scalbnf(1.f / std::sqrt(a.x), -p), 0.f, 0.f, 0.f);
#endif
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    a = fscalbn(negate(a), (p << 1) - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));

    return fscalbn(x, p);
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
