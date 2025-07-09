#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct complex_float4 {
  float4 real;
  float4 imag;
};

#ifdef __CUDACC__
namespace device::qf {
  __device__ __forceinline__ complex_float4 make_complex_float4(float4 real, float4 imag) {
    return complex_float4({ real, imag });
  }

  __device__ __forceinline__ float2 fneg2(float2 a) {
    return make_float2(-a.x, -a.y);
  }

  __device__ __forceinline__ float4 negate(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
  }

  __device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
    return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
  }

  __device__ __forceinline__ float4 fadd4(float4 a, float4 b) {
    return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
  }

  __device__ __forceinline__ float4 collapse_float4(float4 a) {
    union float_vec { float4 vec4; float2 vec2[2]; } s = {a}, delta;
    float2 sum = fadd2(s.vec2[0], s.vec2[1]);

    delta.vec2[1] = fadd2(s.vec2[0], fneg2(sum));
    delta.vec2[0] = fneg2(fadd2(sum, delta.vec2[1]));
    delta.vec4 = fadd4(s.vec4, delta.vec4);
    delta.vec2[0] = fadd2(delta.vec2[0], delta.vec2[1]);
    s.vec4 = make_float4(sum.x, delta.vec4.x, sum.y, delta.vec4.y);
    sum = fadd2(s.vec2[0], s.vec2[1]);

    delta.vec2[1] = fadd2(s.vec2[0], fneg2(sum));
    delta.vec2[0] = fneg2(fadd2(sum, delta.vec2[1]));
    delta.vec4 = fadd4(s.vec4, delta.vec4);
    s.vec2[0] = sum;
    s.vec2[1] = fadd2(delta.vec2[0], delta.vec2[1]);
    sum.x = __fadd_rn(s.vec4.x, s.vec4.y);

    delta.vec4.y = __fadd_rn(s.vec4.x, -sum.x);
    delta.vec4.x = -(__fadd_rn(sum.x, delta.vec4.y));
    delta.vec2[0] = fadd2(s.vec2[0], delta.vec2[0]);
    s.vec2[0] = make_float2(sum.x, __fadd_rn(delta.vec4.x, delta.vec4.y));
    return s.vec4;
  }

  __device__ __forceinline__ float4 normalize(float4 a) {
    a = collapse_float4(a);
    float4 q = collapse_float4(make_float4(a.y, a.z, a.w, 0.f));
    
    float sum = __fadd_rn(q.y, q.z);
    float delta = __fadd_rn(q.y, -sum);
    float2 err = fadd2(make_float2(q.y, q.z), make_float2(-(__fadd_rn(sum, delta)), delta));
    return make_float4(a.x, q.x, sum, __fadd_rn(err.x, err.y));
  }

  __device__ __forceinline__ void fadd4_err(float4 a, float4 b, float4& sum, float4& err) {
    sum = fadd4(a, b);
    float4 delta = fadd4(a, negate(sum));
    err = fadd4(fadd4(a, negate(fadd4(sum, delta))), fadd4(b, delta));
  }

  __device__ __forceinline__ float4 add(float4 a, float4 b) {
    fadd4_err(a, b, a, b);
    b = make_float4(b.w, b.x, b.y, b.z);
    fadd4_err(a, b, a, b);
    b = make_float4(b.w, b.x, b.y, b.z);
    fadd4_err(a, b, a, b);
    return normalize(fadd4(a, make_float4(b.w, b.x, b.y, b.z)));
  }

  __device__ __forceinline__ float4 mul(float4 a, float4 b) {
    float2 prod = make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
    float2 err = make_float2(__fmaf_rn(a.x, b.x, -prod.x), __fmaf_rn(a.y, b.y, -prod.y));
    float4 c = make_float4(prod.x, err.x, prod.y, err.y);

    prod = make_float2(__fmul_rn(a.y, b.x), __fmul_rn(a.z, b.x));
    err = make_float2(__fmaf_rn(a.y, b.x, -prod.x), __fmaf_rn(a.z, b.x, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    prod = make_float2(__fmul_rn(a.x, b.y), __fmul_rn(a.x, b.z));
    err = make_float2(__fmaf_rn(a.x, b.y, -prod.x), __fmaf_rn(a.x, b.z, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    return add(c, make_float4(__fmul_rn(a.w, b.x), __fmul_rn(a.z, b.y), __fmul_rn(a.y, b.z), __fmul_rn(a.x, b.w)));
  }

  __device__ __forceinline__ float4 fma(float4 a, float4 b, float4 c) {
    float2 prod = make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
    float2 err = make_float2(__fmaf_rn(a.x, b.x, -prod.x), __fmaf_rn(a.y, b.y, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    prod = make_float2(__fmul_rn(a.y, b.x), __fmul_rn(a.z, b.x));
    err = make_float2(__fmaf_rn(a.y, b.x, -prod.x), __fmaf_rn(a.z, b.x, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    prod = make_float2(__fmul_rn(a.x, b.y), __fmul_rn(a.x, b.z));
    err = make_float2(__fmaf_rn(a.x, b.y, -prod.x), __fmaf_rn(a.x, b.z, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    return add(c, make_float4(__fmul_rn(a.w, b.x), __fmul_rn(a.z, b.y), __fmul_rn(a.y, b.z), __fmul_rn(a.x, b.w)));
  }

  __device__ __forceinline__ float4 fscalbn(float4 a, int32_t exp) {
    return make_float4(scalbnf(a.x, exp), scalbnf(a.y, exp), scalbnf(a.z, exp), scalbnf(a.w, exp));
  }

  __device__ __forceinline__ float4 frsqrt(float4 a) {
    int32_t p = int32_t(scalbnf(-log2f(a.x), -1));

    float4 x = make_float4(scalbnf(rsqrtf(a.x), -p), 0.f, 0.f, 0.f);
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    a = fscalbn(negate(a), (p << 1) - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));

    return fscalbn(x, p);
  }

  __device__ __forceinline__ complex_float4 negate(complex_float4 a) {
    return make_complex_float4(negate(a.real), negate(a.imag));
  }

  __device__ __forceinline__ complex_float4 conj(complex_float4 a) {
    return make_complex_float4(a.real, negate(a.imag));
  }

  __device__ __forceinline__ complex_float4 add(complex_float4 a, complex_float4 b) {
    return make_complex_float4(add(a.real, b.real), add(a.imag, b.imag));
  }

  __device__ __forceinline__ complex_float4 mul(complex_float4 a, complex_float4 b) {
    return make_complex_float4(fma(a.real, b.real, mul(negate(a.imag), b.imag)),
      fma(a.real, b.imag, mul(a.imag, b.real)));
  }

  __device__ __forceinline__ complex_float4 fma(complex_float4 a, complex_float4 b, complex_float4 c) {
    return make_complex_float4(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
#endif

namespace host::qf {
  inline complex_float4 make_complex_float4(float4 real, float4 imag) {
    return complex_float4({ real, imag });
  }

  inline float2 fneg2(float2 a) {
    return make_float2(-a.x, -a.y);
  }

  inline float4 negate(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
  }

  inline float2 fadd2(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
  }

  inline float4 fadd4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }

  inline float4 collapse_float4(float4 a) { // 30 additions
    union float_vec { float4 vec4; float2 vec2[2]; } s = {a}, delta;
    float2 sum = fadd2(s.vec2[0], s.vec2[1]);

    delta.vec2[1] = fadd2(s.vec2[0], fneg2(sum));
    delta.vec2[0] = fneg2(fadd2(sum, delta.vec2[1]));
    delta.vec4 = fadd4(s.vec4, delta.vec4);
    delta.vec2[0] = fadd2(delta.vec2[0], delta.vec2[1]);
    s.vec4 = make_float4(sum.x, delta.vec4.x, sum.y, delta.vec4.y);
    sum = fadd2(s.vec2[0], s.vec2[1]);

    delta.vec2[1] = fadd2(s.vec2[0], fneg2(sum));
    delta.vec2[0] = fneg2(fadd2(sum, delta.vec2[1]));
    delta.vec4 = fadd4(s.vec4, delta.vec4);
    s.vec2[0] = sum;
    s.vec2[1] = fadd2(delta.vec2[0], delta.vec2[1]);
    sum.x = s.vec4.x + s.vec4.y;

    delta.vec4.y = s.vec4.x - sum.x;
    delta.vec4.x = -(sum.x + delta.vec4.y);
    delta.vec2[0] = fadd2(s.vec2[0], delta.vec2[0]);
    s.vec2[0] = make_float2(sum.x, delta.vec4.x + delta.vec4.y);
    return s.vec4;
  }

  inline float4 normalize(float4 a) { // 66 additions
    a = collapse_float4(a); // s13, s24, e1, e2 -> s1234, e12, e3, e4 -> s_all, e_all, e3, e4
    float4 q = collapse_float4(make_float4(a.y, a.z, a.w, 0.f));
    
    float sum = q.y + q.z;
    float delta = q.y - sum;
    float2 err = fadd2(make_float2(q.y, q.z), make_float2(-(sum + delta), delta));
    return make_float4(a.x, q.x, sum, err.x + err.y);
  }

  inline void fadd4_err(float4 a, float4 b, float4& sum, float4& err) { // 24 additions
    sum = fadd4(a, b);
    float4 delta = fadd4(a, negate(sum));
    err = fadd4(fadd4(a, negate(fadd4(sum, delta))), fadd4(b, delta));
  }

  inline float4 add(float4 a, float4 b) { // 138 additions
    fadd4_err(a, b, a, b);
    b = make_float4(b.w, b.x, b.y, b.z);
    fadd4_err(a, b, a, b);
    b = make_float4(b.w, b.x, b.y, b.z);
    fadd4_err(a, b, a, b);
    return normalize(fadd4(a, make_float4(b.w, b.x, b.y, b.z)));
  }

  inline float4 mul(float4 a, float4 b) { // 16 fma, 414 additions
    float2 prod = make_float2(a.x * b.x, a.y * b.y);
    float2 err = make_float2(std::fmaf(a.x, b.x, -prod.x), std::fmaf(a.y, b.y, -prod.y));
    float4 c = make_float4(prod.x, err.x, prod.y, err.y);

    prod = make_float2(a.y * b.x, a.z * b.x);
    err = make_float2(std::fmaf(a.y, b.x, -prod.x), std::fmaf(a.z, b.x, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    prod = make_float2(a.x * b.y, a.x * b.z);
    err = make_float2(std::fmaf(a.x, b.y, -prod.x), std::fmaf(a.x, b.z, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    return add(c, make_float4(a.w * b.x, a.z * b.y, a.y * b.z, a.x * b.w));
  }

  inline float4 fma(float4 a, float4 b, float4 c) { // 16 fma, 552 additions
    float2 prod = make_float2(a.x * b.x, a.y * b.y);
    float2 err = make_float2(std::fmaf(a.x, b.x, -prod.x), std::fmaf(a.y, b.y, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    prod = make_float2(a.y * b.x, a.z * b.x);
    err = make_float2(std::fmaf(a.y, b.x, -prod.x), std::fmaf(a.z, b.x, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    prod = make_float2(a.x * b.y, a.x * b.z);
    err = make_float2(std::fmaf(a.x, b.y, -prod.x), std::fmaf(a.x, b.z, -prod.y));
    c = add(c, make_float4(prod.x, err.x, prod.y, err.y));

    return add(c, make_float4(a.w * b.x, a.z * b.y, a.y * b.z, a.x * b.w));
  }

  inline float4 fscalbn(float4 a, int32_t exp) {
    return make_float4(std::scalbnf(a.x, exp), std::scalbnf(a.y, exp), std::scalbnf(a.z, exp), std::scalbnf(a.w, exp));
  }

  inline float4 frsqrt(float4 a) {
    int32_t p = int32_t(-0.5f * std::log2f(a.x));

    float4 x = make_float4(std::scalbnf(1.f / std::sqrt(a.x), -p), 0.f, 0.f, 0.f); // init
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    a = fscalbn(negate(a), 2 * p - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c)); // x *= (1.5 + (-0.5 * a) * (x * x))

    return fscalbn(x, p);
  }

  inline bool isnormal(float4 a) {
    return std::isnormal(a.x) && std::isfinite(a.y) && std::isfinite(a.z) && std::isfinite(a.w);
  }

  inline complex_float4 negate(complex_float4 a) {
    return make_complex_float4(negate(a.real), negate(a.imag));
  }

  inline complex_float4 conj(complex_float4 a) {
    return make_complex_float4(a.real, negate(a.imag));
  }

  inline complex_float4 add(complex_float4 a, complex_float4 b) {
    return make_complex_float4(add(a.real, b.real), add(a.imag, b.imag));
  }

  inline complex_float4 mul(complex_float4 a, complex_float4 b) {
    return make_complex_float4(fma(a.real, b.real, mul(negate(a.imag), b.imag)),
      fma(a.real, b.imag, mul(a.imag, b.real)));
  }

  inline complex_float4 fma(complex_float4 a, complex_float4 b, complex_float4 c) {
    return make_complex_float4(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
