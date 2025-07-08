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

  __device__ __forceinline__ float4 ffma4(float4 a, float4 b, float4 c) {
    return make_float4(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
  }

  __device__ __forceinline__ float4 fadd4_err(float4 a, float4 b, float4 sum) {
    float4 err = fadd4(a, negate(sum));
    return fadd4(fadd4(a, negate(fadd4(sum, err))), fadd4(b, err));
  }

  __device__ __forceinline__ float4 collapse_float4(float4 a) {
    union float_vec { float4 vec4; float2 vec2[2]; } s = {a}, delta;
    float2 sum = fadd2(s.vec2[0], s.vec2[1]);

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

  __device__ __forceinline__ float4 add(float4 a, float4 b) {
    float4 c = fadd4(a, b);
    a = fadd4_err(a, b, c);

    a = make_float4(a.w, a.x, a.y, a.z);
    b = fadd4(a, c);
    a = fadd4_err(a, c, b);

    a = make_float4(a.w, a.x, a.y, a.z);
    c = fadd4(a, b);
    a = fadd4_err(a, b, c);

    a = make_float4(a.w, a.x, a.y, a.z);
    b = fadd4(a, c);

    return normalize(b);
  }

  __device__ __forceinline__ float4 fma(float4 a, float4 b, float4 c) {
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 tail = ffma4(make_float4(a.w, a.z, a.y, a.x), b, z);
    a = make_float4(a.y, a.x, a.z, a.x);
    b = make_float4(b.x, b.y, b.x, b.z);
    float4 p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, negate(p)));

    float2 hi_term = make_float2(__fmul_rn(a.y, b.x), __fmul_rn(a.x, b.y));
    float2 hi_err = make_float2(__fmaf_rn(a.y, b.x, -hi_term.x), __fmaf_rn(a.x, b.y, -hi_term.y));
    return add(tail, add(c, make_float4(hi_term.x, hi_err.x, hi_term.y, hi_err.y)));
  }

  __device__ __forceinline__ float4 fscalbn(float4 a, int32_t exp) {
    return make_float4(scalbnf(a.x, exp), scalbnf(a.y, exp), scalbnf(a.z, exp), scalbnf(a.w, exp));
  }

  __device__ __forceinline__ float4 frsqrt(float4 a) {
    int32_t p = int32_t(-0.5f * log2f(a.x));

    float4 x = make_float4(scalbnf(rsqrtf(a.x), -p), 0.f, 0.f, 0.f); // init
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    a = fscalbn(negate(a), 2 * p - 1);

    x = fma(x, fma(a, fma(x, x, z), c), z);
    x = fma(x, fma(a, fma(x, x, z), c), z);
    x = fma(x, fma(a, fma(x, x, z), c), z); // x *= (1.5 + (-0.5 * a) * (x * x))

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

  inline float4 ffma4(float4 a, float4 b, float4 c) {
    return make_float4(std::fmaf(a.x, b.x, c.x), std::fmaf(a.y, b.y, c.y), std::fmaf(a.z, b.z, c.z), std::fmaf(a.w, b.w, c.w));
  }

  inline float4 fadd4_err(float4 a, float4 b, float4 sum) {
    float4 err = fadd4(a, negate(sum));
    return fadd4(fadd4(a, negate(fadd4(sum, err))), fadd4(b, err));
  }

  inline float4 collapse_float4(float4 a) { // 18 additions
    union float_vec { float4 vec4; float2 vec2[2]; } s = {a}, delta;
    float2 sum = fadd2(s.vec2[0], s.vec2[1]);

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

  inline float4 normalize(float4 a) { // 42 additions
    a = collapse_float4(a); // s12, s34, e1, e2 -> s1234, e3, e1, e2
    float4 q = collapse_float4(make_float4(a.y, a.z, a.w, 0.f));
    
    float sum = q.y + q.z;
    float delta = q.y - sum;
    float2 err = fadd2(make_float2(q.y, q.z), make_float2(-(sum + delta), delta));
    return make_float4(a.x, q.x, sum, err.x + err.y);
  }

  inline float4 add(float4 a, float4 b) { // 130 additions
    float4 c = fadd4(a, b);
    a = fadd4_err(a, b, c);

    a = make_float4(a.w, a.x, a.y, a.z);
    b = fadd4(a, c);
    a = fadd4_err(a, c, b);

    a = make_float4(a.w, a.x, a.y, a.z);
    c = fadd4(a, b);
    a = fadd4_err(a, b, c);

    a = make_float4(a.w, a.x, a.y, a.z);
    b = fadd4(a, c);

    return normalize(b);
  }

  inline float4 fma(float4 a, float4 b, float4 c) { // 16 fma, 520 additions
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 tail = ffma4(make_float4(a.w, a.z, a.y, a.x), b, z);
    a = make_float4(a.y, a.x, a.z, a.x);
    b = make_float4(b.x, b.y, b.x, b.z);
    float4 p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, negate(p)));

    float2 hi_term = make_float2(a.y * b.x, a.x * b.y);
    float2 hi_err = make_float2(std::fmaf(a.y, b.x, -hi_term.x), std::fmaf(a.x, b.y, -hi_term.y));
    return add(tail, add(c, make_float4(hi_term.x, hi_err.x, hi_term.y, hi_err.y)));
  }

  inline float4 fscalbn(float4 a, int32_t exp) {
    return make_float4(std::scalbnf(a.x, exp), std::scalbnf(a.y, exp), std::scalbnf(a.z, exp), std::scalbnf(a.w, exp));
  }

  inline float4 frsqrt(float4 a) {
    int32_t p = int32_t(-0.5f * std::log2f(a.x));

    float4 x = make_float4(std::scalbnf(1.f / std::sqrt(a.x), -p), 0.f, 0.f, 0.f); // init
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    a = fscalbn(negate(a), 2 * p - 1);

    x = fma(x, fma(a, fma(x, x, z), c), z);
    x = fma(x, fma(a, fma(x, x, z), c), z);
    x = fma(x, fma(a, fma(x, x, z), c), z); // x *= (1.5 + (-0.5 * a) * (x * x))

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

  inline complex_float4 fma(complex_float4 a, complex_float4 b, complex_float4 c) {
    return make_complex_float4(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
