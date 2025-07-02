#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct complex_float4 {
  float4 real;
  float4 imag;
};

#ifdef __CUDACC__
namespace device::f4 {
  __device__ __forceinline__ complex_float4 make_complex_float4(float4 real, float4 imag) {
    return complex_float4({ real, imag });
  }

  __device__ __forceinline__ float2 fneg2(float2 a) {
    return make_float2(-a.x, -a.y);
  }

  __device__ __forceinline__ float4 negate(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
  }

  __device__ __forceinline__ float2 fadd2_rn(float2 a, float2 b) {
    return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
  }

  __device__ __forceinline__ float4 fadd4_rn(float4 a, float4 b) {
    return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
  }

  __device__ __forceinline__ float4 ffma4_rn(float4 a, float4 b, float4 c) {
    return make_float4(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
  }

  __device__ __forceinline__ float2 fadd2_err(float2 a, float2 b, float2 sum) {
    sum = fneg2(sum);
    float2 err = fadd2_rn(a, sum);
    return fadd2_rn(fadd2_rn(a, fadd2_rn(sum, fneg2(err))), fadd2_rn(b, err));
  }

  __device__ __forceinline__ float4 fadd4_err(float4 a, float4 b, float4 sum) {
    sum = negate(sum);
    float4 err = fadd4_rn(a, sum); // err = a + (-sum);
    return fadd4_rn(fadd4_rn(a, fadd4_rn(sum, negate(err))), fadd4_rn(b, err)); // (a + ((-sum) + (-err))) + (b + err);
  }

  __device__ __forceinline__ float4 normalize(float4 a) {
    float2 a0 = make_float2(a.x, a.z);
    float2 a1 = make_float2(a.y, a.w);
    float2 s = fadd2_rn(a0, a1); // (x+y, z+w)
    float2 e = fadd2_err(a0, a1, s); // a:(x=s0, y=e0, z=s1, w=e1)

    a0 = make_float2(s.x, e.x);
    a1 = make_float2(s.y, e.y);
    s = fadd2_rn(a0, a1); // (x+z, y+w)
    e = fadd2_err(a0, a1, s); // a:(x=s0, y=s1, z=e0, w=e1)

    a0 = make_float2(s.x, s.y);
    a1 = make_float2(e.y, e.x);
    s = fadd2_rn(a0, a1); // (x+w, y+z)
    e = fadd2_err(a0, a1, s); // a:(x=s0, y=s1, z=e1, w=e0)

    return make_float4(s.x, s.y, e.y, e.x);
  }

  __device__ __forceinline__ float4 add(float4 a, float4 b) {
    float4 c = fadd4_rn(a, b);
    a = fadd4_err(a, b, c);

    a = make_float4(a.w, a.x, a.y, a.z);
    b = fadd4_rn(a, c);
    a = fadd4_err(a, c, b);

    a = make_float4(a.w, a.x, a.y, a.z);
    c = fadd4_rn(a, b);
    a = fadd4_err(a, b, c);

    a = make_float4(a.w, a.x, a.y, a.z);
    b = fadd4_rn(a, c);

    return normalize(b);
  }

  __device__ __forceinline__ float4 fma(float4 a, float4 b, float4 c) {
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, negate(p)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, negate(p)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, negate(p)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, negate(p)));

    return c;
  }

  __device__ __forceinline__ bool a_less_than_b(float4 a, float4 b) {
    b = add(negate(a), b);
    return 0.f < b.x;
  }

  __device__ __forceinline__ bool a_eq_to_b(float4 a, float4 b) {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
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

namespace host::f4 {
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

  inline float2 fadd2_err(float2 a, float2 b, float2 sum) {
    sum = fneg2(sum);
    float2 err = fadd2(a, sum);
    return fadd2(fadd2(a, fadd2(sum, fneg2(err))), fadd2(b, err));
  }

  inline float4 fadd4_err(float4 a, float4 b, float4 sum) {
    sum = negate(sum);
    float4 err = fadd4(a, sum); // err = a + (-sum);
    return fadd4(fadd4(a, fadd4(sum, negate(err))), fadd4(b, err)); // (a + ((-sum) + (-err))) + (b + err);
  }

  inline float4 normalize(float4 a) {
    float2 a0 = make_float2(a.x, a.z);
    float2 a1 = make_float2(a.y, a.w);
    float2 s = fadd2(a0, a1); // (x+y, z+w)
    float2 e = fadd2_err(a0, a1, s); // a:(x=s0, y=e0, z=s1, w=e1)

    a0 = make_float2(s.x, e.x);
    a1 = make_float2(s.y, e.y);
    s = fadd2(a0, a1); // (x+z, y+w)
    e = fadd2_err(a0, a1, s); // a:(x=s0, y=s1, z=e0, w=e1)

    a0 = make_float2(s.x, s.y);
    a1 = make_float2(e.y, e.x);
    s = fadd2(a0, a1); // (x+w, y+z)
    e = fadd2_err(a0, a1, s); // a:(x=s0, y=s1, z=e1, w=e0)

    return make_float4(s.x, s.y, e.y, e.x);
  }

  inline float4 add(float4 a, float4 b) {
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

  inline float4 fma(float4 a, float4 b, float4 c) {
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, negate(p)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, negate(p)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, negate(p)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, negate(p)));

    return c;
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
