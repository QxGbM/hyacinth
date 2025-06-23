#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace host::f4 {
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
    sum = make_float2(-sum.x, -sum.y);
    float2 err = fadd2(a, sum);
    return fadd2(fadd2(a, fadd2(sum, make_float2(-err.x, -err.y))), fadd2(b, err));
  }

  inline float4 fadd4_err(float4 a, float4 b, float4 sum) {
    sum = make_float4(-sum.x, -sum.y, -sum.z, -sum.w);
    float4 err = fadd4(a, sum);
    return fadd4(fadd4(a, fadd4(sum, make_float4(-err.x, -err.y, -err.z, -err.w))), fadd4(b, err));
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
    c = add(add(c, p), ffma4(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4(a, b, z);
    c = add(add(c, p), ffma4(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    return c;
  }

  inline float4 fscalbnf4(float4 a, int32_t exp) {
    return make_float4(scalbnf(a.x, exp), scalbnf(a.y, exp), scalbnf(a.z, exp), scalbnf(a.w, exp));
  }

  inline float4 rsqrt(float4 a) {
    int32_t p = int32_t(-0.5f * std::log2(a.x));

    float4 x = make_float4(scalbnf(1.f / std::sqrt(a.x), -p), 0.f, 0.f, 0.f); // init
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    a = fscalbnf4(make_float4(-a.x, -a.y, -a.z, -a.w), 2 * p - 1);

    x = fma(x, fma(a, fma(x, x, z), c), z);
    x = fma(x, fma(a, fma(x, x, z), c), z);
    x = fma(x, fma(a, fma(x, x, z), c), z); // x *= (1.5 + (-0.5 * a) * (x * x))

    return fscalbnf4(x, p);
  }

};

