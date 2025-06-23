#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace device::f4 {
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
    sum = make_float2(-sum.x, -sum.y);
    float2 err = fadd2_rn(a, sum);
    return fadd2_rn(fadd2_rn(a, fadd2_rn(sum, make_float2(-err.x, -err.y))), fadd2_rn(b, err));
  }

  __device__ __forceinline__ float4 fadd4_err(float4 a, float4 b, float4 sum) {
    sum = make_float4(-sum.x, -sum.y, -sum.z, -sum.w);
    float4 err = fadd4_rn(a, sum);
    return fadd4_rn(fadd4_rn(a, fadd4_rn(sum, make_float4(-err.x, -err.y, -err.z, -err.w))), fadd4_rn(b, err));
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
    c = add(add(c, p), ffma4_rn(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    a = make_float4(a.w, a.x, a.y, a.z);
    p = ffma4_rn(a, b, z);
    c = add(add(c, p), ffma4_rn(a, b, make_float4(-p.x, -p.y, -p.z, -p.w)));

    return c;
  }

};
