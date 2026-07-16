#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(32) complex_float4 { float4 real, imag; };

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

  __host__ __device__ __forceinline__ float4 renormalize(float4 a) {
    float s0, s1, d0, d1; 
    s0 = a.x + a.y; d0 = a.x - s0; a.x = s0; a.y += d0;
    s0 = a.x + a.z; d0 = a.x - s0; a.x = s0; a.z += d0;
    s1 = a.y + a.z; d1 = a.y - s1; a.y = s1; a.z += d1;
    s0 = a.x + a.w; d0 = a.x - s0; a.x = s0; a.w += d0;
    s1 = a.y + a.w; d1 = a.y - s1; a.y = s1; a.w += d1;
    s1 = a.z + a.w; d1 = a.z - s1; a.z = s1; a.w += d1;
    return a;
  }

  __host__ __device__ __forceinline__ float4 add(float4 a, float4 b) {
    float2 a0, a1;
    fadd2_err(make_float2(a.x, a.y), make_float2(b.x, b.y), a0, a1); // 1122 - 1223
    fadd_err(a.z, b.z, a.z, b.z); // 33 - 34, 4@b.z

    float r0 = a0.x;
    fadd2_err(make_float2(a0.y, a.z), a1, a0, a1); // 2233 - 2334, 4@a1.y
    fadd_err(a0.y, a1.x, a0.y, a1.x); // 33 - 34, 4@a1.x
    return renormalize(make_float4(r0, a0.x, a0.y, a.w + (b.w + b.z) + (a1.y + a1.x)));
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

    return renormalize(make_float4(c1, c23.x, c23.y, c4));
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

    return renormalize(make_float4(c1, c23.x, c23.y, c4));
  }

  __host__ __device__ __forceinline__ float4 fldexp(float4 a, int32_t e) {
#ifndef __CUDA_ARCH__
    return make_float4(std::ldexp(a.x, e), std::ldexp(a.y, e), std::ldexp(a.z, e), std::ldexp(a.w, e));
#else
    return make_float4(ldexpf(a.x, e), ldexpf(a.y, e), ldexpf(a.z, e), ldexpf(a.w, e));
#endif
  }

  __host__ __device__ __forceinline__ void frsqrt(float4 a, float4& sq, float4& rsq) {
    int32_t p;
#ifndef __CUDA_ARCH__
    float r = 1. / std::sqrt(a.x);
    float4 x = make_float4(std::frexp(r, &p), 0.f, 0.f, 0.f);
#else
    float r = rsqrtf(a.x);
    float4 x = make_float4(frexpf(r, &p), 0.f, 0.f, 0.f);
#endif
    float4 c = make_float4(1.5f, 0.f, 0.f, 0.f);
    float4 s = fldexp(negate(a), (p << 1) - 1);

    x = mul(x, add(mul(x, mul(s, x)), c));
    x = mul(x, add(mul(x, mul(s, x)), c));
    x = mul(x, add(mul(x, mul(s, x)), c));
    rsq = (x = fldexp(x, p));
    sq = mul(a, x);
  }

  __host__ __device__ __forceinline__ float4 conv_i64_qf_m126(uint64_t i) {
    constexpr uint32_t i24 = 0xffffff;
#ifndef __CUDA_ARCH__
    float x = std::ldexp(float(int16_t(i >> 48)), -78), y = std::ldexp(float(uint32_t(i >> 24) & i24), -102), z = std::ldexp(float(uint32_t(i) & i24), -126);
#else
    float x = ldexpf(float(int16_t(i >> 48)), -78), y = ldexpf(float(uint32_t(i >> 24) & i24), -102), z = ldexpf(float(uint32_t(i) & i24), -126);
#endif
    float sum = x + y, delta = x - sum; x = sum; y += delta;
    sum = x + z; delta = x - sum; x = sum; z += delta;
    sum = y + z; delta = y - sum; y = sum; z += delta;
    return make_float4(x, y, z, 0.f);
  }

  template <uint32_t ORDER>
  __host__ __device__ __forceinline__ float4 conv_a63_qf(uint64_t const (&a)[ORDER], int32_t e) {
    static_assert(1 <= ORDER && ORDER <= 3, "Integer 64 accumulation order must be in [1,3]");

    float4 res = conv_i64_qf_m126(a[ORDER - 1]);
    if constexpr(2 < ORDER) res = add(fldexp(res, 63), conv_i64_qf_m126(a[1]));
    if constexpr(1 < ORDER) res = add(fldexp(res, 63), conv_i64_qf_m126(a[0]));
    return fldexp(res, e + 126);
  }

  __host__ __device__ __forceinline__ double qf2double(float4 a) {
    return (double(a.x) + double(a.y)) + (double(a.z) + double(a.w));
  }

  __host__ __device__ __forceinline__ float4 double2qf(double a) {
    float a0 = float(a); a = a - double(a0);
    float a1 = float(a);
    return make_float4(a0, a1, a - double(a1), 0.f);
  }

};
