
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ __forceinline__ float2 float2_vec_sum(float2 a, float2 b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

__host__ __device__ __forceinline__ float4 float4_vec_sum(float4 a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

__host__ __device__ __forceinline__ float4 float4_vec_fma(float4 a, float4 b, float4 c) {
  c.x = fmaf(a.x, b.x, c.x);
  c.y = fmaf(a.y, b.y, c.y);
  c.z = fmaf(a.z, b.z, c.z);
  c.w = fmaf(a.w, b.w, c.w);
  return c;
}

__host__ __device__ __forceinline__ float2 float2_vec_sum_err(float2 a, float2 b, float2 sum) {
  sum = make_float2(-sum.x, -sum.y);
  float2 err = float2_vec_sum(a, sum);
  return float2_vec_sum(float2_vec_sum(a, float2_vec_sum(sum, make_float2(-err.x, -err.y))), float2_vec_sum(b, err));
}

__host__ __device__ __forceinline__ float4 float4_vec_sum_err(float4 a, float4 b, float4 sum) {
  sum = make_float4(-sum.x, -sum.y, -sum.z, -sum.w);
  float4 err = float4_vec_sum(a, sum);
  return float4_vec_sum(float4_vec_sum(a, float4_vec_sum(sum, make_float4(-err.x, -err.y, -err.z, -err.w))), float4_vec_sum(b, err));
}

__host__ __device__ __forceinline__ float4 normalize(float4 a) {
  float2 a0 = make_float2(a.x, a.z);
  float2 a1 = make_float2(a.y, a.w);
  float2 s = float2_vec_sum(a0, a1); // (x+y, z+w)
  float2 e = float2_vec_sum_err(a0, a1, s); // a:(x=s0, y=e0, z=s1, w=e1)

  a0 = make_float2(s.x, e.x);
  a1 = make_float2(s.y, e.y);
  s = float2_vec_sum(a0, a1); // (x+z, y+w)
  e = float2_vec_sum_err(a0, a1, s); // a:(x=s0, y=s1, z=e0, w=e1)

  a0 = make_float2(s.x, s.y);
  a1 = make_float2(e.y, e.x);
  s = float2_vec_sum(a0, a1); // (x+w, y+z)
  e = float2_vec_sum_err(a0, a1, s); // a:(x=s0, y=s1, z=e1, w=e0)

  return make_float4(s.x, s.y, e.y, e.x);
}

__host__ __device__ __forceinline__ float4 float4_sum(float4 a, float4 b) {
  float4 c = float4_vec_sum(a, b);
  a = float4_vec_sum_err(a, b, c);

  a = make_float4(a.w, a.x, a.y, a.z);
  b = float4_vec_sum(a, c);
  a = float4_vec_sum_err(a, c, b);

  a = make_float4(a.w, a.x, a.y, a.z);
  c = float4_vec_sum(a, b);
  a = float4_vec_sum_err(a, b, c);

  a = make_float4(a.w, a.x, a.y, a.z);
  b = float4_vec_sum(a, c);

  return normalize(b);
}

__host__ __device__ __forceinline__ float4 float4_sum_err(float4 a, float4 b, float4& err) {
  float4 c = float4_vec_sum(a, b);
  a = float4_vec_sum_err(a, b, c);

  a = make_float4(a.w, a.x, a.y, a.z);
  b = float4_vec_sum(a, c);
  a = float4_vec_sum_err(a, c, b);

  a = make_float4(a.w, a.x, a.y, a.z);
  c = float4_vec_sum(a, b);
  a = float4_vec_sum_err(a, b, c);

  a = make_float4(a.w, a.x, a.y, a.z);
  b = float4_vec_sum(a, c);
  a = float4_vec_sum_err(a, c, b);

  err = normalize(a);
  return normalize(b);
}

__host__ __device__ __forceinline__ float4 float4_fma(float4 a, float4 b, float4 c) {
  float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 prod = float4_vec_fma(a, b, zero);
  c = float4_sum(float4_sum(c, prod), float4_vec_fma(a, b, make_float4(-prod.x, -prod.y, -prod.z, -prod.w)));

  a = make_float4(a.w, a.x, a.y, a.z);
  prod = float4_vec_fma(a, b, zero);
  c = float4_sum(float4_sum(c, prod), float4_vec_fma(a, b, make_float4(-prod.x, -prod.y, -prod.z, -prod.w)));

  a = make_float4(a.w, a.x, a.y, a.z);
  prod = float4_vec_fma(a, b, zero);
  c = float4_sum(float4_sum(c, prod), float4_vec_fma(a, b, make_float4(-prod.x, -prod.y, -prod.z, -prod.w)));

  a = make_float4(a.w, a.x, a.y, a.z);
  prod = float4_vec_fma(a, b, zero);
  c = float4_sum(float4_sum(c, prod), float4_vec_fma(a, b, make_float4(-prod.x, -prod.y, -prod.z, -prod.w)));

  return c;
}

__host__ __device__ __forceinline__ float4 float4_scalbnf(float4 a, int32_t exp) {
#ifdef __CUDACC__
  return make_float4(ldexpf(a.x, exp), ldexpf(a.y, exp), ldexpf(a.z, exp), ldexpf(a.w, exp));
#else
  return make_float4(scalbnf(a.x, exp), scalbnf(a.y, exp), scalbnf(a.z, exp), scalbnf(a.w, exp));
#endif
}

__host__ __device__ __forceinline__ float4 float4_reciprocal(float4 a) {
#ifdef __CUDACC__
  int32_t p = int32_t(log2f(a.x));
  float ax = ldexpf(1.f / a.x, -p);
#else
  int32_t p = int32_t(std::log2(a.x));
  float ax = scalbnf(1.f / a.x, -p);
#endif

  float4 res = make_float4(ax, 0.f, 0.f, 0.f); // init
  float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 c = make_float4(2.f, 0.f, 0.f, 0.f);
  a = float4_scalbnf(make_float4(-a.x, -a.y, -a.z, -a.w), p);

  res = float4_fma(res, float4_fma(a, res, c), zero); // x *= (2 + (-1 * a) * x)
  res = float4_fma(res, float4_fma(a, res, c), zero);
  res = float4_fma(res, float4_fma(a, res, c), zero);

  return float4_scalbnf(res, p);
}
