
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ __forceinline__ void two_sum_f4(float4 a0, float4 a1, float4& sum, float4& err) {
  sum.x = a0.x + a1.x;
  sum.y = a0.y + a1.y;
  sum.z = a0.z + a1.z;
  sum.w = a0.w + a1.w;

  err.x = sum.x - a0.x;
  err.y = sum.y - a0.y;
  err.z = sum.z - a0.z;
  err.w = sum.w - a0.w;

  a1.x = a1.x - err.x;
  a1.y = a1.y - err.y;
  a1.z = a1.z - err.z;
  a1.w = a1.w - err.w;

  err.x = sum.x - err.x;
  err.y = sum.y - err.y;
  err.z = sum.z - err.z;
  err.w = sum.w - err.w;

  a0.x = a0.x - err.x;
  a0.y = a0.y - err.y;
  a0.z = a0.z - err.z;
  a0.w = a0.w - err.w;

  err.x = a0.x + a1.x;
  err.y = a0.y + a1.y;
  err.z = a0.z + a1.z;
  err.w = a0.w + a1.w;
}

__host__ __device__ __forceinline__ void two_prod_f4(float4 a0, float4 a1, float4& prod, float4& err) {
  prod.x = a0.x * a1.x;
  prod.y = a0.y * a1.y;
  prod.z = a0.z * a1.z;
  prod.w = a0.w * a1.w;

  err.x = fmaf(a0.x, a1.x, -prod.x);
  err.y = fmaf(a0.y, a1.y, -prod.y);
  err.z = fmaf(a0.z, a1.z, -prod.z);
  err.w = fmaf(a0.w, a1.w, -prod.w);
}

__host__ __device__ __forceinline__ float4 renormalize(float4 a) {
  float sum;
  sum = a.x + a.y;
  a.y += a.x - sum;
  a.x = sum;

  sum = a.y + a.z;
  a.z += a.y - sum;
  a.y = sum;

  sum = a.z + a.w;
  a.w += a.z - sum;
  a.z = sum;

  return a;
}

__host__ __device__ __forceinline__ float4 float4_add(float4 a, float4 b) {
  two_sum_f4(a, b, a, b);

  b = make_float4(b.w, b.x, b.y, b.z);
  two_sum_f4(a, b, a, b);

  b = make_float4(b.w, b.x, b.y, b.z);
  two_sum_f4(a, b, a, b);

  b = make_float4(b.w, b.x, b.y, b.z);
  two_sum_f4(a, b, a, b);

  return renormalize(a);
}

__host__ __device__ __forceinline__ float4 float4_fma(float4 a, float4 b, float4 c) {
  float4 prod, err;
  two_prod_f4(a, b, prod, err);
  c = float4_add(c, prod);
  c = float4_add(c, err);

  b = make_float4(b.w, b.x, b.y, b.z);
  two_prod_f4(a, b, prod, err);
  c = float4_add(c, prod);
  c = float4_add(c, err);

  b = make_float4(b.w, b.x, b.y, b.z);
  two_prod_f4(a, b, prod, err);
  c = float4_add(c, prod);
  c = float4_add(c, err);

  b = make_float4(b.w, b.x, b.y, b.z);
  two_prod_f4(a, b, prod, err);
  c = float4_add(c, prod);
  c = float4_add(c, err);

  return c;
}
