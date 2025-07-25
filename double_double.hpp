#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct complex_double2 {
  double2 real;
  double2 imag;
};

namespace device::dd {
  __host__ __device__ __forceinline__ complex_double2 make_complex_double2(double2 real, double2 imag) {
    return complex_double2({ real, imag });
  }

  __host__ __device__ __forceinline__ double2 negate(double2 a) {
    return make_double2(-a.x, -a.y);
  }

  __host__ __device__ __forceinline__ void fadd2_err(double2 a, double2 b, double2& sum, double2& err) {
    constexpr uint64_t i63 = ~(uint64_t(1) << 63);
    union { double2 fp; uint64_t in[2]; } va{a}, vb{b};
    sum = make_double2(a.x + b.x, a.y + b.y);
    int32_t pred_x = (va.in[0] & i63) < (vb.in[0] & i63);
    int32_t pred_y = (va.in[1] & i63) < (vb.in[1] & i63);

    a = make_double2(pred_x ? vb.fp.x : va.fp.x, pred_y ? vb.fp.y : va.fp.y);
    b = make_double2(pred_x ? va.fp.x : vb.fp.x, pred_y ? va.fp.y : vb.fp.y);
    err = make_double2(b.x + (a.x - sum.x), b.y + (a.y - sum.y));
  }

  __host__ __device__ __forceinline__ double2 normalize(double2 a) {
    constexpr uint64_t i63 = ~(uint64_t(1) << 63);
    union { double2 fp; uint64_t in[2]; } va{a};
    double sum = a.x + a.y;

    int32_t pred = (va.in[0] & i63) < (va.in[1] & i63);
    a = make_double2(pred ? va.fp.y : va.fp.x, pred ? va.fp.x : va.fp.y);
    return make_double2(sum, a.y + (a.x - sum));
  }

  __host__ __device__ __forceinline__ double2 add(double2 a, double2 b) {
    fadd2_err(a, b, a, b);
    a = normalize(a);
    b.x = b.x + b.y;
    return make_double2(a.x, a.y + b.x);
  }

  __host__ __device__ __forceinline__ double2 mul(double2 a, double2 b) {
    double2 d;
#ifdef __CUDA_ARCH__
    d.x = __dmul_rn(a.x, b.x);
    d.y = __fma_rn(a.x, b.x, -d.x);
    d.y = __fma_rn(a.x, b.y, d.y);
    d.y = __fma_rn(a.y, b.x, d.y);

    double s = __dadd_rn(d.x, d.y);
    double delta = __dadd_rn(d.x, -s);
    return make_double2(s, __dadd_rn(d.y, delta));
#else
    d.x = a.x * b.x;
    d.y = std::fma(a.x, b.x, -d.x);
    d.y = std::fma(a.x, b.y, d.y);
    d.y = std::fma(a.y, b.x, d.y);

    double s = d.x + d.y;
    double delta = d.x - s;
    return make_double2(s, d.y + delta);
#endif
  }

  __host__ __device__ __forceinline__ double2 fma(double2 a, double2 b, double2 c) {
    double2 d;
#ifdef __CUDA_ARCH__
    d.x = __dmul_rn(a.x, b.x);
    d.y = __fma_rn(a.x, b.x, -d.x);
    d.y = __fma_rn(a.x, b.y, d.y);
    d.y = __fma_rn(a.y, b.x, d.y);
#else
    d.x = a.x * b.x;
    d.y = std::fma(a.x, b.x, -d.x);
    d.y = std::fma(a.x, b.y, d.y);
    d.y = std::fma(a.y, b.x, d.y);
#endif
    return add(c, d);
  }

  __host__ __device__ __forceinline__ double2 fscalbn2(double2 a, int32_t exp) {
#ifdef __CUDA_ARCH__
    return make_double2(scalbn(a.x, exp), scalbn(a.y, exp));
#else
    return make_double2(std::scalbn(a.x, exp), std::scalbn(a.y, exp));
#endif
  }

  __host__ __device__ __forceinline__ double2 frsqrt(double2 a) {
#ifdef __CUDA_ARCH__
    int32_t p = int32_t(scalbn(-log2(a.x), -1));
    double2 x = make_double2(scalbn(rsqrt(a.x), -p), 0.);
#else
    int32_t p = int32_t(-0.5 * std::log2(a.x));
    double2 x = make_double2(std::scalbn(1. / std::sqrt(a.x), -p), 0.);
#endif
    double2 c = make_double2(1.5, 0.);
    a = fscalbn2(negate(a), (p << 1) - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));

    return fscalbn2(x, p);
  }

  __host__ __device__ __forceinline__ int32_t fisfinite(double2 a) {
#ifdef __CUDA_ARCH__
    return isfinite(a.x) && isfinite(a.y);
#else
    return std::isfinite(a.x) && std::isfinite(a.y);
#endif
  }

  __host__ __device__ __forceinline__ complex_double2 negate(complex_double2 a) {
    return make_complex_double2(negate(a.real), negate(a.imag));
  }

  __host__ __device__ __forceinline__ complex_double2 conj(complex_double2 a) {
    return make_complex_double2(a.real, negate(a.imag));
  }

  __host__ __device__ __forceinline__ complex_double2 add(complex_double2 a, complex_double2 b) {
    return make_complex_double2(add(a.real, b.real), add(a.imag, b.imag));
  }

  __host__ __device__ __forceinline__ complex_double2 mul(complex_double2 a, complex_double2 b) {
    return make_complex_double2(fma(a.real, b.real, mul(negate(a.imag), b.imag)),
      fma(a.real, b.imag, mul(a.imag, b.real)));
  }

  __host__ __device__ __forceinline__ complex_double2 fma(complex_double2 a, complex_double2 b, complex_double2 c) {
    return make_complex_double2(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};

