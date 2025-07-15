#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct complex_double2 {
  double2 real;
  double2 imag;
};

#ifdef __CUDACC__
namespace device::dd {
  __device__ __forceinline__ complex_double2 make_complex_double2(double2 real, double2 imag) {
    return complex_double2({ real, imag });
  }

  __device__ __forceinline__ double2 negate(double2 a) {
    return make_double2(-a.x, -a.y);
  }

  __device__ __forceinline__ double2 fadd2(double2 a, double2 b) {
    return make_double2(__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y));
  }

  __device__ __forceinline__ double4 fadd4(double4 a, double4 b) {
    return make_double4(__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y), __dadd_rn(a.z, b.z), __dadd_rn(a.w, b.w));
  }

  __device__ __forceinline__ double2 normalize(double2 a) {
    double sum = __dadd_rn(a.x, a.y);
    double delta = __dadd_rn(a.x, -sum);
    double2 err = fadd2(a, make_double2(-(__dadd_rn(sum, delta)), delta));
    return make_double2(sum, __dadd_rn(err.x, err.y));
  }

  __device__ __forceinline__ double2 add(double2 a, double2 b) {
    union { double2 vec2[2]; double4 vec4; } s = { a, b }, delta;
    double2 sum = fadd2(s.vec2[0], s.vec2[1]);

    delta.vec2[1] = fadd2(s.vec2[0], negate(sum));
    delta.vec2[0] = negate(fadd2(sum, delta.vec2[1]));
    delta.vec4 = fadd4(s.vec4, delta.vec4);
    delta.vec2[0] = fadd2(delta.vec2[0], delta.vec2[1]);

    s.vec4.x = __dadd_rn(sum.x, sum.y);
    s.vec4.y = __dadd_rn(sum.x, -s.vec4.x);
    s.vec4.z = __dadd_rn(sum.y, s.vec4.y);
    s.vec4.w = __dadd_rn(delta.vec4.x, delta.vec4.y);
    return make_double2(s.vec4.x, __dadd_rn(s.vec4.z, s.vec4.w));
  }

  __device__ __forceinline__ double2 mul(double2 a, double2 b) {
    double2 d;
    d.x = __dmul_rn(a.x, b.x);
    d.y = __fma_rn(a.x, b.x, -d.x);
    d.y = __fma_rn(a.x, b.y, d.y);
    d.y = __fma_rn(a.y, b.x, d.y);

    double s = __dadd_rn(d.x, d.y);
    double delta = __dadd_rn(d.x, -s);
    return make_double2(s, __dadd_rn(d.y, delta));
  }

  __device__ __forceinline__ double2 fma(double2 a, double2 b, double2 c) {
    double2 d;
    d.x = __dmul_rn(a.x, b.x);
    d.y = __fma_rn(a.x, b.x, -d.x);
    d.y = __fma_rn(a.x, b.y, d.y);
    d.y = __fma_rn(a.y, b.x, d.y);
    return add(c, d);
  }

  __device__ __forceinline__ double2 fscalbn2(double2 a, int32_t exp) {
    return make_double2(scalbn(a.x, exp), scalbn(a.y, exp));
  }

  __device__ __forceinline__ double2 frsqrt(double2 a) {
    int32_t p = int32_t(scalbn(-log2(a.x), -1));

    double2 x = make_double2(scalbn(rsqrt(a.x), -p), 0.);
    double2 c = make_double2(1.5, 0.);
    a = fscalbn2(negate(a), (p << 1) - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));

    return fscalbn2(x, p);
  }

  __device__ __forceinline__ complex_double2 negate(complex_double2 a) {
    return make_complex_double2(negate(a.real), negate(a.imag));
  }

  __device__ __forceinline__ complex_double2 conj(complex_double2 a) {
    return make_complex_double2(a.real, negate(a.imag));
  }

  __device__ __forceinline__ complex_double2 add(complex_double2 a, complex_double2 b) {
    return make_complex_double2(add(a.real, b.real), add(a.imag, b.imag));
  }

  __device__ __forceinline__ complex_double2 mul(complex_double2 a, complex_double2 b) {
    return make_complex_double2(fma(a.real, b.real, mul(negate(a.imag), b.imag)),
      fma(a.real, b.imag, mul(a.imag, b.real)));
  }

  __device__ __forceinline__ complex_double2 fma(complex_double2 a, complex_double2 b, complex_double2 c) {
    return make_complex_double2(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
#endif

namespace host::dd {
  inline complex_double2 make_complex_double2(double2 real, double2 imag) {
    return complex_double2({ real, imag });
  }

  inline double2 negate(double2 a) {
    return make_double2(-a.x, -a.y);
  }

  inline double2 fadd2(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
  }

  inline double4 fadd4(double4 a, double4 b) {
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }

  inline double2 normalize(double2 a) {
    double sum = a.x + a.y;
    double delta = a.x - sum;
    double2 err = fadd2(a, make_double2(-(sum + delta), delta));
    return make_double2(sum, err.x + err.y);
  }

  inline double2 add(double2 a, double2 b) {
    union { double2 vec2[2]; double4 vec4; } s = { a, b }, delta;
    double2 sum = fadd2(s.vec2[0], s.vec2[1]);

    delta.vec2[1] = fadd2(s.vec2[0], negate(sum));
    delta.vec2[0] = negate(fadd2(sum, delta.vec2[1]));
    delta.vec4 = fadd4(s.vec4, delta.vec4);
    delta.vec2[0] = fadd2(delta.vec2[0], delta.vec2[1]);

    s.vec4.x = sum.x + sum.y;
    s.vec4.y = sum.x - s.vec4.x;
    s.vec4.z = sum.y + s.vec4.y;
    s.vec4.w = delta.vec4.x + delta.vec4.y;
    return make_double2(s.vec4.x, s.vec4.z + s.vec4.w);
  }

  inline double2 mul(double2 a, double2 b) {
    double2 d;
    d.x = a.x * b.x;
    d.y = std::fma(a.x, b.x, -d.x);
    d.y = std::fma(a.x, b.y, d.y);
    d.y = std::fma(a.y, b.x, d.y);

    double s = d.x + d.y;
    double delta = d.x - s;
    return make_double2(s, d.y + delta);
  }

  inline double2 fma(double2 a, double2 b, double2 c) {
    double2 d;
    d.x = a.x * b.x;
    d.y = std::fma(a.x, b.x, -d.x);
    d.y = std::fma(a.x, b.y, d.y);
    d.y = std::fma(a.y, b.x, d.y);
    return add(c, d);
  }

  inline double2 fscalbn2(double2 a, int32_t exp) {
    return make_double2(std::scalbn(a.x, exp), std::scalbn(a.y, exp));
  }

  inline double2 frsqrt(double2 a) {
    int32_t p = int32_t(-0.5 * std::log2(a.x));

    double2 x = make_double2(std::scalbn(1. / std::sqrt(a.x), -p), 0.); // init
    double2 c = make_double2(1.5, 0.);
    a = fscalbn2(negate(a), 2 * p - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c)); // x *= (1.5 + (-0.5 * a) * (x * x))

    return fscalbn2(x, p);
  }

  inline bool isnormal(double2 a) {
    return std::isnormal(a.x) && std::isfinite(a.y);
  }

  inline complex_double2 negate(complex_double2 a) {
    return make_complex_double2(negate(a.real), negate(a.imag));
  }

  inline complex_double2 conj(complex_double2 a) {
    return make_complex_double2(a.real, negate(a.imag));
  }

  inline complex_double2 add(complex_double2 a, complex_double2 b) {
    return make_complex_double2(add(a.real, b.real), add(a.imag, b.imag));
  }

  inline complex_double2 mul(complex_double2 a, complex_double2 b) {
    return make_complex_double2(fma(a.real, b.real, mul(negate(a.imag), b.imag)),
      fma(a.real, b.imag, mul(a.imag, b.real)));
  }

  inline complex_double2 fma(complex_double2 a, complex_double2 b, complex_double2 c) {
    return make_complex_double2(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
