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

  __device__ __forceinline__ double2 fadd2_err(double2 a, double2 b, double2 sum) {
    sum = negate(sum);
    double2 err = fadd2(a, sum);
    return fadd2(fadd2(a, fadd2(sum, negate(err))), fadd2(b, err));
  }

  __device__ __forceinline__ double2 ffma2(double2 a, double2 b, double2 c) {
    return make_double2(__fma_rn(a.x, b.x, c.x), __fma_rn(a.y, b.y, c.y));
  }

  __device__ __forceinline__ double2 normalize(double2 a) {
    double sum = -(a.x + a.y);
    double delta = a.x + sum;
    double err = (a.x + (sum - delta)) + (a.y + delta);
    return make_double2(-sum, err);
  }

  __device__ __forceinline__ double2 add(double2 a, double2 b) {
    double2 c = fadd2(a, b);
    double2 d = fadd2_err(a, b, c);
    return normalize(fadd2(make_double2(c.y, c.x), d));
  }

  __device__ __forceinline__ double2 fma(double2 a, double2 b, double2 c) {
    double2 d;
    d.x = a.x * b.x;
    d.y = __fma_rn(a.x, b.x, -d.x);
    d.y = __fma_rn(a.x, b.y, d.y);
    d.y = __fma_rn(a.y, b.x, d.y);
    return add(c, d);
  }

  __device__ __forceinline__ double2 fscalbn2(double2 a, int32_t exp) {
    return make_double2(scalbn(a.x, exp), scalbn(a.y, exp));
  }

  __device__ __forceinline__ double2 frsqrt(double2 a) {
    int32_t p = int32_t(-0.5 * log2(a.x));

    double2 x = make_double2(scalbn(rsqrt(a.x), -p), 0.);
    double2 z = make_double2(0., 0.);
    double2 c = make_double2(1.5, 0.);
    a = fscalbn2(negate(a), 2 * p - 1);

    x = fma(x, fma(x, fma(a, x, z), c), z);
    x = fma(x, fma(x, fma(a, x, z), c), z);
    x = fma(x, fma(x, fma(a, x, z), c), z);

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

  inline double2 fadd2_err(double2 a, double2 b, double2 sum) {
    sum = negate(sum);
    double2 err = fadd2(a, sum);
    return fadd2(fadd2(a, fadd2(sum, negate(err))), fadd2(b, err));
  }

  inline double2 ffma2(double2 a, double2 b, double2 c) {
    return make_double2(std::fma(a.x, b.x, c.x), std::fma(a.y, b.y, c.y));
  }

  inline double2 normalize(double2 a) {
    double sum = -(a.x + a.y);
    double delta = a.x + sum;
    double err = (a.x + (sum - delta)) + (a.y + delta);
    return make_double2(-sum, err);
  }

  inline double2 add(double2 a, double2 b) {
    double2 c = fadd2(a, b); // 2 additions
    double2 d = fadd2_err(a, b, c); // 10 additions
    return normalize(fadd2(make_double2(c.y, c.x), d)); // total 20 additions
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
    double2 z = make_double2(0., 0.);
    double2 c = make_double2(1.5, 0.);
    a = fscalbn2(negate(a), 2 * p - 1);

    x = fma(x, fma(x, fma(a, x, z), c), z);
    x = fma(x, fma(x, fma(a, x, z), c), z);
    x = fma(x, fma(x, fma(a, x, z), c), z); // x *= (1.5 + (-0.5 * a) * (x * x))

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

  inline complex_double2 fma(complex_double2 a, complex_double2 b, complex_double2 c) {
    return make_complex_double2(fma(a.real, b.real, fma(negate(a.imag), b.imag, c.real)),
      fma(a.real, b.imag, fma(a.imag, b.real, c.imag)));
  }

};
