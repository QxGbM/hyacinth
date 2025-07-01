#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace host::dd {
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
    double2 c = fadd2(a, b);
    double2 d = fadd2_err(a, b, c);
    return normalize(fadd2(make_double2(c.y, c.x), d));
  }

  inline double2 fma(double2 a, double2 b, double2 c) {
    double2 z = make_double2(0., 0.);
    double2 p = ffma2(a, b, z);
    c = add(add(c, p), ffma2(a, b, negate(p)));

    a = make_double2(a.y, a.x);
    p = ffma2(a, b, z);
    return add(add(c, p), ffma2(a, b, negate(p)));
  }

  inline double2 fscalbn2(double2 a, int32_t exp) {
    return make_double2(std::scalbn(a.x, exp), std::scalbn(a.y, exp));
  }

  inline double2 rsqrt(double2 a) {
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

};
