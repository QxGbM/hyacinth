#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct complex_double2 {
  double2 real;
  double2 imag;
};

namespace device::dd {
  constexpr int32_t use_pred_fast_sum = 1;

  __host__ __device__ __forceinline__ complex_double2 make_complex_double2(double2 real, double2 imag) {
    return complex_double2({ real, imag });
  }

  __host__ __device__ __forceinline__ double2 negate(double2 a) {
    return make_double2(-a.x, -a.y);
  }

  __host__ __device__ __forceinline__ void fadd_err(double a, double b, double& sum, double& err) {
    sum = a + b;
    if constexpr(use_pred_fast_sum) {
      union { double fp; uint64_t in; } va{a}, vb{b};
      int32_t pred = (va.in << 1) < (vb.in << 1);

      a = pred ? vb.fp : va.fp;
      b = pred ? va.fp : vb.fp;
      err = b + (a - sum);
    }
    else {
      double delta = a - sum;
      double s_delta = sum + delta;
      double b_delta = b + delta;
      err = (a - s_delta) + b_delta;
    }
  }

  __host__ __device__ __forceinline__ void fadd2_err(double2 a, double2 b, double2& sum, double2& err) {
    sum = make_double2(a.x + b.x, a.y + b.y);
    if constexpr(use_pred_fast_sum) {
      union { double2 fp; uint64_t in[2]; } va{a}, vb{b};
      int32_t pred_x = (va.in[0] << 1) < (vb.in[0] << 1);
      int32_t pred_y = (va.in[1] << 1) < (vb.in[1] << 1);

      a = make_double2(pred_x ? vb.fp.x : va.fp.x, pred_y ? vb.fp.y : va.fp.y);
      b = make_double2(pred_x ? va.fp.x : vb.fp.x, pred_y ? va.fp.y : vb.fp.y);
      err = make_double2(b.x + (a.x - sum.x), b.y + (a.y - sum.y));
    }
    else {
      double2 delta = make_double2(a.x - sum.x, a.y - sum.y);
      double2 s_delta = make_double2(sum.x + delta.x, sum.y + delta.y);
      double2 b_delta = make_double2(b.x + delta.x, b.y + delta.y);
      err = make_double2((a.x - s_delta.x) + b_delta.x, (a.y - s_delta.y) + b_delta.y);
    }
  }

  __host__ __device__ __forceinline__ double2 normalize(double2 a) {
    fadd_err(a.x, a.y, a.x, a.y);
    return a;
  }

  __host__ __device__ __forceinline__ double2 add(double2 a, double2 b) {
    fadd2_err(a, b, a, b);
    a.y += b.x + b.y;

    double s = a.x + a.y;
    double delta = a.x - s;
    return make_double2(s, a.y + delta);
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
    return add(c, mul(a, b));
  }

  __host__ __device__ __forceinline__ double2 fscalbn(double2 a, int32_t exp) {
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
    a = fscalbn(negate(a), (p << 1) - 1);

    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));
    x = mul(x, fma(x, mul(a, x), c));

    return fscalbn(x, p);
  }

  __host__ __device__ __forceinline__ double2 add_double(double2 a, double b) {
    double2 c = normalize(make_double2(a.x, b));
    c.y += a.y;

    double s = c.x + c.y;
    double delta = c.x - s;
    return make_double2(s, c.y + delta);
  }

  __host__ __device__ __forceinline__ double conv_i31_f64(uint32_t i, int32_t expon) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    int32_t sign_i = int32_t(i) | (int32_t(i << 1) & (1 << 31));
    return scalbn(double(sign_i), expon);
  }

  __host__ __device__ __forceinline__ double conv_u31_f64(uint32_t i, int32_t expon) {
#ifndef __CUDA_ARCH__
    using std::scalbn;
#endif
    return scalbn(double(i), expon);
  }

  template<int32_t ORDER>
  __host__ __device__ __forceinline__ double2 conv_i31(uint32_t const (&code)[ORDER], int32_t expon) {
    static_assert(1 <= ORDER && ORDER <= 6, "Integer order must be in [1,6]");

    if constexpr(1 == ORDER)
      return make_double2(conv_i31_f64(code[0], expon), 0.);
    else if constexpr(2 == ORDER)
      return normalize(make_double2(conv_i31_f64(code[1], expon + 31), conv_u31_f64(code[0], expon)));
    else if constexpr(3 == ORDER) {
      double2 res = normalize(make_double2(conv_i31_f64(code[2], expon + 31), conv_u31_f64(code[1], expon)));
      return add_double(fscalbn(res, 31), conv_u31_f64(code[0], expon));
    }
    else if constexpr(4 == ORDER) {
      double2 res = normalize(make_double2(conv_i31_f64(code[3], expon + 31), conv_u31_f64(code[2], expon)));
      return add(fscalbn(res, 62), normalize(make_double2(conv_u31_f64(code[1], expon + 31), conv_u31_f64(code[0], expon))));
    }
    else if constexpr(5 == ORDER) {
      double2 res = normalize(make_double2(conv_i31_f64(code[4], expon + 31), conv_u31_f64(code[3], expon)));
      res = add(fscalbn(res, 62), normalize(make_double2(conv_u31_f64(code[2], expon + 31), conv_u31_f64(code[1], expon))));
      return add_double(fscalbn(res, 31), conv_u31_f64(code[0], expon));
    }
    else {
      double2 res = normalize(make_double2(conv_i31_f64(code[5], expon + 31), conv_u31_f64(code[4], expon)));
      res = add(fscalbn(res, 62), normalize(make_double2(conv_u31_f64(code[3], expon + 31), conv_u31_f64(code[2], expon))));
      return add(fscalbn(res, 62), normalize(make_double2(conv_u31_f64(code[1], expon + 31), conv_u31_f64(code[0], expon))));
    }
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

