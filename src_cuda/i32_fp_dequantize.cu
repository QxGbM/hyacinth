
#include <hyacin.hpp>
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <uint32_t depth, class real_t> struct dequantize_func {
  real_t* __restrict__ A;
  const int32_t* __restrict__ B;
  uint32_t expon; int64_t M;
  dequantize_func(int64_t M, int32_t expon, real_t* A, const int32_t* B) : A(A), B(B), expon(expon & 0x7fffffff), M(M) {}

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i64(double& f, uint64_t const (&a)[ORDER], int32_t expon) {
    f += device::dd::conv_a63_f64(a, expon);
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i64(float& f, uint64_t const (&a)[ORDER], int32_t expon) {
    f += float(device::dd::conv_a63_f64(a, expon));
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i64(double2& f, uint64_t const (&a)[ORDER], int32_t expon) {
    f = device::dd::add(f, device::dd::conv_a63_dd(a, expon));
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i64(float4& f, uint64_t const (&a)[ORDER], int32_t expon) {
    f = device::qf::add(f, device::qf::conv_a63_qf(a, expon));
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    // Bounded by 2 * INT_MAX(31-bits) * 2^(BASE * (D-1)). order = (32+B(D-1)+62) / 63
    constexpr int32_t acc_order = (94 + (depth - 1) * device::Config::exp_base) / 63;
    uint64_t acc[acc_order]{};

    #pragma unroll
    for (int32_t col = 0; col < depth; ++col) {
      uint32_t e = device::Config::exp_base * col;
      int64_t B_i = i + int64_t(col) * M;
      device::int8::add_shifted(acc, B[B_i], e);
    }
    acc_i64(A[i], acc, expon);
  }
};

template <class real_t, class real_ptr>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t depth_lo, int32_t depth_hi, int32_t N, real_ptr A, const int32_t* B, int32_t ld) {
  int32_t depth = depth_hi - depth_lo;
  int32_t expon = device::Config::exp_base * depth_lo;
  int64_t M = int64_t(N) * int64_t(ld);
  thrust::counting_iterator<int64_t> iter(0);

  switch (depth) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<1, real_t>(M, expon, A, B)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<2, real_t>(M, expon, A, B)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<3, real_t>(M, expon, A, B)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<4, real_t>(M, expon, A, B)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<5, real_t>(M, expon, A, B)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<6, real_t>(M, expon, A, B)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<7, real_t>(M, expon, A, B)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<8, real_t>(M, expon, A, B)); break;
    case 9: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<9, real_t>(M, expon, A, B)); break;
    case 10: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<10, real_t>(M, expon, A, B)); break;
    default: break;
  }

  if constexpr (device::Config::exp_base < 7)
    switch (depth) {
      case 11: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<11, real_t>(M, expon, A, B)); break;
      case 12: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<12, real_t>(M, expon, A, B)); break;
      case 13: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<13, real_t>(M, expon, A, B)); break;
      default: break;
    }

  if constexpr (device::Config::exp_base < 5)
    switch (depth) {
      case 14: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<14, real_t>(M, expon, A, B)); break;
      case 15: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<15, real_t>(M, expon, A, B)); break;
      case 16: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, dequantize_func<16, real_t>(M, expon, A, B)); break;
      default: break;
    }
}

void internal::int8::dequantize_f64_i32tensor(cudaStream_t stream, int32_t depth_lo, int32_t depth_hi, int32_t N, double* A, const int32_t* B, int32_t ld) {
  dequantize_dispatcher<double, double* __restrict__>(stream, depth_lo, depth_hi, N, A, B, ld);
}

void internal::int8::dequantize_f32_i32tensor(cudaStream_t stream, int32_t depth_lo, int32_t depth_hi, int32_t N, float* A, const int32_t* B, int32_t ld) {
  dequantize_dispatcher<float, float* __restrict__>(stream, depth_lo, depth_hi, N, A, B, ld);
}

void internal::int8::dequantize_f128_dd_i32tensor(cudaStream_t stream, int32_t depth_lo, int32_t depth_hi, int32_t N, double2* A, const int32_t* B, int32_t ld) {
  dequantize_dispatcher<double2, double2* __restrict__>(stream, depth_lo, depth_hi, N, A, B, ld);
}

void internal::int8::dequantize_f128_qf_i32tensor(cudaStream_t stream, int32_t depth_lo, int32_t depth_hi, int32_t N, float4* A, const int32_t* B, int32_t ld) {
  dequantize_dispatcher<float4, float4* __restrict__>(stream, depth_lo, depth_hi, N, A, B, ld);
}

