
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <uint32_t order, class real_t> struct decode_func {
  real_t* __restrict__ A;
  const int32_t* __restrict__ B;
  int32_t expon; int64_t M;
  decode_func(int64_t M, int32_t expon, real_t* A, const int32_t* B) : A(A), B(B), expon(expon), M(M) {}

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i32(double& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f += device::dd::conv_a31_f64(a, expon);
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i32(float& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f += device::qf::conv_a31_f32(a, expon);
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i32(double2& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f = device::dd::add(f, device::dd::conv_a31_dd(a, expon));
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void acc_i32(float4& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f = device::qf::add(f, device::qf::conv_a31_qf(a, expon));
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    constexpr int32_t acc_bits = 31;
    constexpr int32_t acc_order = 1 + ((order * device::Config::exp_base) + (acc_bits - 1)) / acc_bits;
    uint32_t acc[acc_order]{};

    #pragma unroll
    for (int32_t col = 0; col < order; ++col) {
      int32_t e = device::Config::exp_base * col;
      int64_t B_i = i + int64_t(col) * M;
      device::int8::add_shifted(acc, B[B_i], e);
    }
    acc_i32(A[i], acc, expon);
  }
};

template <class real_t, class real_ptr>
inline void decode_dispatcher(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, real_ptr A, const int32_t* B, int32_t ld) {
  int32_t order = order_hi - order_lo;
  int32_t expon = device::Config::exp_base * order_lo;
  int64_t M = int64_t(N) * int64_t(ld);
  thrust::counting_iterator<int64_t> iter(0);

  switch (order) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<1, real_t>(M, expon, A, B)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<2, real_t>(M, expon, A, B)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<3, real_t>(M, expon, A, B)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<4, real_t>(M, expon, A, B)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<5, real_t>(M, expon, A, B)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<6, real_t>(M, expon, A, B)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<7, real_t>(M, expon, A, B)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<8, real_t>(M, expon, A, B)); break;
    case 9: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<9, real_t>(M, expon, A, B)); break;
    case 10: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<10, real_t>(M, expon, A, B)); break;
    default: break;
  }

  if constexpr (device::Config::exp_base < 7)
    switch (order) {
      case 11: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<11, real_t>(M, expon, A, B)); break;
      case 12: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<12, real_t>(M, expon, A, B)); break;
      case 13: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<13, real_t>(M, expon, A, B)); break;
      default: break;
    }

  if constexpr (device::Config::exp_base < 5)
    switch (order) {
      case 14: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<14, real_t>(M, expon, A, B)); break;
      case 15: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<15, real_t>(M, expon, A, B)); break;
      case 16: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, decode_func<16, real_t>(M, expon, A, B)); break;
      default: break;
    }
}

void internal::int8::decode_f64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<double, double* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

void internal::int8::decode_f32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<float, float* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

void internal::int8::decode_f128_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double2* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<double2, double2* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

void internal::int8::decode_f128_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float4* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<float4, float4* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

