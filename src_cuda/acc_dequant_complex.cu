
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <limits>

template <int32_t ORDER>
__device__ __forceinline__ void cross_sum(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER]) {
  uint64_t t[ORDER]{};
  if constexpr(0 < ORDER) { device::int8::add_shifted(t, -int64_t(im[0]), uint32_t(0)); }
  if constexpr(1 < ORDER) { device::int8::add_shifted(t, -int64_t(im[1]), uint32_t(63)); }
  if constexpr(2 < ORDER) { device::int8::add_shifted(t, -int64_t(im[2]), uint32_t(126)); }
  if constexpr(0 < ORDER) { int64_t r = -int64_t(rl[0]); device::int8::add_shifted(t, r, uint32_t(0)); device::int8::add_shifted(im, r, uint32_t(0)); }
  if constexpr(1 < ORDER) { int64_t r = -int64_t(rl[1]); device::int8::add_shifted(t, r, uint32_t(63)); device::int8::add_shifted(im, r, uint32_t(63)); }
  if constexpr(2 < ORDER) { int64_t r = -int64_t(rl[2]); device::int8::add_shifted(t, r, uint32_t(126)); device::int8::add_shifted(im, r, uint32_t(126)); }
  if constexpr(0 < ORDER) { rl[0] = t[0]; }
  if constexpr(1 < ORDER) { rl[1] = t[1]; }
  if constexpr(2 < ORDER) { rl[2] = t[2]; }
}

constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e), device::qf::conv_a63_qf(im, e)); }

template<int32_t orderA, class complex_t>
__global__ void dequantize_complex_kernel(int64_t K, int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int32_t umax, const int32_t* __restrict__ vec_expon, complex_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);

  if (y <= x) {
    constexpr uint32_t shifts[]{ uint32_t(0), uint32_t(63), uint32_t(126) };
    int64_t iter = x + y * N + strideA * int64_t(orderA);
    uint64_t acc_rl[orderA], acc_im[orderA];

    #pragma unroll
    for (int32_t i = 0; i < orderA; ++i)
    { acc_rl[i] = A[iter]; iter += strideA; }

    iter = y + x * N + strideA * int64_t(orderA);
    #pragma unroll
    for (int32_t i = 0; i < orderA; ++i)
    { acc_im[i] = A[iter]; iter += strideA; }

    if (K) {
      iter = (strideA * int64_t(orderA * 2)) + y;
      #pragma unroll
      for (int32_t i = 0; i < orderA; ++i)
      { device::int8::add_shifted(acc_rl, int64_t(A[iter]), uint32_t(umax) + shifts[i]); iter += N; }

      #pragma unroll
      for (int32_t i = 0; i < orderA; ++i)
      { device::int8::add_shifted(acc_im, int64_t(A[iter]), uint32_t(umax) + shifts[i]); iter += N; }

      iter = (strideA * int64_t(orderA * 2)) + x;
      #pragma unroll
      for (int32_t i = 0; i < orderA; ++i)
      { device::int8::add_shifted(acc_im, int64_t(A[iter]), uint32_t(umax) + shifts[i]); iter += N; }

      #pragma unroll
      for (int32_t i = 0; i < orderA; ++i)
      { device::int8::add_shifted(acc_rl, int64_t(A[iter]), uint32_t(umax) + shifts[i]); iter += N; }

      cross_sum(acc_rl, acc_im);
      device::int8::add_shifted(acc_rl, K, uint32_t(umax <<= 1));
    }
    else { cross_sum(acc_rl, acc_im); }
    
    iter = y + x * N;
    #pragma unroll
    for (int32_t i = 0; i < orderA; ++i)
    { device::int8::add_shifted(acc_rl, int64_t(A[iter]), shifts[i]); iter += strideA; }

    int32_t ex = vec_expon[x], ey = vec_expon[y]; iter = y + x * ldb;
    if (ex == int_min || ey == int_min) B[iter] = complex_t();
      else cscal(acc_rl, acc_im, ex + ey - umax, B[iter]);
  }
}

template<class complex_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int64_t K, int64_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_t* B, int64_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideA = N * N; K <<= 1;

  switch(orderA) {
    case 1: dequantize_complex_kernel<1> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 2: dequantize_complex_kernel<2> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 3: dequantize_complex_kernel<3> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    default: return;
  }
}

namespace internal::int8 {

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuDoubleComplex* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuComplex* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_double2* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_float4* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

}
