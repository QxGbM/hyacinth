
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <limits>

constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, double& f) { f = device::dd::conv_a63_f64(a, e); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, float& f) { f = float(device::dd::conv_a63_f64(a, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, double2& f) { f = device::dd::conv_a63_dd(a, e); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, float4& f) { f = device::qf::conv_a63_qf(a, e); }

template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e), device::qf::conv_a63_qf(im, e)); }

template <int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < ORDER) { a[0] = *in; } if constexpr(1 < ORDER) { a[1] = *(in += stride); } if constexpr(2 < ORDER) { a[2] = *(in += stride); }
}

template <int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&r)[ORDER], uint64_t (&i)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < ORDER) { r[0] = *in; } if constexpr(1 < ORDER) { r[1] = *(in += stride); } if constexpr(2 < ORDER) { r[2] = *(in += stride); }
  if constexpr(0 < ORDER) { i[0] = *(in += stride); } if constexpr(1 < ORDER) { i[1] = *(in += stride); } if constexpr(2 < ORDER) { i[2] = *(in += stride); }
}

template<int32_t orderA, int32_t Complex, class matrix_t>
__global__ void triangle_unpack_dequantize_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int32_t umax2, const int32_t* __restrict__ vec_expon, matrix_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)];
    B = &B[y + (x * ldb)];

    if constexpr(Complex) {
      uint64_t acc_rl[orderA], acc_im[orderA];
      load_i(acc_rl, acc_im, A, strideA);

      int32_t ex = vec_expon[x], ey = vec_expon[y];
      if (ex == int_min || ey == int_min) *B = matrix_t();
        else cscal(acc_rl, acc_im, ex + ey - umax2, *B);
    }
    else {
      uint64_t acc[orderA];
      load_i(acc, A, strideA);
      
      int32_t ex = vec_expon[x], ey = vec_expon[y];
      if (ex == int_min || ey == int_min) *B = matrix_t();
        else fscal(acc, ex + ey - umax2, *B);
    }
  }
}

template<int32_t Complex, class matrix_t>
inline void tp_deq_dispatcher(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, matrix_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t N64 = int64_t(N), strideA = (N64 * N64 + N64) / int64_t(2), ldb64 = int64_t(ldb);
  umax <<= 1;
  switch(orderA) {
    case 1: triangle_unpack_dequantize_kernel<1, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, umax, vec_expon, B, ldb64); return;
    case 2: triangle_unpack_dequantize_kernel<2, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, umax, vec_expon, B, ldb64); return;
    case 3: triangle_unpack_dequantize_kernel<3, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, umax, vec_expon, B, ldb64); return;
    default: return;
  }
}

namespace internal::int8 {

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb)
  { tp_deq_dispatcher<0>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb)
  { tp_deq_dispatcher<0>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb)
  { tp_deq_dispatcher<0>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb)
  { tp_deq_dispatcher<0>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuDoubleComplex* B, int32_t ldb)
  { tp_deq_dispatcher<1>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuComplex* B, int32_t ldb)
  { tp_deq_dispatcher<1>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_double2* B, int32_t ldb)
  { tp_deq_dispatcher<1>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_float4* B, int32_t ldb)
  { tp_deq_dispatcher<1>(stream, orderA, N, A, umax, vec_expon, B, ldb); }

}
