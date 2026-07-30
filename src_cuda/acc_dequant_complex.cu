
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <limits>

constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e), device::qf::conv_a63_qf(im, e)); }

template<int32_t orderA, int32_t LimbCount, int32_t LimbSpace, class complex_t>
__global__ void dequantize_complex_kernel(int64_t K, int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int32_t umax, const int32_t* __restrict__ vec_expon, complex_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);

  if (y < N) {
    constexpr uint32_t shifts[]{ uint32_t(0), uint32_t(LimbSpace), uint32_t(LimbSpace * 2), uint32_t(LimbSpace * 3), uint32_t(LimbSpace * 4), uint32_t(LimbSpace * 5) };
    int64_t x = int64_t(blockIdx.y), iter = x < y ? (x + y * N) : (y + x * N);

    uint64_t acc_rl[orderA] { A[iter] };
    if constexpr(orderA == LimbCount) {
      #pragma unroll
      for (int32_t limb = 1; limb < orderA; ++limb)
        acc_rl[limb] = A[iter += strideA];
    }
    else {
      #pragma unroll
      for (int32_t limb = 1; limb < LimbCount; ++limb)
        device::int8::add_shifted(acc_rl, int64_t(A[iter += strideA]), shifts[limb]);
    }

    iter = y + x * N + strideA * int64_t(LimbCount);
    uint64_t acc_im[orderA] {};
    #pragma unroll
    for (int32_t limb = 0; limb < LimbCount; ++limb)
    { device::int8::add_shifted(acc_im, int64_t(A[iter]), shifts[limb]); iter += strideA; }

    iter = x + y * N + strideA * int64_t(LimbCount);
    #pragma unroll
    for (int32_t limb = 0; limb < LimbCount; ++limb)
    { device::int8::add_shifted(acc_im, -int64_t(A[iter]), shifts[limb]); iter += strideA; }

    iter = strideA + y - N;
    #pragma unroll
    for (int32_t limb = 0; limb < LimbCount; ++limb) {
      int64_t kz_rl = -int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, kz_rl, uint32_t(umax) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_rl, uint32_t(umax) + shifts[limb]);
    }

    #pragma unroll
    for (int32_t limb = 0; limb < LimbCount; ++limb) {
      int64_t kz_im = int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, -kz_im, uint32_t(umax) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_im, uint32_t(umax) + shifts[limb]);
    }

    iter = strideA + x - N;
    #pragma unroll
    for (int32_t limb = 0; limb < LimbCount; ++limb) {
      int64_t kz_rl = int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, -kz_rl, uint32_t(umax) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_rl, uint32_t(umax) + shifts[limb]);
    }

    #pragma unroll
    for (int32_t limb = 0; limb < LimbCount; ++limb) {
      int64_t kz_im = -int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, kz_im, uint32_t(umax) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_im, uint32_t(umax) + shifts[limb]);
    }

    device::int8::add_shifted(acc_rl, K, uint32_t(umax <<= 1));
    int32_t ex = vec_expon[x], ey = vec_expon[y]; iter = y + x * ldb;
    if (ex == int_min || ey == int_min) B[iter] = complex_t();
      else cscal(acc_rl, acc_im, ex + ey - umax, B[iter]);
  }
}

template<class complex_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int64_t K, int64_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_t* B, int64_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideA = N * N + N; K <<= 1;

  if (orderA == 1) switch (LimbCount) {
    case 1: dequantize_complex_kernel<1, 1, 63> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 2: dequantize_complex_kernel<1, 2, 32> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    default: return;
  } else if (orderA == 2) switch(LimbCount) {
    case 2: dequantize_complex_kernel<2, 2, 63> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 3: dequantize_complex_kernel<2, 3, 43> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 4: dequantize_complex_kernel<2, 4, 32> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    default: return;
  } else if (orderA == 3) switch(LimbCount) {
    case 3: dequantize_complex_kernel<3, 3, 63> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 4: dequantize_complex_kernel<3, 4, 48> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 5: dequantize_complex_kernel<3, 5, 38> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 6: dequantize_complex_kernel<3, 6, 32> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    default: return;
  }
}

namespace internal::int8 {

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuDoubleComplex* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, LimbCount, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuComplex* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, LimbCount, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_double2* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, LimbCount, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize_complex(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_float4* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, LimbCount, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

}
