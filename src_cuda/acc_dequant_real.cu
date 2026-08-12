
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

template<int32_t orderA, class real_t>
__global__ void dequantize_kernel(int64_t K, int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int32_t umax, const int32_t* __restrict__ vec_expon, real_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);

  if (y <= x) {
    constexpr uint32_t shifts[]{ uint32_t(0), uint32_t(63), uint32_t(126) };
    int64_t iter = y + x * N;

    uint64_t acc[orderA];
    #pragma unroll
    for (int32_t i = 0; i < orderA; ++i)
    { acc[i] = A[iter]; iter += strideA; }

    if (K) {
      iter = (strideA * int64_t(orderA)) + y;
      #pragma unroll
      for (int32_t i = 0; i < orderA; ++i)
      { device::int8::add_shifted(acc, int64_t(A[iter]), uint32_t(umax) + shifts[i]); iter += N; }

      iter = (strideA * int64_t(orderA)) + x;
      #pragma unroll
      for (int32_t i = 0; i < orderA; ++i)
      { device::int8::add_shifted(acc, int64_t(A[iter]), uint32_t(umax) + shifts[i]); iter += N; }

      device::int8::add_shifted(acc, K, uint32_t(umax <<= 1));
    }

    int32_t ex = vec_expon[x], ey = vec_expon[y]; iter = y + x * ldb;
    if (ex == int_min || ey == int_min) B[iter] = real_t();
      else fscal(acc, ex + ey - umax, B[iter]);
  }
}

template<class real_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int64_t K, int64_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, real_t* B, int64_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideA = N * N;
  switch(orderA) {
    case 1: dequantize_kernel<1> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 2: dequantize_kernel<2> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    case 3: dequantize_kernel<3> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax, vec_expon, B, ldb); return;
    default: return;
  }
}

namespace internal::int8 {

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

  void dequantize(cudaStream_t stream, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb) {
    dequantize_dispatcher(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  }

}
