
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

template<uint32_t orderA, uint32_t lSize, class real_t>
__global__ void dequantize_kernel(int64_t K, int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int32_t umax, const int32_t* __restrict__ vec_expon, real_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);

  if (y < N) {
    constexpr uint32_t orderL = orderA + uint32_t(lSize < uint32_t(63));
    constexpr uint32_t shifts[4] = { uint32_t(0), lSize, lSize * uint32_t(2), lSize * uint32_t(3) };
    int64_t x = int64_t(blockIdx.y), iter = x + y * N;

    uint64_t acc[orderA] { A[iter] };
    if constexpr(orderA == orderL) {
      #pragma unroll
      for (uint32_t limb = 1; limb < orderA; ++limb)
        acc[limb] = A[iter += strideA];
    }
    else {
      #pragma unroll
      for (uint32_t limb = 1; limb < orderL; ++limb)
        device::int8::add_shifted(acc, int64_t(A[iter += strideA]), shifts[limb]);
    }

    iter = y + x * N;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc, int64_t(A[iter]), shifts[limb]); iter += strideA; }

    iter = strideA + y - N;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc, -int64_t(A[iter]), uint32_t(umax) + shifts[limb]); iter += strideA; }

    iter = strideA + x - N;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc, -int64_t(A[iter]), uint32_t(umax) + shifts[limb]); iter += strideA; }

    device::int8::add_shifted(acc, K, uint32_t(umax = (umax << 1) - 1));
    int32_t ex = vec_expon[x], ey = vec_expon[y]; iter = y + x * ldb;
    if (ex == int_min || ey == int_min) B[iter] = real_t();
      else fscal(acc, ex + ey - umax, B[iter]);
  }
}

template<uint32_t lSize, class real_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int64_t K, int64_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, real_t* B, int64_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideA = N * N + N;

  switch (orderA) {
    case 1: dequantize_kernel<1, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax + 1, vec_expon, B, ldb); break;
    case 2: dequantize_kernel<2, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax + 1, vec_expon, B, ldb); break;
    case 3: dequantize_kernel<3, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, strideA, umax + 1, vec_expon, B, ldb); break;
    default: break;
  }
}

void internal::int8::dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, umax, vec_expon, B, int64_t(ldb));
}
