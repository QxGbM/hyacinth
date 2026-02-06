
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, double& f) { f = device::dd::conv_a63_f64(a, e); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, float& f) { f = float(device::dd::conv_a63_f64(a, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, double2& f) { f = device::dd::conv_a63_dd(a, e); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t (&a)[ORDER], int32_t e, float4& f) { f = device::qf::conv_a63_qf(a, e); }

template<uint32_t orderA, uint32_t lSize, class real_t>
__global__ void dequantize_kernel(int64_t K, int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, int32_t umax, const int32_t* __restrict__ vec_expon, real_t* __restrict__ B, int64_t ldb) {
  int64_t y = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);

  if (y < N) {
    constexpr int32_t i63_limbs = int32_t(lSize == uint32_t(63));
    constexpr uint32_t orderL = i63_limbs ? orderA : (orderA + ((orderA + uint32_t(1)) >> 1));
    int64_t x = int64_t(blockIdx.y), iter = x + y * lda;

    uint64_t acc[orderA] { A[iter] };
    if constexpr(i63_limbs) {
      #pragma unroll
      for (uint32_t limb = 1; limb < orderA; ++limb)
        acc[limb] = A[iter += strideA];
    }
    else {
      #pragma unroll
      for (uint32_t limb = 1; limb < orderL; ++limb)
        device::int8::add_shifted(acc, int64_t(A[iter += strideA]), limb * lSize);
    }

    iter = y + x * lda;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc, int64_t(A[iter]), limb * lSize); iter += strideA; }

    iter = y + N * lda;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc, -int64_t(A[iter]), uint32_t(umax + 1) + limb * lSize); iter += strideA; }

    iter = x + N * lda;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc, -int64_t(A[iter]), uint32_t(umax + 1) + limb * lSize); iter += strideA; }

    device::int8::add_shifted(acc, K, uint32_t(umax = (umax << 1) | 1));
    int32_t expon = vec_expon[x] + vec_expon[y] - umax;
    fscal(acc, expon, B[y + x * ldb]);
  }
}

constexpr int32_t block_threads = 512;

template<uint32_t lSize, class real_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int64_t K, int64_t N, const uint64_t* A, int64_t lda, int32_t umax, const int32_t* vec_expon, real_t* B, int64_t ldb) {
  int64_t strideA = N * lda + lda;
  dim3 grid((uint32_t(N) + uint32_t(block_threads - 1)) / uint32_t(block_threads), uint32_t(N));

  switch (orderA) {
    case 1: dequantize_kernel<1, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    case 2: dequantize_kernel<2, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    case 3: dequantize_kernel<3, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    case 4: dequantize_kernel<4, lSize, real_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    default: break;
  }
}

void internal::int8::dequantize_i63_f64(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb) {
  dequantize_dispatcher<63>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i63_f32(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb) {
  dequantize_dispatcher<63>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i63_f128_dd(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb) {
  dequantize_dispatcher<63>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i63_f128_qf(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb) {
  dequantize_dispatcher<63>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i42_f64(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb) {
  dequantize_dispatcher<42>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i42_f32(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb) {
  dequantize_dispatcher<42>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i42_f128_dd(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb) {
  dequantize_dispatcher<42>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i42_f128_qf(cudaStream_t stream, int32_t orderA, int64_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb) {
  dequantize_dispatcher<42>(stream, orderA, K, int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}
