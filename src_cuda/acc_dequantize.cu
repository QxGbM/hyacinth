
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], uint32_t e, double& f) { f = device::dd::conv_a63_f64(a, e); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], uint32_t e, float& f) { f = float(device::dd::conv_a63_f64(a, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], uint32_t e, double2& f) { f = device::dd::conv_a63_dd(a, e); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], uint32_t e, float4& f) { f = device::qf::conv_a63_qf(a, e); }

template<uint32_t orderA, class real_t, int32_t BLOCK_THREADS>
__global__ void dequantize_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, const int32_t* __restrict__ vec_expon, real_t* __restrict__ B, int64_t ldb) {
  int64_t i = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  int64_t x = i / N, y = i - N * x;

  if (x < N && y < N) {
    int64_t iter = y + x * lda;
    uint64_t acc[orderA];
    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { acc[r] = A[iter]; iter += strideA; }

    int32_t expon = vec_expon[x] + vec_expon[y];
    int32_t sgn = 0;
    fscal(acc, uint32_t((sgn << 31) | (expon & 0x7fffffff)), B[y + x * ldb]);
  }
}

template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], uint32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], uint32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], uint32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], uint32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e), device::qf::conv_a63_qf(im, e)); }

template<uint32_t orderA, class complex_t, int32_t BLOCK_THREADS>
__global__ void dequantize_complex_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, const int32_t* __restrict__ vec_expon, complex_t* __restrict__ B, int64_t ldb) {
  int64_t i = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  int64_t x = i / N, y = i - N * x;

  if (x < N && y < N) {
    int64_t iter = y + x * lda, imT = (x - y) + (y - x) * lda + int64_t(orderA) * strideA;
    uint64_t acc_rl[orderA], acc_im[orderA];

    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { acc_rl[r] = A[iter]; acc_im[r] = A[iter + imT]; iter += strideA; }
    device::int8::negate_shifted(acc_im);

    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { device::int8::add_shifted(acc_im, int64_t(A[iter]), r * uint32_t(63)); iter += strideA; }

    int32_t expon = vec_expon[x] + vec_expon[y];
    int32_t sgn = 0;
    cscal(acc_rl, acc_im, uint32_t((sgn << 31) | (expon & 0x7fffffff)), B[y + x * ldb]);
  }
}

constexpr int32_t block_threads = 512;

template<int32_t COMPLEX, class matrix_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, matrix_t* B, int32_t ldb) {
  int64_t len = int64_t(N) * int64_t(N), strideA = int64_t(N) * int64_t(lda);
  int32_t grid = int32_t((len + int64_t(block_threads - 1)) / int64_t(block_threads));

  if constexpr(COMPLEX)
    switch (orderA) {
      case 1: dequantize_complex_kernel<1, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      case 2: dequantize_complex_kernel<2, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      case 3: dequantize_complex_kernel<3, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      case 4: dequantize_complex_kernel<4, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      default: break;
    }
  else
    switch (orderA) {
      case 1: dequantize_kernel<1, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      case 2: dequantize_kernel<2, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      case 3: dequantize_kernel<3, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      case 4: dequantize_kernel<4, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, B, ldb); break;
      default: break;
    }
}

void internal::int8::dequantize_f64(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, double* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, B, ldb);
}

void internal::int8::dequantize_f32(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, float* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, B, ldb);
}

void internal::int8::dequantize_f128_dd(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, double2* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, B, ldb);
}

void internal::int8::dequantize_f128_qf(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, float4* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, B, ldb);
}

void internal::int8::dequantize_cf64(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, std::complex<double>* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, (cuDoubleComplex*)B, ldb);
}

void internal::int8::dequantize_cf32(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, std::complex<float>* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, (cuComplex*)B, ldb);
}

void internal::int8::dequantize_cf128_dd(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, complex_double2* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, B, ldb);
}

void internal::int8::dequantize_cf128_qf(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const int32_t* vec_expon, complex_float4* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, B, ldb);
}
