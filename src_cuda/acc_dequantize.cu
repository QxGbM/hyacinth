
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], int32_t e, double& f) { f = device::dd::conv_a63_f64(a, e - 1); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], int32_t e, float& f) { f = float(device::dd::conv_a63_f64(a, e - 1)); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], int32_t e, double2& f) { f = device::dd::conv_a63_dd(a, e - 1); }
template<uint32_t ORDER> __device__ __forceinline__ void fscal(uint64_t const (&a)[ORDER], int32_t e, float4& f) { f = device::qf::conv_a63_qf(a, e - 1); }

template<uint32_t orderA, class real_t, int32_t BLOCK_THREADS>
__global__ void dequantize_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, const uint64_t* __restrict__ vec_expon, int64_t incv, real_t* __restrict__ B, int64_t ldb) {
  int64_t i = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  int64_t x = i / N, y = i - N * x;

  if (x < N && y < N) {
    int64_t iter = x + y * lda;
    uint64_t acc[orderA];
    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { acc[r] = A[iter]; iter += strideA; }

    iter = y + x * lda;
    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { device::int8::add_shifted(acc, int64_t(A[iter]), r * uint32_t(63)); iter += strideA; }

    uint32_t sft_zx = uint32_t(vec_expon[x] >> 32) & device::int8::i31;
    int64_t zx = int64_t(vec_expon[x + incv]);
    device::int8::ima_shifted(acc, zx, vec_expon[y + (incv << 1)], sft_zx);
    device::int8::ima_shifted(acc, zx, vec_expon[(y + incv) + (incv << 1)], sft_zx + uint32_t(63));

    uint32_t sft_zy = uint32_t(vec_expon[y] >> 32) & device::int8::i31;
    int64_t zy = int64_t(vec_expon[y + incv]);
    device::int8::ima_shifted(acc, zy, vec_expon[y + (incv << 2)], sft_zy);
    device::int8::ima_shifted(acc, zy, vec_expon[(y + incv) + (incv << 2)], sft_zy + uint32_t(63));

    int32_t expon = int32_t(vec_expon[x]) + int32_t(vec_expon[y]);
    fscal(acc, expon, B[y + x * ldb]);
  }
}

template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], int32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e - 1), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], int32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e - 1)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], int32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e - 1), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t const (&rl)[ORDER], uint64_t const (&im)[ORDER], int32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e - 1), device::qf::conv_a63_qf(im, e)); }

template<uint32_t orderA, class complex_t, int32_t BLOCK_THREADS>
__global__ void dequantize_complex_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, const uint64_t* __restrict__ vec_expon, int64_t incv, complex_t* __restrict__ B, int64_t ldb) {
  int64_t i = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  int64_t x = i / N, y = i - N * x;

  if (x < N && y < N) {
    int64_t iter = x + y * lda;
    uint64_t acc_rl[orderA], acc_im[orderA];

    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { acc_rl[r] = A[iter]; iter += strideA; }

    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { acc_im[r] = A[iter]; iter += strideA; }
    device::int8::negate_shifted(acc_im);

    iter = y + x * lda;
    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { device::int8::add_shifted(acc_rl, int64_t(A[iter]), r * uint32_t(63)); iter += strideA; }

    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { device::int8::add_shifted(acc_im, int64_t(A[iter]), r * uint32_t(63)); iter += strideA; }

    uint32_t sft_zx = uint32_t(vec_expon[x] >> 32) & device::int8::i31;
    int64_t zx = int64_t(vec_expon[x + incv]), incv3x = incv + (incv << 1);
    device::int8::ima_shifted(acc_rl, zx, vec_expon[y + (incv << 1)], sft_zx);
    device::int8::ima_shifted(acc_rl, zx, vec_expon[(y + incv) + (incv << 1)], sft_zx + uint32_t(63));
    device::int8::ima_shifted(acc_im, zx, vec_expon[y + (incv3x << 1)], sft_zx);
    device::int8::ima_shifted(acc_im, zx, vec_expon[(y + incv) + (incv3x << 1)], sft_zx + uint32_t(63));

    uint32_t sft_zy = uint32_t(vec_expon[y] >> 32) & device::int8::i31;
    int64_t zy = int64_t(vec_expon[y + incv]);
    device::int8::ima_shifted(acc_rl, zy, vec_expon[y + (incv << 2)], sft_zy);
    device::int8::ima_shifted(acc_rl, zy, vec_expon[(y + incv) + (incv << 2)], sft_zy + uint32_t(63));
    device::int8::ima_shifted(acc_im, -zy, vec_expon[y + (incv3x << 1)], sft_zx);
    device::int8::ima_shifted(acc_im, -zy, vec_expon[(y + incv) + (incv3x << 1)], sft_zx + uint32_t(63));

    int32_t expon = int32_t(vec_expon[x]) + int32_t(vec_expon[y]);
    cscal(acc_rl, acc_im, expon, B[y + x * ldb]);
  }
}

constexpr int32_t block_threads = 512;

template<int32_t COMPLEX, class matrix_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int64_t incv, matrix_t* B, int32_t ldb) {
  int64_t len = int64_t(N) * int64_t(N), strideA = int64_t(N) * int64_t(lda);
  int32_t grid = int32_t((len + int64_t(block_threads - 1)) / int64_t(block_threads));

  if constexpr(COMPLEX)
    switch (orderA) {
      case 1: dequantize_complex_kernel<1, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      case 2: dequantize_complex_kernel<2, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      case 3: dequantize_complex_kernel<3, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      case 4: dequantize_complex_kernel<4, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      default: break;
    }
  else
    switch (orderA) {
      case 1: dequantize_kernel<1, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      case 2: dequantize_kernel<2, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      case 3: dequantize_kernel<3, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      case 4: dequantize_kernel<4, matrix_t, block_threads> <<< grid, block_threads, 0, stream >>> (N, A, lda, strideA, vec_expon, incv, B, ldb); break;
      default: break;
    }
}

void internal::int8::dequantize_f64(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, double* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), B, ldb);
}

void internal::int8::dequantize_f32(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, float* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), B, ldb);
}

void internal::int8::dequantize_f128_dd(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, double2* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), B, ldb);
}

void internal::int8::dequantize_f128_qf(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, float4* B, int32_t ldb) {
  dequantize_dispatcher<0>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), B, ldb);
}

void internal::int8::dequantize_cf64(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, std::complex<double>* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), (cuDoubleComplex*)B, ldb);
}

void internal::int8::dequantize_cf32(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, std::complex<float>* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), (cuComplex*)B, ldb);
}

void internal::int8::dequantize_cf128_dd(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, complex_double2* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), B, ldb);
}

void internal::int8::dequantize_cf128_qf(cudaStream_t stream, int32_t orderA, int32_t N, const uint64_t* A, int32_t lda, const uint64_t* vec_expon, int32_t incv, complex_float4* B, int32_t ldb) {
  dequantize_dispatcher<1>(stream, orderA, N, A, lda, vec_expon, int64_t(incv), B, ldb);
}
