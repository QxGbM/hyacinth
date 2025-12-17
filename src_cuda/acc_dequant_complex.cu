
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e - 1), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e - 1)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e - 1), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e - 1), device::qf::conv_a63_qf(im, e)); }

template<uint32_t orderA, class complex_t, int32_t BLOCK_THREADS>
__global__ void dequantize_complex_kernel(int64_t M, int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, int32_t umax, const int64_t* __restrict__ vec_expon, int64_t incv, complex_t* __restrict__ B, int64_t ldb) {
  int64_t y = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);

  if (y < N) {
    int64_t x = int64_t(blockIdx.y), iter = x + y * lda;
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

    device::int8::add_shifted(acc_rl, vec_expon[iter = y + incv], uint32_t(umax + 1));
    device::int8::add_shifted(acc_rl, vec_expon[iter += incv], uint32_t(umax + 64));
    device::int8::add_shifted(acc_im, -vec_expon[iter += incv], uint32_t(umax));
    device::int8::add_shifted(acc_im, -vec_expon[iter += incv], uint32_t(umax + 63));

    device::int8::add_shifted(acc_rl, vec_expon[iter = x + incv], uint32_t(umax + 1));
    device::int8::add_shifted(acc_rl, vec_expon[iter += incv], uint32_t(umax + 64));
    device::int8::add_shifted(acc_rl, M, uint32_t((umax << 1) | 1));
    device::int8::add_shifted(acc_im, vec_expon[iter += incv], uint32_t(umax));
    device::int8::add_shifted(acc_im, vec_expon[iter += incv], uint32_t(umax + 63));

    int32_t expon = int32_t(vec_expon[x]) + int32_t(vec_expon[y]);
    cscal(acc_rl, acc_im, expon, B[y + x * ldb]);
  }
}

constexpr int32_t block_threads = 512;

template<class complex_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int64_t M, int64_t N, const uint64_t* A, int64_t lda, int32_t umax, const int64_t* vec_expon, int64_t incv, complex_t* B, int64_t ldb) {
  int64_t strideA = N * lda;
  dim3 grid((uint32_t(N) + uint32_t(block_threads - 1)) / uint32_t(block_threads), uint32_t(N));

  switch (orderA) {
    case 1: dequantize_complex_kernel<1, complex_t, block_threads> <<< grid, block_threads, 0, stream >>> (-M, N, A, lda, strideA, umax, vec_expon, incv, B, ldb); break;
    case 2: dequantize_complex_kernel<2, complex_t, block_threads> <<< grid, block_threads, 0, stream >>> (-M, N, A, lda, strideA, umax, vec_expon, incv, B, ldb); break;
    case 3: dequantize_complex_kernel<3, complex_t, block_threads> <<< grid, block_threads, 0, stream >>> (-M, N, A, lda, strideA, umax, vec_expon, incv, B, ldb); break;
    case 4: dequantize_complex_kernel<4, complex_t, block_threads> <<< grid, block_threads, 0, stream >>> (-M, N, A, lda, strideA, umax, vec_expon, incv, B, ldb); break;
    default: break;
  }
}

void internal::int8::dequantize_cf64(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, std::complex<double>* B, int32_t ldb) {
  dequantize_dispatcher(stream, orderA, M, N, A, lda, umax, (const int64_t*)vec_expon, incv, (cuDoubleComplex*)B, ldb);
}

void internal::int8::dequantize_cf32(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, std::complex<float>* B, int32_t ldb) {
  dequantize_dispatcher(stream, orderA, M, N, A, lda, umax, (const int64_t*)vec_expon, incv, (cuComplex*)B, ldb);
}

void internal::int8::dequantize_cf128_dd(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, complex_double2* B, int32_t ldb) {
  dequantize_dispatcher(stream, orderA, M, N, A, lda, umax, (const int64_t*)vec_expon, incv, B, ldb);
}

void internal::int8::dequantize_cf128_qf(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, complex_float4* B, int32_t ldb) {
  dequantize_dispatcher(stream, orderA, M, N, A, lda, umax, (const int64_t*)vec_expon, incv, B, ldb);
}
