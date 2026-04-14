
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <limits>

constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuDoubleComplex& f) {
  f = make_cuDoubleComplex(device::dd::conv_a63_f64(rl, e - 1), device::dd::conv_a63_f64(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, cuComplex& f) {
  f = make_cuComplex(float(device::dd::conv_a63_f64(rl, e - 1)), float(device::dd::conv_a63_f64(im, e))); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_double2& f) {
  f = device::dd::make_complex_double2(device::dd::conv_a63_dd(rl, e - 1), device::dd::conv_a63_dd(im, e)); }
template<uint32_t ORDER> __device__ __forceinline__ void cscal(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER], int32_t e, complex_float4& f) {
  f = device::qf::make_complex_float4(device::qf::conv_a63_qf(rl, e - 1), device::qf::conv_a63_qf(im, e)); }

template<uint32_t orderA, uint32_t lSize, class complex_t>
__global__ void dequantize_complex_kernel(int64_t K, int64_t N, const uint64_t* __restrict__ A, int64_t lda, int64_t strideA, int32_t umax, const int32_t* __restrict__ vec_expon, complex_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);

  if (y < N) {
    constexpr uint32_t orderL = orderA + uint32_t(lSize < uint32_t(63));
    constexpr uint32_t shifts[4] = { uint32_t(0), lSize, lSize * uint32_t(2), lSize * uint32_t(3) };
    int64_t x = int64_t(blockIdx.y), iter = x + y * lda;

    uint64_t acc_rl[orderA] { A[iter] };
    if constexpr(orderA == orderL) {
      #pragma unroll
      for (uint32_t limb = 1; limb < orderA; ++limb)
        acc_rl[limb] = A[iter += strideA];
    }
    else {
      #pragma unroll
      for (uint32_t limb = 1; limb < orderL; ++limb)
        device::int8::add_shifted(acc_rl, int64_t(A[iter += strideA]), shifts[limb]);
    }

    uint64_t acc_im[orderA] {};
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
      device::int8::add_shifted(acc_im, -int64_t(A[iter += strideA]), shifts[limb]);
    iter = y + x * lda;

    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc_rl, int64_t(A[iter]), shifts[limb]); iter += strideA; }

    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb)
    { device::int8::add_shifted(acc_im, int64_t(A[iter]), shifts[limb]); iter += strideA; }

    iter = strideA + y - lda;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb) {
      int64_t kz_rl = -int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, kz_rl, uint32_t(umax + 1) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_rl, uint32_t(umax) + shifts[limb]);
    }

    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb) {
      int64_t kz_im = int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, -kz_im, uint32_t(umax + 1) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_im, uint32_t(umax) + shifts[limb]);
    }

    iter = strideA + x - lda;
    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb) {
      int64_t kz_rl = int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, -kz_rl, uint32_t(umax + 1) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_rl, uint32_t(umax) + shifts[limb]);
    }

    #pragma unroll
    for (uint32_t limb = 0; limb < orderL; ++limb) {
      int64_t kz_im = -int64_t(A[iter]); iter += strideA;
      device::int8::add_shifted(acc_rl, kz_im, uint32_t(umax + 1) + shifts[limb]);
      device::int8::add_shifted(acc_im, kz_im, uint32_t(umax) + shifts[limb]);
    }

    device::int8::add_shifted(acc_rl, K, uint32_t((umax <<= 1) + 2));
    int32_t ex = vec_expon[x], ey = vec_expon[y]; iter = y + x * ldb;
    if (ex == int_min || ey == int_min) B[iter] = complex_t();
      else cscal(acc_rl, acc_im, ex + ey - umax, B[iter]);
  }
}

template<uint32_t lSize, class complex_t>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t orderA, int64_t K, int64_t N, const uint64_t* A, int64_t lda, int32_t umax, const int32_t* vec_expon, complex_t* B, int64_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideA = N * lda + lda;

  switch (orderA) {
    case 1: dequantize_complex_kernel<1, lSize, complex_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    case 2: dequantize_complex_kernel<2, lSize, complex_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    case 3: dequantize_complex_kernel<3, lSize, complex_t> <<< grid, block_threads, 0, stream >>> (K, N, A, lda, strideA, umax, vec_expon, B, ldb); break;
    default: break;
  }
}

void internal::int8::dequantize_i63_cf64(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, std::complex<double>* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, (cuDoubleComplex*)B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, (cuDoubleComplex*)B, int64_t(ldb));
}

void internal::int8::dequantize_i63_cf32(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, std::complex<float>* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, (cuComplex*)B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, (cuComplex*)B, int64_t(ldb));
}

void internal::int8::dequantize_i63_cf128_dd(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, complex_double2* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}

void internal::int8::dequantize_i63_cf128_qf(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const int32_t* vec_expon, complex_float4* B, int32_t ldb) {
  if (bits == 63)
    dequantize_dispatcher<63>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
  else if (bits == 47)
    dequantize_dispatcher<47>(stream, orderA, int64_t(K), int64_t(N), A, int64_t(lda), umax, vec_expon, B, int64_t(ldb));
}
