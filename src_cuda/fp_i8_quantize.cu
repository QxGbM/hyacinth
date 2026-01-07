
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_selector.hpp>

__device__ __forceinline__ void quantize_f64_i8limbs(double x, int32_t expon, int32_t umax, uint32_t (&code)[3]) {
  uint32_t c = 0; device::int8::quantize_f64_u32limbs(x, expon, umax, code);
  device::int8::conv_u8i8(code[0], c); device::int8::conv_u8i8(code[1], c); device::int8::conv_u8i8(code[2], c);
}

template <uint64_t MO, uint64_t R32>
__device__ __forceinline__ uint64_t quantize_f64_i8rems(double x, int32_t expon, int32_t umax, uint32_t (&code)[3]) {
  device::int8::quantize_f64_u32limbs(x, expon, umax, code);
  return device::int8::conv_u32i8_modular<MO, R32>(code);
}

template <uint32_t ORDER, uint64_t MO, uint64_t R32, class matrix_t, class matrix_const_ptr, int32_t op>
__global__ void quantize_kernel(int64_t M, int64_t N, matrix_const_ptr A, int64_t lda, int32_t umax, const uint64_t* __restrict__ vec_expon, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  int64_t y = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (y < M) {
    int64_t x = int64_t(blockIdx.y);
    int32_t expon = -int32_t(vec_expon[x]);
    int8_t* B_i = &B[y + x * ldb];
    matrix_t A_i = A[y + x * lda];

    uint32_t code[3]; uint64_t r; int8_t* bytes;
    if constexpr(op & 2) bytes = (int8_t*)&r; else bytes = (int8_t*)&code[0]; 

    if constexpr(op == 0)
      quantize_f64_i8limbs(double(A_i), expon, umax, code);
    else if constexpr(op == 1)
      quantize_f64_i8limbs(double(A_i.x), expon, umax, code);
    else if constexpr(op == 2)
      r = quantize_f64_i8rems<MO, R32>(double(A_i), expon, umax, code);
    else if constexpr(op == 3)
      r = quantize_f64_i8rems<MO, R32>(double(A_i.x), expon, umax, code);

    int64_t iter = 0;
    #pragma unroll
    for (uint32_t k = 0; k < ORDER; ++k)
    { B_i[iter] = bytes[k]; iter += strideB; }

    if constexpr(op & 1) {
      if constexpr(op == 1)
        quantize_f64_i8limbs(double(A_i.y), expon, umax, code);
      else if constexpr(op == 3)
        r = quantize_f64_i8rems<MO, R32>(double(A_i.y), expon, umax, code);

      #pragma unroll
      for (uint32_t k = 0; k < ORDER; ++k)
      { B_i[iter] = bytes[k]; iter += strideB; }
    }
  }
};

constexpr int32_t block_threads = 512;

template <class matrix_t, class matrix_const_ptr, int32_t iter, int32_t op>
inline void quantize_dispatcher(cudaStream_t stream, int64_t M, int64_t N, matrix_const_ptr C, int64_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int64_t lda) {
  constexpr uint64_t MO = CRT::modular(iter), R32 = CRT::rem_e32(iter);
  dim3 grid((uint32_t(M) + uint32_t(block_threads - 1)) / uint32_t(block_threads), uint32_t(N));
  int64_t strideA = N * lda;

  switch (orderA) {
    case 1: quantize_kernel<1, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 2: quantize_kernel<2, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 3: quantize_kernel<3, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 4: quantize_kernel<4, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 5: quantize_kernel<5, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 6: quantize_kernel<6, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 7: quantize_kernel<7, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 8: quantize_kernel<8, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 9: quantize_kernel<9, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 10: quantize_kernel<10, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 11: quantize_kernel<11, MO, R32, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    default: break;
  }
}

template <class matrix_t, class matrix_const_ptr, int32_t op>
inline void quantize_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, matrix_const_ptr C, int64_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int64_t lda) {
  switch (iter) {
    case 0: quantize_dispatcher<matrix_t, matrix_const_ptr, 0, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    case 1: quantize_dispatcher<matrix_t, matrix_const_ptr, 1, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    case 2: quantize_dispatcher<matrix_t, matrix_const_ptr, 2, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    case 3: quantize_dispatcher<matrix_t, matrix_const_ptr, 3, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    default: break;
  }
}

void internal::int8::quantize_f64(cudaStream_t stream, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double, const double* __restrict__, -1, 0>(stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double2, const double2* __restrict__, -1, 1>(stream, M, N, (double2*)C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_f32(cudaStream_t stream, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float, const float* __restrict__, -1, 0>(stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float2, const float2* __restrict__, -1, 1>(stream, M, N, (float2*)C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_f64_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double, const double* __restrict__, 2>(stream, M, N, iter, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf64_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double2, const double2* __restrict__, 3>(stream, M, N, iter, (double2*)C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_f32_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float, const float* __restrict__, 2>(stream, M, N, iter, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf32_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float2, const float2* __restrict__, 3>(stream, M, N, iter, (float2*)C, ldc, umax, vec_expon, orderA, A, lda);
}
