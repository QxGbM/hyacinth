
#include <hyacin.hpp>
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template <uint32_t ORDER, class matrix_t, class matrix_const_ptr, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void quantize_kernel(int64_t M, int64_t N, matrix_const_ptr A, int64_t lda, int32_t umax, const uint64_t* __restrict__ vec_expon, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  int64_t y = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  if (y < M) {
    int64_t x = int64_t(blockIdx.y);
    int32_t expon = -int32_t(vec_expon[x]);
    int8_t* B_i = &B[y + x * ldb];
    matrix_t A_i = A[y + x * lda];

    uint32_t code[3]; int8_t* bytes = (int8_t*)&code[0];
    if constexpr(COMPLEX)
      device::int8::quantize_f64_i8limbs(double(A_i.x), expon, umax, code);
    else
      device::int8::quantize_f64_i8limbs(double(A_i), expon, umax, code);

    int64_t iter = 0;
    #pragma unroll
    for (uint32_t k = 0; k < ORDER; ++k)
    { B_i[iter] = bytes[k]; iter += strideB; }

    if constexpr(COMPLEX) {
      device::int8::quantize_f64_i8limbs(double(A_i.y), expon, umax, code);
      #pragma unroll
      for (uint32_t k = 0; k < ORDER; ++k)
      { B_i[iter] = bytes[k]; iter += strideB; }
    }
  }
};

constexpr int32_t block_threads = 512;

template <class matrix_t, class matrix_const_ptr, int32_t COMPLEX>
inline void quantize_dispatcher(cudaStream_t stream, int32_t order, int64_t M, int64_t N, matrix_const_ptr C, int64_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int64_t lda) {
  int64_t strideA = N * lda;
  dim3 grid((uint32_t(M) + uint32_t(block_threads - 1)) / uint32_t(block_threads), uint32_t(N));

  switch (order) {
    case 1: quantize_kernel<1, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 2: quantize_kernel<2, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 3: quantize_kernel<3, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 4: quantize_kernel<4, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 5: quantize_kernel<5, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 6: quantize_kernel<6, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 7: quantize_kernel<7, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 8: quantize_kernel<8, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 9: quantize_kernel<9, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 10: quantize_kernel<10, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    case 11: quantize_kernel<11, matrix_t, matrix_const_ptr, COMPLEX, block_threads> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, lda, strideA); break;
    default: break;
  }
}

void internal::int8::quantize_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<double, const double* __restrict__, 0>(stream, order, M, N, C, ldc, umax, vec_expon, A, lda);
}

void internal::int8::quantize_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<double2, const double2* __restrict__, 1>(stream, order, M, N, (double2*)C, ldc, umax, vec_expon, A, lda);
}

void internal::int8::quantize_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<float, const float* __restrict__, 0>(stream, order, M, N, C, ldc, umax, vec_expon, A, lda);
}

void internal::int8::quantize_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<float2, const float2* __restrict__, 1>(stream, order, M, N, (float2*)C, ldc, umax, vec_expon, A, lda);
}
