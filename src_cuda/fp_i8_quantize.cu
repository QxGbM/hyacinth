
#include <hyacin.hpp>
#include <internal.hpp>
#include <int_fp_quantize.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <int32_t order, class real_t, class matrix_t> struct quantize_func {
  const matrix_t* __restrict__ A;
  const uint64_t* __restrict__ vec_expon;
  int8_t* __restrict__ B;
  int64_t M, lda, ldb, strideB;
  quantize_func(int64_t M, int64_t N, const matrix_t* A, int64_t lda, const uint64_t* vec_expon, int8_t* B, int64_t ldb) :
    A(A), vec_expon(vec_expon), B(B), M(M), lda(lda), ldb(ldb), strideB(int64_t(N) * int64_t(ldb)) {}

  __device__ __forceinline__ void operator()(int64_t i) {
    union { uint32_t code[3]; int8_t bytes[order]; } c;
    int64_t x = i / M, y = i - M * x;
    int32_t sgn = 0, expon = 0;
    device::int8::extract_scale(uint32_t(vec_expon[x]), sgn, expon); expon = -expon;
    int8_t* B_rl = &B[y + x * ldb];
    matrix_t A_i = A[y + x * lda];

    if constexpr(sizeof(real_t) < sizeof(matrix_t)) {
      union { uint32_t code[3]; int8_t bytes[order]; } c_im;
      int8_t* B_im = &B_rl[strideB * int64_t(order)];
      device::int8::quantize_double_align(double(A_i.x), expon, c.code);
      device::int8::quantize_double_align(double(A_i.y), expon, c_im.code);

      #pragma unroll
      for (int32_t k = 0; k < order; ++k) {
        B_rl[int64_t(k) * strideB] = c.bytes[k];
        B_im[int64_t(k) * strideB] = c_im.bytes[k];
      }
    }
    else {
      device::int8::quantize_double_align(double(A_i), expon, c.code);

      #pragma unroll
      for (int32_t k = 0; k < order; ++k)
        B_rl[int64_t(k) * strideB] = c.bytes[k];
    }
  }
};

template <class real_t, class matrix_t>
inline void quantize_dispatcher(cudaStream_t stream, int32_t order, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, const uint64_t* vec_expon, int8_t* A, int64_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  int64_t stride = M * N;

  switch (order) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<1, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<2, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<3, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<4, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<5, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<6, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<7, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<8, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 9: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<9, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 10: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<10, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 11: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<11, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    default: break;
  }
}

void internal::int8::quantize_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<double>(stream, order, M, N, C, ldc, vec_expon, A, lda);
}

void internal::int8::quantize_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<double>(stream, order, M, N, (double2*)C, ldc, vec_expon, A, lda);
}

void internal::int8::quantize_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<float>(stream, order, M, N, C, ldc, vec_expon, A, lda);
}

void internal::int8::quantize_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<float>(stream, order, M, N, (float2*)C, ldc, vec_expon, A, lda);
}
