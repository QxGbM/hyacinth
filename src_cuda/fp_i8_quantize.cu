
#include <hyacin.hpp>
#include <internal.hpp>
#include <int_fp_quantize.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <int32_t order, class real_t, class matrix_t> struct quantize_func {
  const matrix_t* __restrict__ A;
  const int32_t* __restrict__ vec_expon;
  int8_t* __restrict__ B;
  int64_t M, lda, ldb, strideB;
  quantize_func(int64_t M, int64_t N, const matrix_t* A, int64_t lda, const int32_t* vec_expon, int8_t* B, int64_t ldb) :
    A(A), vec_expon(vec_expon), B(B), M(M), lda(lda), ldb(ldb), strideB(int64_t(N) * int64_t(ldb)) {}

  __device__ __forceinline__ void operator()(int64_t i) {
    constexpr int32_t code_words = (order + 3) / 4;
    constexpr int32_t gemm_expon = order * device::Config::exp_base;
    union { uint32_t code[code_words]; int8_t bytes[order]; } c;
    int64_t x = i / M, y = i - M * x;
    int32_t expon = gemm_expon - vec_expon[x];
    int8_t* B_rl = &B[y + x * ldb];
    matrix_t A_i = A[y + x * lda];

    if constexpr(sizeof(real_t) < sizeof(matrix_t)) {
      union { uint32_t code[code_words]; int8_t bytes[order]; } c_im;
      int8_t* B_im = &B_rl[strideB * int64_t(order)];
      device::int8::quantize_double_align<device::Config::exp_base>(double(A_i.x), expon, c.code);
      device::int8::quantize_double_align<device::Config::exp_base>(double(A_i.y), expon, c_im.code);

      #pragma unroll
      for (int32_t k = 0; k < order; ++k) {
        B_rl[int64_t(k) * strideB] = c.bytes[k];
        B_im[int64_t(k) * strideB] = c_im.bytes[k];
      }
    }
    else {
      device::int8::quantize_double_align<device::Config::exp_base>(double(A_i), expon, c.code);

      #pragma unroll
      for (int32_t k = 0; k < order; ++k)
        B_rl[int64_t(k) * strideB] = c.bytes[k];
    }
  }
};

template <class matrix_t>
inline void quantize_dispatcher_f64(cudaStream_t stream, int32_t order, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, const int32_t* vec_expon, int8_t* A, int64_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  int64_t stride = M * N;

  switch (order) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<1, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<2, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<3, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<4, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<5, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<6, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<7, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<8, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 9: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<9, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 10: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<10, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    default: break;
  }

  if constexpr (device::Config::exp_base < 7)
    switch (order) {
      case 11: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<11, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 12: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<12, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 13: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<13, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      default: break;
    }

  if constexpr (device::Config::exp_base < 5)
    switch (order) {
      case 14: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<14, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 15: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<15, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 16: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<16, double, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      default: break;
    }
}

void internal::int8::quantize_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher_f64(stream, order, M, N, C, ldc, vec_expon, A, lda);
}

void internal::int8::quantize_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher_f64(stream, order, M, N, (double2*)C, ldc, vec_expon, A, lda);
}

template <class matrix_t>
inline void quantize_dispatcher_f32(cudaStream_t stream, int32_t order, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, const int32_t* vec_expon, int8_t* A, int64_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  int64_t stride = M * N;

  switch (order) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<1, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<2, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<3, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<4, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<5, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<6, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<7, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<8, float, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    default: break;
  }
}

void internal::int8::quantize_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher_f32(stream, order, M, N, C, ldc, vec_expon, A, lda);
}

void internal::int8::quantize_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher_f32(stream, order, M, N, (float2*)C, ldc, vec_expon, A, lda);
}
