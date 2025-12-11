
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct minmax {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) {
    return make_double2(fmin(a.x, b.x), fmax(a.y, b.y));
  }
};

template <class real_const_ptr, int32_t BLOCK_THREADS, int32_t i8_correct>
__global__ void asq_kernel(int32_t M, real_const_ptr A, int64_t lda, uint32_t umax, uint64_t* __restrict__ scale, int32_t incv) {
  __shared__ typename cub::BlockReduce<double2, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<double2, BLOCK_THREADS> block_reduce(temp_reduce);
  minmax mm_func;

  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  double2 threadA = make_double2(DBL_MAX, -DBL_MAX);

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    double a = double(A_i[i]);
    threadA = mm_func(make_double2(a, a), threadA);
  }

  if (0 < M)
    threadA = block_reduce.Reduce(threadA, mm_func);

  if (threadIdx.x == 0) {
    if constexpr(i8_correct)
      device::int8::quant_bounds(threadA.x, threadA.y, umax, (umax + uint32_t(7)) & (~uint32_t(7)), scale[blockIdx.x], scale[int32_t(blockIdx.x) + incv]);
    else
      device::int8::quant_bounds(threadA.x, threadA.y, umax, uint32_t(0), scale[blockIdx.x], scale[int32_t(blockIdx.x) + incv]);
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::asq_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint32_t umax, uint64_t* vec_expon, int32_t incv) {
  asq_kernel <const double* __restrict__, block_threads, 0>
    <<< N, block_threads, 0, stream >>> (M, A, lda, umax, vec_expon, incv);
}

void internal::int8::asq_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint32_t umax, uint64_t* vec_expon, int32_t incv) {
  asq_kernel <const float* __restrict__, block_threads, 0>
    <<< N, block_threads, 0, stream >>> (M, A, lda, umax, vec_expon, incv);
}

void internal::int8::asq_f64_i8_correct(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint32_t umax, uint64_t* vec_expon, int32_t incv) {
  asq_kernel <const double* __restrict__, block_threads, 1>
    <<< N, block_threads, 0, stream >>> (M, A, lda, umax, vec_expon, incv);
}

void internal::int8::asq_f32_i8_correct(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint32_t umax, uint64_t* vec_expon, int32_t incv) {
  asq_kernel <const float* __restrict__, block_threads, 1>
    <<< N, block_threads, 0, stream >>> (M, A, lda, umax, vec_expon, incv);
}
