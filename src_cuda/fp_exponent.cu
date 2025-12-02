
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct abs_max {
  __device__ __forceinline__ double operator()(double a) { return fabs(a); }
  __device__ __forceinline__ double operator()(float a) { return double(fabsf(a)); }
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
};

template <class real_t, class real_const_ptr, int32_t BLOCK_THREADS>
__global__ void vector_exponent_kernel(int32_t M, real_const_ptr A, int64_t lda, int32_t umax, int32_t* __restrict__ vec_expon) {
  __shared__ typename cub::BlockReduce<double, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<double, BLOCK_THREADS> block_reduce(temp_reduce);
  abs_max amax_func;

  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  double threadB = 0.;

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS)
    threadB = amax_func(amax_func(A_i[i]), threadB);

  if (0 < M)
    threadB = block_reduce.Reduce(threadB, amax_func);

  if (threadIdx.x == 0) {
    int32_t e; frexp(threadB, &e);
    vec_expon[blockIdx.x] = umax + e;
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, int32_t* vec_expon) {
  vector_exponent_kernel <double, const double* __restrict__, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, -umax, vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, int32_t* vec_expon) {
  vector_exponent_kernel <float, const float* __restrict__, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, -umax, vec_expon);
}
