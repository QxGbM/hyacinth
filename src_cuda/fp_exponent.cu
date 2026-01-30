
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <cfloat>

struct f64max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
};

__device__ __forceinline__ double conv_abs(double a) { return fabs(a); }
__device__ __forceinline__ double conv_abs(float a) { return double(fabsf(a)); }

template <class real_t, class real_const_ptr, int32_t BLOCK_THREADS>
__global__ void vector_exponent_kernel(int32_t M, real_const_ptr A, int64_t lda, uint64_t* __restrict__ vec_expon) {
  __shared__ typename cub::BlockReduce<double, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<double, BLOCK_THREADS> block_reduce(temp_reduce);
  f64max max_func;

  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  double thread_data = 0.;

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS)
    thread_data = max_func(conv_abs(A_i[i]), thread_data);

  thread_data = block_reduce.Reduce(thread_data, max_func);
  if (threadIdx.x == 0) {
    int32_t emax; frexp(thread_data, &emax);
    vec_expon[blockIdx.x] = uint32_t(emax);
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint64_t* vec_expon) {
  vector_exponent_kernel<double, const double* __restrict__, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint64_t* vec_expon) {
  vector_exponent_kernel<float, const float* __restrict__, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, vec_expon);
}
