
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct f64max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
};

__device__ __forceinline__ double conv_abs(double a) { return fabs(a); }
__device__ __forceinline__ double conv_abs(float a) { return double(fabsf(a)); }

template <class real_t, int32_t BLOCK_THREADS>
__global__ void vector_exponent_kernel(int64_t M, const real_t* __restrict__ A, int64_t lda, int32_t* __restrict__ vec_expon) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  __shared__ typename cub::BlockReduce<double, BLOCK_THREADS>::TempStorage temp_reduce;
  f64max max_func;

  A = &A[int64_t(blockIdx.x) * lda];
  double thread_data = 0.;

  for (int64_t i = threadIdx.x; i < M; i += inci)
    thread_data = max_func(conv_abs(A[i]), thread_data);

  thread_data = cub::BlockReduce<double, BLOCK_THREADS>(temp_reduce).Reduce(thread_data, max_func);
  if (threadIdx.x == 0) { 
    if (thread_data == 0.) vec_expon[blockIdx.x] = int32_t(device::int8::u31);
      else frexp(thread_data, &vec_expon[blockIdx.x]);
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<double, block_threads> <<< N, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<float, block_threads> <<< N, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), vec_expon);
}

void internal::int8::vexp_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<double, block_threads> <<< N, block_threads, 0, stream >>> (int64_t(M) << 1, (const double*)A, int64_t(lda) << 1, vec_expon);
}

void internal::int8::vexp_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<float, block_threads> <<< N, block_threads, 0, stream >>> (int64_t(M) << 1, (const float*)A, int64_t(lda) << 1, vec_expon);
}
