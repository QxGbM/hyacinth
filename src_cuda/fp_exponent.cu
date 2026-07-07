
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <limits>

constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
struct float_max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
};
__device__ __forceinline__ double float_abs(double a) { return fabs(a); }
__device__ __forceinline__ float float_abs(float a) { return fabsf(a); }
__device__ __forceinline__ float float_abs(half a) { return __half2float(__habs(a)); }

__device__ __forceinline__ void float_frexp(double a, int32_t& e) { if (a == 0.) e = int_min; else frexp(a, &e); }
__device__ __forceinline__ void float_frexp(float a, int32_t& e) { if (a == 0.f) e = int_min; else frexpf(a, &e); }

template <int32_t BLOCK_THREADS, class reduc_t, class real_t>
__global__ void vector_exponent_kernel(int64_t M, const real_t* __restrict__ A, int64_t lda, int32_t* __restrict__ vec_expon) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
  float_max max_func;

  A = &A[int64_t(blockIdx.x) * lda];
  reduc_t thread_data = reduc_t();

  for (int64_t i = threadIdx.x; i < M; i += inci)
    thread_data = max_func(float_abs(A[i]), thread_data);

  thread_data = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_data, max_func);
  if (threadIdx.x == 0)
    float_frexp(thread_data, vec_expon[blockIdx.x]);
}

constexpr int32_t block_threads = 512;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<block_threads, double> <<< N, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<block_threads, float> <<< N, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), vec_expon);
}

void internal::int8::vexp_f16(cudaStream_t stream, int32_t M, int32_t N, const half* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<block_threads, float> <<< N, block_threads, 0, stream >>> (int64_t(M), (const half*)A, int64_t(lda), vec_expon);
}

void internal::int8::vexp_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<block_threads, double> <<< N, block_threads, 0, stream >>> (int64_t(M) << 1, (const double*)A, int64_t(lda) << 1, vec_expon);
}

void internal::int8::vexp_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<block_threads, float> <<< N, block_threads, 0, stream >>> (int64_t(M) << 1, (const float*)A, int64_t(lda) << 1, vec_expon);
}

void internal::int8::vexp_cf16(cudaStream_t stream, int32_t M, int32_t N, const std::complex<half>* A, int32_t lda, int32_t* vec_expon) {
  vector_exponent_kernel<block_threads, float> <<< N, block_threads, 0, stream >>> (int64_t(M) << 1, (const half*)A, int64_t(lda) << 1, vec_expon);
}
