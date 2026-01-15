
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <cfloat>

struct minmax {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return make_double2(fmin(a.x, b.x), fmax(a.y, b.y)); }
};

__device__ __forceinline__ double conv_abs(double a) { return fabs(a); }
__device__ __forceinline__ double conv_abs(float a) { return double(fabsf(a)); }

template <class real_t, class real_const_ptr, int32_t BLOCK_THREADS>
__global__ void vector_exponent_kernel(int32_t M, real_const_ptr A, int64_t lda, uint64_t* __restrict__ vec_expon) {
  __shared__ typename cub::BlockReduce<double2, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<double2, BLOCK_THREADS> block_reduce(temp_reduce);
  minmax minmax_func;

  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  double2 thread_data = make_double2(DBL_MAX, 0.);

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    double a = conv_abs(A_i[i]);
    thread_data = minmax_func(make_double2(a, a), thread_data);
  }

  thread_data = block_reduce.Reduce(thread_data, minmax_func);
  if (threadIdx.x == 0) {
    int32_t emin, emax; frexp(thread_data.x, &emin); frexp(thread_data.y, &emax);
    uint32_t lo = uint32_t(emax), hi = uint32_t(emax - emin);
    vec_expon[blockIdx.x] = uint64_t(lo) | (uint64_t(hi) << 32);
  }
}

struct u32max {
  __device__ __forceinline__ uint32_t operator()(uint32_t a, uint32_t b) { return max(a, b); }
};

template <int32_t BLOCK_THREADS>
__global__ void diff_reduction_kernel(int32_t N, uint64_t* __restrict__ vec_expon) {
  __shared__ typename cub::BlockReduce<uint32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<uint32_t, BLOCK_THREADS> block_reduce(temp_reduce);
  u32max max_func;

  uint32_t thread_data = uint32_t(0);
  for (int32_t i = threadIdx.x; i < N; i += BLOCK_THREADS)
    thread_data = max_func(uint32_t(vec_expon[i] >> 32), thread_data);
  
  thread_data = block_reduce.Reduce(thread_data, max_func);
  if (threadIdx.x == 0)
  { uint32_t lo = uint32_t(vec_expon[0]); vec_expon[0] = uint64_t(lo) | (uint64_t(thread_data) << 32); }
}

constexpr int32_t block_threads = 512;

void internal::int8::vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint64_t* vec_expon) {
  vector_exponent_kernel<double, const double* __restrict__, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, vec_expon);
  diff_reduction_kernel<block_threads> <<< 1, block_threads, 0, stream >>> (N, vec_expon);
}

void internal::int8::vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint64_t* vec_expon) {
  vector_exponent_kernel<float, const float* __restrict__, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, vec_expon);
  diff_reduction_kernel<block_threads> <<< 1, block_threads, 0, stream >>> (N, vec_expon);
}
