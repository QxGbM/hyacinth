
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct minmax {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) {
    return make_double2(fmin(a.x, b.x), fmax(a.y, b.y));
  }
};

template <class real_const_ptr, int32_t BLOCK_THREADS>
__global__ void asq_kernel(int32_t M, real_const_ptr A, int64_t lda, uint32_t umax, uint32_t* __restrict__ scale, double* __restrict__ z) {
  __shared__ typename cub::BlockReduce<double2, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<double2, BLOCK_THREADS> block_reduce(temp_reduce);
  minmax mm_func;

  constexpr double inf = INFINITY;
  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  double2 threadA = make_double2(inf, -inf);

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    double a = double(A_i[i]);
    threadA = mm_func(make_double2(a, a), threadA);
  }

  if (0 < M)
    threadA = block_reduce.Reduce(threadA, mm_func);

  if (threadIdx.x == 0)
    device::int8::quant_bounds(threadA.x, threadA.y, umax, scale[blockIdx.x], z[blockIdx.x]);
}

constexpr int32_t block_threads = 512;

void internal::int8::asq_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint32_t umax, uint32_t* scale, double* z) {
  asq_kernel <const double* __restrict__, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, umax, scale, z);
}

void internal::int8::asq_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint32_t umax, uint32_t* scale, double* z) {
  asq_kernel <const float* __restrict__, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, umax, scale, z);
}
