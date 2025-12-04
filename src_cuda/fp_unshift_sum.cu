
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct __align__(32) i126 { uint64_t e[2]{}; };

struct i126_add {
  __device__ __forceinline__ i126 operator()(i126 a, i126 b) {
    device::int8::add_shifted(a.e, int64_t(b.e[0]), uint32_t(0));
    device::int8::add_shifted(a.e, int64_t(b.e[1]), uint32_t(63));
    return a;
  }
};

template <class real_t, class real_const_ptr, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int32_t M, real_const_ptr A, int64_t lda, const uint32_t* __restrict__ scale, i126* __restrict__ vec_sum) {
  __shared__ typename cub::BlockReduce<i126, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<i126, BLOCK_THREADS> block_reduce(temp_reduce);

  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  i126 threadA; int32_t sgn, expon;
  device::int8::extract_scale(scale[blockIdx.x], sgn, expon);
  expon = -expon;

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    int64_t q; uint32_t sft; real_t x = A_i[i];
    device::int8::round_f64_i64s(scalbn(double(sgn ? -x : x), expon), q, sft);
    device::int8::add_shifted(threadA.e, q, sft);
  }

  if (0 < M)
    threadA = block_reduce.Reduce(threadA, i126_add());

  if (threadIdx.x == 0)
    vec_sum[blockIdx.x] = threadA;
}

constexpr int32_t block_threads = 512;

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, const uint32_t* scale, uint64_t* vec_sum) {
  vector_sum_kernel <double, const double* __restrict__, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, scale, (i126*)vec_sum);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, const uint32_t* scale, uint64_t* vec_sum) {
  vector_sum_kernel <float, const float* __restrict__, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, scale, (i126*)vec_sum);
}
