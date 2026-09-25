
#include <internal.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <limits>
#include <stdexcept>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr double f64_min = std::numeric_limits<double>::min();
constexpr float f32_min = std::numeric_limits<float>::min();
__device__ __forceinline__ double _abs(double a) { return fabs(a); }
__device__ __forceinline__ float _abs(float a) { return fabsf(a); }
__device__ __forceinline__ float _abs(__half a) { return __half2float(__habs(a)); }

struct _max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return max(a, b); }
};

__device__ __forceinline__ int32_t float_frexp(double a) { if (a < f64_min) { return int_max; } else { int32_t x; frexp(a, &x); return x; }}
__device__ __forceinline__ int32_t float_frexp(float a) { if (a < f32_min) { return int_max; } else { int32_t x; frexpf(a, &x); return x; }}
template <int32_t beta> __device__ __forceinline__ void exp_update(int32_t& e, int32_t i) { if constexpr(beta) { e = min(e, i); } else { e = i; }}

template <int32_t beta, int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_exponent_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, int32_t u, int32_t* __restrict__ vexp) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _max cmp; reduc_t threadA = reduc_t(); A = &A[int64_t(blockIdx.x) * lda]; vexp = &vexp[blockIdx.x];
  for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS) {
    matrix_t Aij = A[i];
    if constexpr(Complex) { reduc_t r = _abs(Aij.x), i = _abs(Aij.y); threadA = cmp(threadA, cmp(r, i)); }
      else { reduc_t a = _abs(Aij); threadA = cmp(threadA, a); }
  }
  threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, cmp);
  if (threadIdx.x == 0) { int32_t e = float_frexp(threadA); exp_update<beta>(vexp[0], (e == int_max) ? int_max : (u - e)); }
}

template <int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_range_kernel(int32_t M, int32_t N, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int32_t* __restrict__ vbuf, int32_t* __restrict__ out) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _max cmp; int32_t threadI = -1;
  for (int32_t j = int32_t(blockIdx.x); j < N; j += int32_t(gridDim.x)) {
    reduc_t threadA = reduc_t();
    const matrix_t* Aj = &A[int64_t(j) * lda];
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS) {
      matrix_t Aij = Aj[i];
      if constexpr(Complex) { reduc_t r = _abs(Aij.x), i = _abs(Aij.y); threadA = cmp(threadA, cmp(r, i)); }
        else { reduc_t a = _abs(Aij); threadA = cmp(threadA, a); }
    }
    int32_t e = float_frexp(threadA); if (e < int_max) { threadI = cmp(threadI, e + vexp[j]); }
  }
  threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadI, cmp);
  if (gridDim.x == 1) { if (threadIdx.x == 0) { *out = threadI; } return; }

  if (threadIdx.x == 0) { vbuf[blockIdx.x] = threadI; } else { threadI = -1; }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS)
    { threadI = cmp(threadI, vbuf[i]); }
    threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadI, cmp);
    if (threadIdx.x == 0) { *out = threadI; }
  }
}

template<class reduc_t, class matrix_t>
inline void vector_exponents_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp) {
  constexpr int32_t block_threads = 512;
  int64_t lda64 = int64_t(lda);
  if (beta) { vector_exponent_kernel<1, block_threads, reduc_t> <<< N, block_threads, 0, stream >>> (M, A, lda64, u, vexp); }
    else { vector_exponent_kernel<0, block_threads, reduc_t> <<< N, block_threads, 0, stream >>> (M, A, lda64, u, vexp); }
}

template<class reduc_t, class matrix_t>
inline void vector_range_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf) {
  constexpr int32_t block_threads = 512, grid_blocks = 512;
  int64_t lda64 = int64_t(lda);
  int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_range_kernel<block_threads, reduc_t, matrix_t>, block_threads, 0);
  
  int32_t grid = std::min(grid_blocks, std::min(N, device_sms * maxBlocksPerSM));
  void* kernelArgs[]{ &M, &N, &A, &lda64, &vexp, &vbuf, &u };
  cudaLaunchCooperativeKernel(vector_range_kernel<block_threads, reduc_t, matrix_t>, grid, block_threads, kernelArgs, 0, stream);
  cudaStreamSynchronize(stream);
}

namespace internal::int8 {

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp)
  { vector_exponents_dispatcher<double>(stream, M, N, A, lda, u, beta, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp)
  { vector_exponents_dispatcher<float>(stream, M, N, A, lda, u, beta, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp)
  { vector_exponents_dispatcher<float>(stream, M, N, A, lda, u, beta, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp)
  { vector_exponents_dispatcher<double>(stream, M, N, A, lda, u, beta, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp)
  { vector_exponents_dispatcher<float>(stream, M, N, A, lda, u, beta, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t u, int32_t beta, int32_t* vexp)
  { vector_exponents_dispatcher<float>(stream, M, N, A, lda, u, beta, vexp); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher<double>(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher<double>(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp, vbuf); }

}
