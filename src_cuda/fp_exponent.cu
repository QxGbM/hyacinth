
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <limits>
#include <stdexcept>

constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
struct A_max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return max(a, b); }
};
__device__ __forceinline__ double float_abs(double a) { return fabs(a); }
__device__ __forceinline__ float float_abs(float a) { return fabsf(a); }
__device__ __forceinline__ float float_abs(__half a) { return __half2float(__habs(a)); }
__device__ __forceinline__ double float_abs(cuDoubleComplex a) { return fmax(fabs(a.x), fabs(a.y)); }
__device__ __forceinline__ float float_abs(cuComplex a) { return fmaxf(fabsf(a.x), fabsf(a.y)); }
__device__ __forceinline__ float float_abs(__half2 a) { return fmaxf(__half2float(__habs(a.x)), __half2float(__habs(a.y))); }

template <int32_t sign> __device__ __forceinline__ int32_t float_frexp(double a, int32_t e)
{ if (a == 0.) { if constexpr(sign) return int_max; else return int_min; } else { int32_t x; frexp(a, &x); if constexpr(sign) return e - x; else return e + x; }}
template <int32_t sign> __device__ __forceinline__ int32_t float_frexp(float a, int32_t e)
{ if (a == 0.f) { if constexpr(sign) return int_max; else return int_min; } else { int32_t x; frexpf(a, &x); if constexpr(sign) return e - x; else return e + x; }}

template <int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_exponent_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, int32_t umax, int32_t* __restrict__ vexp) {
  __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
  A_max cmp_max; reduc_t thread_x = reduc_t();

  A = &A[int64_t(blockIdx.x) * lda];
  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS)
    thread_x = cmp_max(float_abs(A[i]), thread_x);
  thread_x = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  if (threadIdx.x == 0) { vexp[blockIdx.x] = float_frexp<1>(thread_x, umax); }
}

template <int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_range_kernel(int32_t M, int32_t N, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int32_t* __restrict__ vbuf, int32_t* __restrict__ out) {
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
  A_max cmp_max; int32_t thread_i = int_min;

  for (int32_t j = blockIdx.x; j < N; j += gridDim.x) {
    reduc_t thread_x = reduc_t(); 
    const matrix_t* Aj = &A[int64_t(j) * lda];
    for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS)
      thread_x = cmp_max(float_abs(Aj[i]), thread_x);
    thread_i = max(float_frexp<0>(thread_x, vexp[j]), thread_i);
  }
  thread_i = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(thread_i, cmp_max);
  if (gridDim.x == 1) { if (threadIdx.x == 0) { *out = thread_i; } return; }

  if (threadIdx.x == 0) { vbuf[blockIdx.x] = thread_i; } else { thread_i = int_min; }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      thread_i = max(thread_i, vbuf[i]);
    thread_i = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_i, cmp_max);
    if (threadIdx.x == 0) { *out = thread_i; }
  }
}

template<class matrix_t>
inline void vector_exponents_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t* umax, int32_t* vexp) {
  constexpr int32_t block_threads = 512, F64 = int32_t(std::is_same_v<matrix_t, double> || std::is_same_v<matrix_t, cuDoubleComplex>);
  int64_t lda64 = int64_t(lda); int32_t u = *umax;
  if (u) {
    if constexpr(F64) { vector_exponent_kernel<block_threads, double> <<< N, block_threads, 0, stream >>> (M, A, lda64, u, vexp); }
      else { vector_exponent_kernel<block_threads, float> <<< N, block_threads, 0, stream >>> (M, A, lda64, u, vexp); }
  }
  else {
    int32_t device_sms = 0, device = -1;
    cudaGetDevice(&device); cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device);
    int32_t maxBlocksPerSM = 0;
    if constexpr(F64) { cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_range_kernel<block_threads, double, matrix_t>, block_threads, 0); }
      else { cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_range_kernel<block_threads, float, matrix_t>, block_threads, 0); }
    
    int32_t grid = std::min(N, device_sms * maxBlocksPerSM), *vbuf = nullptr;
    if (cudaSuccess != cudaMallocAsync((void**)&vbuf, uint64_t(grid) * sizeof(int32_t), stream))
      throw std::runtime_error("Workspace allocation failed at Exponent Range.");

    void* kernelArgs[]{ &M, &N, &A, &lda64, &vexp, &vbuf, &umax };
    if constexpr(F64) { cudaLaunchCooperativeKernel(vector_range_kernel<block_threads, double, matrix_t>, grid, block_threads, kernelArgs, 0, stream); }
      else { cudaLaunchCooperativeKernel(vector_range_kernel<block_threads, float, matrix_t>, grid, block_threads, kernelArgs, 0, stream); }
    cudaFreeAsync(vbuf, stream);
    cudaStreamSynchronize(stream);
  }
}

namespace internal::int8 {

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* umax, int32_t* vexp)
  { vector_exponents_dispatcher(stream, M, N, A, lda, umax, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* umax, int32_t* vexp)
  { vector_exponents_dispatcher(stream, M, N, A, lda, umax, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* umax, int32_t* vexp)
  { vector_exponents_dispatcher(stream, M, N, A, lda, umax, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* umax, int32_t* vexp)
  { vector_exponents_dispatcher(stream, M, N, A, lda, umax, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* umax, int32_t* vexp)
  { vector_exponents_dispatcher(stream, M, N, A, lda, umax, vexp); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* umax, int32_t* vexp)
  { vector_exponents_dispatcher(stream, M, N, A, lda, umax, vexp); }

}
