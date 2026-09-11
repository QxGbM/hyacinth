
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <stdexcept>

constexpr int32_t int_min = 0x80000000, int_max = 0x7fffffff;
constexpr double f64_min = 0x1.0p-1022, f64_max = 0x1.0p1023;
constexpr float f32_min = 0x1.0p-126f, f32_max = 0x1.0p127f;

__device__ __forceinline__ double _abs(double a) { return fabs(a); }
__device__ __forceinline__ float _abs(float a) { return fabsf(a); }
__device__ __forceinline__ float _abs(__half a) { return __half2float(__habs(a)); }

__device__ __forceinline__ double2 _abs2(double a) { double f = fabs(a); return make_double2(f, f64_min < f ? f : f64_max); }
__device__ __forceinline__ float2 _abs2(float a) { float f = fabsf(a); return make_float2(f, f32_min < f ? f : f32_max); }
__device__ __forceinline__ float2 _abs2(__half a) { float f = __half2float(__habs(a)); return make_float2(f, f32_min < f ? f : f32_max); }

struct _max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return max(a, b); }

  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return make_double2(fmax(a.x, b.x), fmin(a.y, b.y)); }
  __device__ __forceinline__ float2 operator()(float2 a, float2 b) { return make_float2(fmaxf(a.x, b.x), fmin(a.y, b.y)); }
};
template <class T> __device__ __forceinline__ T _reduc_init();
template <> __device__ __forceinline__ double2 _reduc_init<double2>() { return make_double2(0., f64_max); }
template <> __device__ __forceinline__ float2 _reduc_init<float2>() { return make_float2(0.f, f32_max); }
__device__ __forceinline__ int32_t float_frexp(double a) { if (a == 0.) { return int_min; } else { int32_t x; frexp(a, &x); return x; }}
__device__ __forceinline__ int32_t float_frexp(float a) { if (a == 0.f) { return int_min; } else { int32_t x; frexpf(a, &x); return x; }}

template <int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_exponent_kernel(int32_t M, int32_t N, const matrix_t* __restrict__ A, int64_t lda, int32_t u, int32_t* __restrict__ vexp, int32_t* __restrict__ vbuf, int32_t* __restrict__ out) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage f_reduce;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage int_reduce;
  _max cmp_max; int32_t threadI = 0;

  for (int32_t j = blockIdx.x; j < N; j += gridDim.x) {
    reduc_t threadA = _reduc_init<reduc_t>();
    const matrix_t* Aj = &A[int64_t(j) * lda];
    for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
      matrix_t Aij = Aj[i];
      if constexpr(Complex) { reduc_t r = _abs2(Aij.x), i = _abs2(Aij.y); threadA = cmp_max(threadA, cmp_max(r, i)); }
        else { reduc_t a = _abs2(Aij); threadA = cmp_max(threadA, a); }
    }
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(f_reduce).Reduce(threadA, cmp_max);
    cooperative_groups::this_thread_block().sync();
    if (threadIdx.x == 0) { int32_t c = float_frexp(threadA.x); vexp[j] = u - c; threadI = max(threadI, (c - float_frexp(threadA.y))); }
  }
  if (gridDim.x == 1) { if (threadIdx.x == 0) { *out = threadI; } return; }

  if (threadIdx.x == 0) { vbuf[blockIdx.x] = threadI; } else { threadI = 0; }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      threadI = max(threadI, vbuf[i]);
    threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(int_reduce).Reduce(threadI, cmp_max);
    if (threadIdx.x == 0) { *out = threadI; }
  }
}

template <int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_range_kernel(int32_t M, int32_t N, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int32_t* __restrict__ vbuf, int32_t* __restrict__ out) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
  _max cmp_max; int32_t threadI = int_min;

  for (int32_t j = blockIdx.x; j < N; j += gridDim.x) {
    reduc_t threadA = reduc_t(); 
    const matrix_t* Aj = &A[int64_t(j) * lda];
    for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
      matrix_t Aij = Aj[i];
      if constexpr(Complex) { reduc_t r = _abs(Aij.x), i = _abs(Aij.y); threadA = cmp_max(threadA, cmp_max(r, i)); }
        else { reduc_t a = _abs(Aij); threadA = cmp_max(threadA, a); }
    }
    threadI = max(threadI, float_frexp(threadA) + vexp[j]);
  }
  threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(threadI, cmp_max);
  if (gridDim.x == 1) { if (threadIdx.x == 0) { *out = threadI; } return; }

  if (threadIdx.x == 0) { vbuf[blockIdx.x] = threadI; } else { threadI = int_min; }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      threadI = max(threadI, vbuf[i]);
    threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(threadI, cmp_max);
    if (threadIdx.x == 0) { *out = threadI; }
  }
}

template<class reduc_t, class matrix_t>
inline void vector_exponents_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e) {
  constexpr int32_t block_threads = 512, grid_blocks = 512;
  int64_t lda64 = int64_t(lda);
  int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_exponent_kernel<block_threads, reduc_t, matrix_t>, block_threads, 0);

  int32_t grid = std::min(N, std::min(grid_blocks, device_sms * maxBlocksPerSM)), *vbuf = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&vbuf, uint64_t(grid) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace allocation failed at Exponent Range.");

  void* kernelArgs[]{ &M, &N, &A, &lda64, &u, &vexp, &vbuf, &e };
  cudaLaunchCooperativeKernel(vector_exponent_kernel<block_threads, reduc_t, matrix_t>, grid, block_threads, kernelArgs, 0, stream);
  cudaFreeAsync(vbuf, stream);
  cudaStreamSynchronize(stream);
}

template<class reduc_t, class matrix_t>
inline void vector_range_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t* u, const int32_t* vexp) {
  constexpr int32_t block_threads = 512, grid_blocks = 512;
  int64_t lda64 = int64_t(lda);
  int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_range_kernel<block_threads, reduc_t, matrix_t>, block_threads, 0);
  
  int32_t grid = std::min(N, std::min(grid_blocks, device_sms * maxBlocksPerSM)), *vbuf = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&vbuf, uint64_t(grid) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace allocation failed at Exponent Range.");

  void* kernelArgs[]{ &M, &N, &A, &lda64, &vexp, &vbuf, &u };
  cudaLaunchCooperativeKernel(vector_range_kernel<block_threads, reduc_t, matrix_t>, grid, block_threads, kernelArgs, 0, stream);
  cudaFreeAsync(vbuf, stream);
  cudaStreamSynchronize(stream);
}

namespace internal::int8 {

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e)
  { vector_exponents_dispatcher<double2>(stream, M, N, A, lda, u, vexp, e); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e)
  { vector_exponents_dispatcher<float2>(stream, M, N, A, lda, u, vexp, e); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e)
  { vector_exponents_dispatcher<float2>(stream, M, N, A, lda, u, vexp, e); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e)
  { vector_exponents_dispatcher<double2>(stream, M, N, A, lda, u, vexp, e); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e)
  { vector_exponents_dispatcher<float2>(stream, M, N, A, lda, u, vexp, e); }

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t u, int32_t* vexp, int32_t* e)
  { vector_exponents_dispatcher<float2>(stream, M, N, A, lda, u, vexp, e); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* u, const int32_t* vexp)
  { vector_range_dispatcher<double>(stream, M, N, A, lda, u, vexp); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* u, const int32_t* vexp)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* u, const int32_t* vexp)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* u, const int32_t* vexp)
  { vector_range_dispatcher<double>(stream, M, N, A, lda, u, vexp); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* u, const int32_t* vexp)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* u, const int32_t* vexp)
  { vector_range_dispatcher<float>(stream, M, N, A, lda, u, vexp); }

}
