
#include <hyacin.h>
#include <internal.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <limits>
#include <stdexcept>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
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
__global__ void vector_exponent_init_kernel(int32_t N, int32_t* __restrict__ vexp) { int32_t i = (int32_t(blockIdx.x) << 7) + int32_t(threadIdx.x); if (i < N) { vexp[i] = int_max; }}

template <int32_t BLOCK_THREADS, class matrix_t> __global__ void vector_exponent_batch_kernel(int32_t segM, int32_t lastM, const matrix_t* __restrict__ A, int64_t lda, int32_t u, int32_t* __restrict__ mexp) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  using reduc_t = typename std::conditional_t<std::is_same_v<matrix_t, double> || std::is_same_v<matrix_t, cuDoubleComplex>, double, float>;
  __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _max cmp; reduc_t threadA = reduc_t(); int32_t M = (int32_t(blockIdx.x) + 1 == int32_t(gridDim.x)) ? lastM : segM;
  A = &A[int64_t(blockIdx.x) * int64_t(segM) + int64_t(blockIdx.y) * lda]; mexp = &mexp[int64_t(blockIdx.x) * int64_t(gridDim.y) + int64_t(blockIdx.y)];
  for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
  { if constexpr(Complex) { matrix_t Aij = A[i]; reduc_t r = _abs(Aij.x), i = _abs(Aij.y); threadA = cmp(threadA, cmp(r, i)); } else { reduc_t a = _abs(A[i]); threadA = cmp(threadA, a); }}
  threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, cmp);
  if (threadIdx.x == 0) { int32_t e = float_frexp(threadA); *mexp = (e == int_max) ? int_max : (u - e); }
}

__global__ void vector_exponent_reduc_kernel(int32_t N, int32_t seg_m1, const int32_t* __restrict__ mexp, int32_t* __restrict__ vexp) {
  int32_t i = (int32_t(blockIdx.x) << 7) + int32_t(threadIdx.x);
  if (i < N) { int32_t threadA = *(mexp += i); vexp = &vexp[i]; for (int32_t j = 0; j < seg_m1; ++j) { mexp += N; threadA = min(threadA, *mexp); } *vexp = threadA; }
}

template <int32_t BLOCK_THREADS> __global__ void vector_range_batch_kernel(int32_t N, int32_t u, const int32_t* __restrict__ mexp, const int32_t* __restrict__ vexp, int32_t* __restrict__ segU) {
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _max cmp; int32_t threadA = int_min; mexp = &mexp[int64_t(blockIdx.x) * int64_t(N)]; segU = &segU[blockIdx.x];
  for (int32_t i = int32_t(threadIdx.x); i < N; i += BLOCK_THREADS) { int32_t e = mexp[i]; if (e < int_max) { threadA = cmp(threadA, vexp[i] - e); }}
  threadA = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, cmp);
  if (threadIdx.x == 0) { *segU = u + threadA; }
}

template <int32_t BLOCK_THREADS, class matrix_t>
__global__ void vector_range_kernel(int32_t M, int32_t N, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int32_t* __restrict__ vbuf, int32_t* __restrict__ out) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  using reduc_t = typename std::conditional_t<std::is_same_v<matrix_t, double> || std::is_same_v<matrix_t, cuDoubleComplex>, double, float>;
  _max cmp; int32_t threadI = int_min;
  for (int32_t j = int32_t(blockIdx.x); j < N; j += int32_t(gridDim.x)) {
    reduc_t threadA = reduc_t();
    const matrix_t* Aj = &A[int64_t(j) * lda];
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    { if constexpr(Complex) { matrix_t Aij = Aj[i]; reduc_t r = _abs(Aij.x), i = _abs(Aij.y); threadA = cmp(threadA, cmp(r, i)); } else { reduc_t a = _abs(Aj[i]); threadA = cmp(threadA, a); }}
    int32_t e = float_frexp(threadA); if (e < int_max) { threadI = cmp(threadI, e + vexp[j]); }
  }
  threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadI, cmp);
  if (gridDim.x == 1) { if (threadIdx.x == 0) { *out = threadI; } return; }

  if (threadIdx.x == 0) { vbuf[blockIdx.x] = threadI; } else { threadI = int_min; }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS) { threadI = cmp(threadI, vbuf[i]); }
    threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadI, cmp);
    if (threadIdx.x == 0) { *out = threadI; }
  }
}

template<class matrix_t> inline void vector_exponents_dispatcher(cudaStream_t stream, int32_t batches, int32_t segM, int32_t lastM, int32_t N, const matrix_t* A, int32_t lda, int32_t u, int32_t* vexp) {
  int64_t lda64 = int64_t(lda);
  if (batches <= 0 || u < 0) { vector_exponent_init_kernel <<< uint32_t(N + 127) >> 7, 128, 0, stream >>> (N, vexp); } else {
    vector_exponent_batch_kernel<512> <<< dim3(batches, N), 512, 0, stream >>> (segM, lastM, A, lda64, u, &vexp[N]);
    vector_exponent_reduc_kernel <<< uint32_t(N + 127) >> 7, 128, 0, stream >>> (N, batches - 1, &vexp[N], vexp);
  }
}

extern "C" void hyacinXquantizeScaleBatch(const hyacinHandle_t* handle, int32_t batches, int32_t segM, int32_t lastM, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t uA, int32_t* vexp) {
  if (N <= 0) { return; }
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  switch (Atype) {
    case HYACIN_F64: vector_exponents_dispatcher(handle->cudaStream, batches, segM, lastM, N, (const double*)A, lda, uA, vexp); return;
    case HYACIN_F32: vector_exponents_dispatcher(handle->cudaStream, batches, segM, lastM, N, (const float*)A, lda, uA, vexp); return;
    case HYACIN_F16: vector_exponents_dispatcher(handle->cudaStream, batches, segM, lastM, N, (const __half*)A, lda, uA, vexp); return;
    case HYACIN_F64_COMPLEX: vector_exponents_dispatcher(handle->cudaStream, batches, segM, lastM, N, (const cuDoubleComplex*)A, lda, uA, vexp); return;
    case HYACIN_F32_COMPLEX: vector_exponents_dispatcher(handle->cudaStream, batches, segM, lastM, N, (const cuComplex*)A, lda, uA, vexp); return;
    case HYACIN_F16_COMPLEX: vector_exponents_dispatcher(handle->cudaStream, batches, segM, lastM, N, (const __half2*)A, lda, uA, vexp); return;
    default: return;
  }
}

template<class matrix_t>
inline void vector_range_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf) {
  constexpr int32_t block_threads = 512, grid_blocks = 512;
  int32_t device_sms = *u, maxBlocksPerSM = 0; int64_t lda64 = int64_t(lda);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_range_kernel<block_threads, matrix_t>, block_threads, 0);
  
  int32_t grid = std::min(grid_blocks, std::min(N, device_sms * maxBlocksPerSM));
  void* kernelArgs[]{ &M, &N, &A, &lda64, &vexp, &vbuf, &u };
  cudaLaunchCooperativeKernel(vector_range_kernel<block_threads, matrix_t>, grid, block_threads, kernelArgs, 0, stream);
}

namespace internal::int8 {

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher(stream, M, N, A, lda, u, vexp, vbuf); }

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf)
  { vector_range_dispatcher(stream, M, N, A, lda, u, vexp, vbuf); }

};
