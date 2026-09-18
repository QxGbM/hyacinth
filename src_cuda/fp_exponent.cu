
#include <hyacin.h>
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <limits>
#include <stdexcept>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr double f64_min = std::numeric_limits<double>::min(), f64_inf = std::numeric_limits<double>::infinity();
constexpr float f32_min = std::numeric_limits<float>::min(), f32_inf = std::numeric_limits<float>::infinity();

__device__ __forceinline__ double _abs(double a) { return fabs(a); }
__device__ __forceinline__ float _abs(float a) { return fabsf(a); }
__device__ __forceinline__ float _abs(__half a) { return __half2float(__habs(a)); }

__device__ __forceinline__ double2 _abs2(double a) { double f = _abs(a); return make_double2(f, f64_min <= f ? f : f64_inf); }
__device__ __forceinline__ float2 _abs2(float a) { float f = _abs(a); return make_float2(f, f32_min <= f ? f : f32_inf); }
__device__ __forceinline__ float2 _abs2(__half a) { float f = _abs(a); return make_float2(f, f32_min <= f ? f : f32_inf); }

struct _max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return max(a, b); }
};

struct _min_max {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return make_double2(fmax(a.x, b.x), fmin(a.y, b.y)); }
  __device__ __forceinline__ float2 operator()(float2 a, float2 b) { return make_float2(fmaxf(a.x, b.x), fminf(a.y, b.y)); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return min(a, b); }
};

template <class T> __device__ __forceinline__ T _reduc_init();
template <> __device__ __forceinline__ double2 _reduc_init<double2>() { return make_double2(0., f64_inf); }
template <> __device__ __forceinline__ float2 _reduc_init<float2>() { return make_float2(0.f, f32_inf); }

__device__ __forceinline__ int32_t float_frexp(double a) { if (a < f64_min) { return int_max; } else { int32_t x; frexp(a, &x); return x; }}
__device__ __forceinline__ int32_t float_frexp(float a) { if (a < f32_min) { return int_max; } else { int32_t x; frexpf(a, &x); return x; }}
template <int32_t beta> __device__ __forceinline__ void exp_update(int32_t& e, int32_t i) { if constexpr(beta) { e = min(e, i); } else { e = i; }}

template <int32_t beta, int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_exponent_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ vexp) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _min_max cmp;

  reduc_t threadA = _reduc_init<reduc_t>();
  A = &A[int64_t(blockIdx.x) * lda]; vexp = &vexp[blockIdx.x];
  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    matrix_t Aij = A[i];
    if constexpr(Complex) { reduc_t r = _abs2(Aij.x), i = _abs2(Aij.y); threadA = cmp(threadA, cmp(r, i)); }
      else { reduc_t a = _abs2(Aij); threadA = cmp(threadA, a); }
  }
  threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, cmp);
  if (threadIdx.x == 0) {
    int32_t e = float_frexp(threadA.x);
    exp_update<beta>(vexp[0], (e == int_max) ? int_max : (-e));
    exp_update<beta>(vexp[gridDim.x], (e == int_max) ? int_max : float_frexp(threadA.y));
  }
}

template <int32_t BLOCK_THREADS>
__global__ void exp_diff_reduct_kernel(int32_t N, int32_t u, int32_t* __restrict__ vexp, int32_t* __restrict__ out) {
  int32_t* vexp2 = &vexp[N]; int32_t threadA = int_max; _min_max cmp;
  for (int32_t i = threadIdx.x; i < N; i += BLOCK_THREADS)
  { if (vexp[i] < int_max) { int32_t e = vexp[i] += u; threadA = cmp(threadA, vexp2[i] + e); }}

  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  threadA = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, cmp);
  if (threadIdx.x == 0) { *out = threadA; }
}

template <int32_t BLOCK_THREADS, class reduc_t, class matrix_t>
__global__ void vector_range_kernel(int32_t M, int32_t N, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int32_t* __restrict__ vbuf, int32_t* __restrict__ out) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _max cmp; int32_t threadI = -1;

  for (int32_t j = blockIdx.x; j < N; j += gridDim.x) {
    reduc_t threadA = reduc_t();
    const matrix_t* Aj = &A[int64_t(j) * lda];
    for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
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
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
    { threadI = cmp(threadI, vbuf[i]); }
    threadI = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadI, cmp);
    if (threadIdx.x == 0) { *out = threadI; }
  }
}

template<class reduc_t, class matrix_t>
inline void vector_exponents_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t beta, int32_t* vexp) {
  constexpr int32_t block_threads = 512;
  int64_t lda64 = int64_t(lda);
  if (beta) { vector_exponent_kernel<1, block_threads, reduc_t> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp); }
    else { vector_exponent_kernel<0, block_threads, reduc_t> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp); }
}

extern "C" void hyacinXquantizeScale(hyacinHandle_t handle, int32_t M, int32_t N, hyacinPrecision_t Atype, const void* A, int32_t lda, int32_t beta, int32_t* vexp) {
  if (M <= 0 || N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch(Atype) {
    case HYACIN_F64: vector_exponents_dispatcher<double2>(handle.cudaStream, M, N, (const double*)A, lda, beta, vexp); return;
    case HYACIN_F32: vector_exponents_dispatcher<float2>(handle.cudaStream, M, N, (const float*)A, lda, beta, vexp); return;
    case HYACIN_F16: vector_exponents_dispatcher<float2>(handle.cudaStream, M, N, (const __half*)A, lda, beta, vexp); return;
    case HYACIN_F64_COMPLEX: vector_exponents_dispatcher<double2>(handle.cudaStream, M, N, (const cuDoubleComplex*)A, lda, beta, vexp); return;
    case HYACIN_F32_COMPLEX: vector_exponents_dispatcher<float2>(handle.cudaStream, M, N, (const cuComplex*)A, lda, beta, vexp); return;
    case HYACIN_F16_COMPLEX: vector_exponents_dispatcher<float2>(handle.cudaStream, M, N, (const __half2*)A, lda, beta, vexp); return;
    default: return;
  }
}

template<class reduc_t, class matrix_t>
inline void vector_range_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t* u, const int32_t* vexp) {
  constexpr int32_t block_threads = 512, grid_blocks = 512;
  int64_t lda64 = int64_t(lda);
  int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, vector_range_kernel<block_threads, reduc_t, matrix_t>, block_threads, 0);
  
  int32_t grid = std::min(grid_blocks, std::min(N, device_sms * maxBlocksPerSM)), *vbuf = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&vbuf, uint64_t(grid) * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace allocation failed at Exponent Range.");

  void* kernelArgs[]{ &M, &N, &A, &lda64, &vexp, &vbuf, &u };
  cudaLaunchCooperativeKernel(vector_range_kernel<block_threads, reduc_t, matrix_t>, grid, block_threads, kernelArgs, 0, stream);
  cudaFreeAsync(vbuf, stream);
  cudaStreamSynchronize(stream);
}

namespace internal::int8 {

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

  void vector_exp_diff(cudaStream_t stream, int32_t N, int32_t u, int32_t* d, int32_t* vexp) {
    constexpr int32_t block_threads = 512;
    exp_diff_reduct_kernel<block_threads> <<< 1, block_threads, 0, stream >>> (N, u, vexp, d);
    cudaStreamSynchronize(stream);
  }

}
