
#include <hyacin.h>
#include <internal.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdexcept>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr int32_t int_min = std::numeric_limits<int32_t>::min();
constexpr double f64_min = std::numeric_limits<double>::min();
constexpr float f32_min = std::numeric_limits<float>::min();
static __device__ __forceinline__ double _abs(double a) { return fabs(a); }
static __device__ __forceinline__ float _abs(float a) { return fabsf(a); }
static __device__ __forceinline__ float _abs(__half a) { return __half2float(__habs(a)); }

struct _max {
  __device__ __forceinline__ double operator()(double a, double b) { return fmax(a, b); }
  __device__ __forceinline__ float operator()(float a, float b) { return fmaxf(a, b); }
  __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) { return max(a, b); }
};

static __device__ __forceinline__ int32_t float_frexp(double a) { if (a < f64_min) { return int_max; } else { int32_t x; frexp(a, &x); return x; }}
static __device__ __forceinline__ int32_t float_frexp(float a) { if (a < f32_min) { return int_max; } else { int32_t x; frexpf(a, &x); return x; }}
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
  if (int32_t(threadIdx.x) == 0) { int32_t e = float_frexp(threadA); *mexp = (e == int_max) ? int_max : (u - e); }
}

__global__ void vector_exponent_reduc_kernel(int32_t N, int32_t seg_m1, int32_t* __restrict__ vexp) {
  int32_t i = (int32_t(blockIdx.x) << 7) + int32_t(threadIdx.x);
  if (i < N) {
    vexp = &vexp[i]; const int32_t* mexp = &vexp[N]; int32_t threadA = *mexp; 
    for (int32_t j = 0; j < seg_m1; ++j) { mexp += N; threadA = min(threadA, *mexp); } *vexp = threadA;
  }
}

template<class matrix_t> static inline void vector_exponents_dispatcher(cudaStream_t stream, int32_t batches, int32_t segM, int32_t lastM, int32_t N, const matrix_t* A, int32_t lda, int32_t u, int32_t* vexp) {
  int64_t lda64 = int64_t(lda);
  if (batches <= 0 || u < 0) { vector_exponent_init_kernel <<< uint32_t(N + 127) >> 7, 128, 0, stream >>> (N, vexp); } else {
    vector_exponent_batch_kernel<512> <<< dim3(batches, N), 512, 0, stream >>> (segM, lastM, A, lda64, u, &vexp[N]);
    vector_exponent_reduc_kernel <<< uint32_t(N + 127) >> 7, 128, 0, stream >>> (N, batches - 1, vexp);
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

template <int32_t BLOCK_THREADS> __global__ void vector_range_batch_kernel(int32_t N, int32_t u, const int32_t* __restrict__ vexp, int32_t* __restrict__ segU) {
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  _max cmp; int32_t threadA = int_min; const int32_t* mexp = &vexp[(int64_t(1) + int64_t(blockIdx.x)) * int64_t(N)]; segU = &segU[blockIdx.x];
  for (int32_t i = int32_t(threadIdx.x); i < N; i += BLOCK_THREADS) { int32_t e = mexp[i]; if (e < int_max) { threadA = cmp(threadA, vexp[i] - e); }}
  threadA = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, cmp);
  if (int32_t(threadIdx.x) == 0) { *segU = threadA == int_min ? int_min : (u + threadA); }
}

void internal::int8::vector_range(cudaStream_t stream, int32_t batches, int32_t N, int32_t u, const int32_t* vexp, int32_t* segU)
{ vector_range_batch_kernel<512> <<< batches, 512, 0, stream >>> (N, u, vexp, segU); }
