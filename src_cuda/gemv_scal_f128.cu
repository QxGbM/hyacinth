
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct fma_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::mul(device::dd::negate(a), b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::mul(device::qf::negate(a), b); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 c) { return device::dd::add(c, operator()(a, b)); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) { return device::qf::add(c, operator()(a, b)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real, operator()(a.imag, b.imag)), 
      operator()(a.real, b.imag, operator()(device::dd::negate(a.imag), b.real))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real, operator()(a.imag, b.imag)), 
      operator()(a.real, b.imag, operator()(device::qf::negate(a.imag), b.real))); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, complex_double2 c) { 
    return device::dd::make_complex_double2(operator()(a.real, b.real, operator()(a.imag, b.imag, c.real)), 
      operator()(a.real, b.imag, operator()(device::dd::negate(a.imag), b.real, c.imag))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) { 
    return device::qf::make_complex_float4(operator()(a.real, b.real, operator()(a.imag, b.imag, c.real)), 
      operator()(a.real, b.imag, operator()(device::qf::negate(a.imag), b.real, c.imag))); }
};

template <int32_t ALG, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ void array_fma(matrix_t const (&a)[ITEMS_PER_THREAD], matrix_t const (&b)[ITEMS_PER_THREAD], matrix_t (&c)[ITEMS_PER_THREAD]) {
  if constexpr(ALG == 0) {
    fma_f128 fma_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      c[i] = fma_func(a[i], b[i]);
  }
  else {
    fma_f128 fma_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      c[i] = fma_func(a[i], b[i], c[i]);
  }
}

struct add_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
};

template <class real_t, class matrix_t, class matrix_ptr, class matrix_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void gemv_kernel(int32_t M, int32_t N, int32_t split_N, matrix_const_ptr A, int32_t lda, matrix_ptr B) {
  constexpr int32_t block_warps = BLOCK_THREADS / 32;
  constexpr int32_t elements = ITEMS_PER_THREAD * 32;
  int32_t inc_row = block_warps * gridDim.x;

  N = (blockIdx.y + 1 == gridDim.y) ? (N - split_N * blockIdx.y) : split_N;
  A = &A[uint64_t(blockIdx.y) * uint64_t(split_N)];
  B = &B[uint64_t(blockIdx.y) * uint64_t(lda)];
  int32_t rem = N & (elements - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::WarpLoad<matrix_t, ITEMS_PER_THREAD, cub::WARP_LOAD_STRIPED>::TempStorage temp_load[block_warps];
  __shared__ typename cub::BlockReduce<matrix_t, 32>::TempStorage temp_reduce[block_warps];
  matrix_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  cub::WarpLoad<matrix_t, ITEMS_PER_THREAD, cub::WARP_LOAD_STRIPED> warp_load(temp_load[threadIdx.y]);
  cub::BlockReduce<matrix_t, 32> block_reduce(temp_reduce[threadIdx.y]);

  for (int32_t i = (block_warps * blockIdx.x + threadIdx.y + 1); i < M; i += inc_row) {
    matrix_const_ptr A_i = &A[uint64_t(i) * uint64_t(lda)];

    warp_load.Load(A_i, threadA, N1, matrix_t());
    warp_load.Load(A, threadX, N1, matrix_t());
    array_fma<0>(threadA, threadX, threadB);

    for (int32_t k = elements; k < N1; k += elements) {
      warp_load.Load(&A_i[k], threadA);
      warp_load.Load(&A[k], threadX);
      array_fma<1>(threadA, threadX, threadB);
    }

    if (0 < N2) {
      warp_load.Load(&A_i[N1], threadA, N2, matrix_t());
      warp_load.Load(&A[N1], threadX, N2, matrix_t());
      array_fma<1>(threadA, threadX, threadB);
    }

    matrix_t block_res;
    block_res = block_reduce.Reduce(threadB, add_f128());

    if (threadIdx.x == 0)
      B[i] = block_res;
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t block_threads = block_warps * 32;
constexpr int32_t thread_bytes = 32;

constexpr int32_t gridy_max = 6; // 2^6 = 64; maximum length of split-k reduction
constexpr int32_t target_blocks = 10; // 2^10 = 1024; ideal grid size for gemv
constexpr int32_t minimal_k = 8; // 2^8 = 256; minimal length of k in each split

template <class real_t, class matrix_t, class matrix_ptr, class matrix_const_ptr>
inline int32_t gemv_dispatcher(cudaStream_t stream, int32_t M, int32_t N, matrix_const_ptr A, int32_t lda, matrix_ptr B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);

  int32_t grid_x = int32_t(M + 7) >> 3;
  int32_t log2_gridx = std::floor(std::log2f(grid_x));
  int32_t log2_N = std::floor(std::log2f(std::max(N, 1)));

  int32_t gridy_occu = 1 << std::max(target_blocks - log2_gridx, 0);
  int32_t gridy_size = 1 << std::max(std::min(gridy_max, log2_N - minimal_k), 0);
  int32_t grid_y = std::max(1, std::min(gridy_occu, gridy_size) - 1);

  int64_t offset = int64_t(-grid_y) * int64_t(lda);
  int32_t split_N = N / grid_y;
  gemv_kernel <real_t, matrix_t, matrix_ptr, matrix_const_ptr, block_threads, items_per_thread>
    <<< dim3(grid_x, grid_y, 1), dim3(32, block_warps, 1), 0, stream >>> (M, N, split_N, A, lda, &B[offset]);
  return grid_y + 1;
}

void internal::Cholesky::gemv_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, const double2* A, int32_t lda, double2* B, double2* D) {
  int32_t reduce = N < 1 || M < 2 ? 1 :
    gemv_dispatcher<double2, double2, double2* __restrict__, const double2* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_f128_dd(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* B, float4* D) {
  int32_t reduce = N < 1 || M < 2 ? 1 :
    gemv_dispatcher<float4, float4, float4* __restrict__, const float4* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_f128_qf(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* B, double2* D) {
  int32_t reduce = N < 1 || M < 2 ? 1 :
    gemv_dispatcher<double2, complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_cf128_dd(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B, float4* D) {
  int32_t reduce = N < 1 || M < 2 ? 1 :
    gemv_dispatcher<float4, complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_cf128_qf(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}
