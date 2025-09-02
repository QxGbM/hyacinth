
#include <hyacinth.hpp>
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
  fma_f128 fma_func;
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    if constexpr(ALG == 0)
      c[i] = fma_func(a[i], b[i]);
    else
      c[i] = fma_func(a[i], b[i], c[i]);
}

struct add_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
};

template <class real_t, class matrix_t> struct fused_scal_a {
  real_t rsq; real_t* __restrict__ D; matrix_t x0; matrix_t* __restrict__ X; int64_t incx;

  fused_scal_a(real_t sq, real_t rsq, matrix_t* X, int64_t incx, real_t* D) :
    rsq(rsq), D(D), X(X), incx(incx) {
      union { real_t f[2]; matrix_t e;} d { sq, real_t() };
      x0 = d.e;
    }

  __device__ __forceinline__ void scal_a(double2 s, double2 a, double2& c_conj, double2& d) const {
    double2 e = c_conj = device::dd::mul(s, a);
    d = device::dd::add(device::dd::mul(device::dd::negate(e), e), d);
  }
  __device__ __forceinline__ void scal_a(float4 s, float4 a, float4& c_conj, float4& d) const {
    float4 e = c_conj = device::qf::mul(s, a);
    d = device::qf::add(device::qf::mul(device::qf::negate(e), e), d);
  }
  __device__ __forceinline__ void scal_a(double2 s, complex_double2 a, complex_double2& c_conj, double2& d) const {
    using device::dd::add, device::dd::mul, device::dd::negate;
    complex_double2 e = c_conj = device::dd::make_complex_double2(mul(s, a.real), mul(negate(s), a.imag));
    d = add(mul(negate(e.real), e.real), add(mul(negate(e.imag), e.imag), d));
  }
  __device__ __forceinline__ void scal_a(float4 s, complex_float4 a, complex_float4& c_conj, float4& d) const {
    using device::qf::add, device::qf::mul, device::qf::negate;
    complex_float4 e = c_conj = device::qf::make_complex_float4(mul(s, a.real), mul(negate(s), a.imag));
    d = add(mul(negate(e.real), e.real), add(mul(negate(e.imag), e.imag), d));
  }

  __device__ __forceinline__ void set_offdiag(matrix_t a, int64_t i) const { scal_a(rsq, a, X[i * incx], D[i]); }
  __device__ __forceinline__ void set_diag() const { X[0] = x0; }
};

template <class matrix_ptr, class matrix_const_ptr, int32_t WARP_THREADS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t FUSE_SCALE, class real_t, class matrix_t>
__global__ void gemv_kernel(int32_t M, int32_t N, int32_t split_N, matrix_const_ptr A, int32_t lda, matrix_ptr B, fused_scal_a<real_t, matrix_t> scal_func) {
  constexpr int32_t block_warps = BLOCK_THREADS / WARP_THREADS;
  constexpr int32_t elements = ITEMS_PER_THREAD * WARP_THREADS;
  int32_t inc_row = block_warps * gridDim.x;

  if constexpr(!FUSE_SCALE) {
    N = (blockIdx.y + 1 == gridDim.y) ? N : split_N;
    A = &A[uint64_t(blockIdx.y) * uint64_t(split_N)];
    B = &B[uint64_t(blockIdx.y) * uint64_t(lda)];
  }

  __shared__ typename cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load[block_warps];
  __shared__ typename cub::BlockReduce<matrix_t, WARP_THREADS>::TempStorage temp_reduce[block_warps];
  cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load[threadIdx.y]);
  cub::BlockReduce<matrix_t, WARP_THREADS> block_reduce(temp_reduce[threadIdx.y]);
  matrix_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  for (int32_t i = (block_warps * blockIdx.x + threadIdx.y + 1); i < M; i += inc_row) {
    matrix_const_ptr A_i = &A[uint64_t(i) * uint64_t(lda)];
    
    for (int32_t k = 0; k < N; k += elements) {
      int32_t num_items = N - k;
      if (elements <= num_items) {
        block_load.Load(&A_i[k], threadA);
        block_load.Load(&A[k], threadX);
      }
      else {
        block_load.Load(&A_i[k], threadA, num_items, matrix_t());
        block_load.Load(&A[k], threadX, num_items, matrix_t());
      }
      if (k == 0)
        array_fma<0>(threadA, threadX, threadB);
      else
        array_fma<1>(threadA, threadX, threadB);
    }

    add_f128 add_func;
    matrix_t block_res = block_reduce.Reduce(threadB, add_func);
    if (threadIdx.x == 0) {
      if constexpr(FUSE_SCALE)
        scal_func.set_offdiag(add_func(block_res, B[i]), i);
      else
        B[i] = block_res;
    }
  }

  if constexpr(FUSE_SCALE) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
      scal_func.set_diag();
  }
}

constexpr int32_t thread_bytes = 32;
constexpr int32_t gridy_max = 64; // maximum length of split-k reduction
constexpr int32_t target_blocks = 512; // ideal grid size for gemv

template <class matrix_ptr, class matrix_const_ptr, class real_t, class matrix_t>
inline int32_t gemv_dispatcher(cudaStream_t stream, real_t* scale, int32_t M, int32_t N, const matrix_t* A, int32_t lda, matrix_t* B, real_t* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);
  constexpr int32_t warp_threads[4] { 64, 128, 256, 512 };
  constexpr int32_t warp_reduces[4] { 128 * items_per_thread, 256 * items_per_thread, 512 * items_per_thread, 1024 * items_per_thread };
  constexpr int32_t block_threads = warp_threads[3];
  constexpr int32_t minimal_k = warp_reduces[3];
  int32_t grid[3] { (M + 7) >> 3, (M + 3) >> 2, (M + 1) >> 1 };
  fused_scal_a<real_t, matrix_t> scal_func(scale[0], scale[1], B, lda, D);

  if (target_blocks <= grid[0] || N < warp_reduces[0]) {
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[0], block_threads, items_per_thread, 1>
      <<< grid[0], dim3(warp_threads[0], block_threads / warp_threads[0], 1), 0, stream >>> (M, N, 0, A, lda, B, scal_func);
    return 0;
  }
  else if (target_blocks <= grid[1] || N < warp_reduces[1]) {
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[1], block_threads, items_per_thread, 1>
      <<< grid[1], dim3(warp_threads[1], block_threads / warp_threads[1], 1), 0, stream >>> (M, N, 0, A, lda, B, scal_func);
    return 0;
  }
  else if (target_blocks <= grid[2] || N < warp_reduces[2]) {
    gemv_kernel <matrix_ptr, matrix_const_ptr, warp_threads[2], block_threads, items_per_thread, 1>
      <<< grid[2], dim3(warp_threads[2], block_threads / warp_threads[2], 1), 0, stream >>> (M, N, 0, A, lda, B, scal_func);
    return 0;
  }
  else {
    int32_t log2_gridx = std::floor(std::log2f((target_blocks + M - 1) / M));
    int32_t log2_N = std::floor(std::log2f(std::max((N + minimal_k - 1) / minimal_k, 1)));
    int32_t gridy_occu = 1 << std::max(log2_gridx, 0);
    int32_t gridy_size = std::min(gridy_max, 1 << std::max(log2_N, 0));
    int32_t grid_y = std::max(1, std::min(gridy_occu, gridy_size) - 1);

    if (grid_y == 1) {
      gemv_kernel <matrix_ptr, matrix_const_ptr, block_threads, block_threads, items_per_thread, 1>
        <<< M, block_threads, 0, stream >>> (M, N, 0, A, lda, B, scal_func);
      return 0;
    }
    else {
      int64_t offset = int64_t(-grid_y) * int64_t(lda);
      int32_t split_N = N / grid_y;
      gemv_kernel <matrix_ptr, matrix_const_ptr, block_threads, block_threads, items_per_thread, 0>
        <<< dim3(M, grid_y, 1), block_threads, 0, stream >>> (M, N + split_N * (1 - grid_y), split_N, A, lda, &B[offset], scal_func);
      return grid_y + 1;
    }
  }
}

void internal::Cholesky::gemv_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, double2* D) {
  swap_cols_f128_dd(stream, j, N, M, A, lda);
  double2* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<double2* __restrict__, const double2* __restrict__>(stream, scale, M, N, A, lda, B, D);
  if (reduce)
    reduce_scal_f128_dd(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, D);
}

void internal::Cholesky::gemv_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, float4* D) {
  swap_cols_f128_qf(stream, j, N, M, A, lda);
  float4* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<float4* __restrict__, const float4* __restrict__>(stream, scale, M, N, A, lda, B, D);
  if (reduce)
    reduce_scal_f128_qf(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, D);
}

void internal::Cholesky::gemv_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D) {
  swap_cols_cf128_dd(stream, j, N, M, A, lda);
  complex_double2* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<complex_double2* __restrict__, const complex_double2* __restrict__>(stream, scale, M, N, A, lda, B, D);
  if (reduce)
    reduce_scal_cf128_dd(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, D);
}

void internal::Cholesky::gemv_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D) {
  swap_cols_cf128_qf(stream, j, N, M, A, lda);
  complex_float4* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<complex_float4* __restrict__, const complex_float4* __restrict__>(stream, scale, M, N, A, lda, B, D);
  if (reduce)
    reduce_scal_cf128_qf(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, D);
}
