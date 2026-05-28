
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) { return device::cmp::double_max(a, b); }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) { return device::cmp::float_max(a, b); }
  __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) { return device::cmp::double2_max(a, b); }
  __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) { return device::cmp::float4_max(a, b); }
};

__device__ __forceinline__ void pp_func(double rsq, double c, double& c_conj, double& d) {
  c_conj = c = rsq * c; d = fma(-c, c, d);
}
__device__ __forceinline__ void pp_func(float rsq, float c, float& c_conj, float& d) {
  c_conj = c = rsq * c; d = fmaf(-c, c, d);
}
__device__ __forceinline__ void pp_func(double2 rsq, double2 c, double2& c_conj, double2& d) {
  c_conj = c = device::dd::mul(rsq, c); d = device::dd::add(d, device::dd::negate(device::dd::square(c)));
}
__device__ __forceinline__ void pp_func(float4 rsq, float4 c, float4& c_conj, float4& d) {
  c_conj = c = device::qf::mul(rsq, c); d = device::qf::add(d, device::qf::negate(device::qf::square(c)));
}
__device__ __forceinline__ void pp_func(double rsq, cuDoubleComplex c, cuDoubleComplex& c_conj, double& d) {
  c_conj = c = make_cuDoubleComplex(rsq * c.x, -rsq * c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d)); 
}
__device__ __forceinline__ void pp_func(float rsq, cuComplex c, cuComplex& c_conj, float& d) {
  c_conj = c = make_cuComplex(rsq * c.x, -rsq * c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d));
}
__device__ __forceinline__ void pp_func(double2 rsq, complex_double2 c, complex_double2& c_conj, double2& d) {
  using device::dd::add, device::dd::mul, device::dd::square, device::dd::negate;
  c_conj = c = device::dd::make_complex_double2(mul(rsq, c.real), negate(mul(rsq, c.imag)));
  d = add(d, negate(add(square(c.real), square(c.imag))));
}
__device__ __forceinline__ void pp_func(float4 rsq, complex_float4 c, complex_float4& c_conj, float4& d) {
  using device::qf::add, device::qf::mul, device::qf::square, device::qf::negate;
  c_conj = c = device::qf::make_complex_float4(mul(rsq, c.real), negate(mul(rsq, c.imag)));
  d = add(d, negate(add(square(c.real), square(c.imag))));
}

__device__ __forceinline__ cuDoubleComplex conj(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
__device__ __forceinline__ cuComplex conj(cuComplex a) { return make_cuComplex(a.x, -a.y); }
__device__ __forceinline__ complex_double2 conj(complex_double2 a) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); }
__device__ __forceinline__ complex_float4 conj(complex_float4 a) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); }

template <int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void gemv_pp_kernel(int32_t j, int32_t M, int32_t N, matrix_t sq, real_t rsq, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ idx) {
  constexpr int32_t elements = GRID_BLOCKS * BLOCK_THREADS;
  constexpr int32_t elem_mask = elements - 1;
  int32_t offset = int32_t(blockIdx.x) * BLOCK_THREADS + int32_t(threadIdx.x) + 1;
  matrix_t* A_col_j = &A[int64_t(j) * lda];

  for (int32_t i = offset - 1; i < M; i += elements)
  { matrix_t a = A[i]; A[i] = A_col_j[i]; A_col_j[i] = a; }
  A = &A[M]; A_col_j = &A_col_j[M];

  if (((j - offset) & elem_mask) == 0)
  { A_col_j[j] = A_col_j[0]; D[j] = D[0]; }

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  real_max cmp_max;
  idx_t thread_x = idx_t();

  for (int32_t i = offset; i < N; i += elements) {
    matrix_t* A_col_i = &A[int64_t(i) * lda];
    idx_t thread_c = idx_t({ D[i], i });
    pp_func(rsq, A_col_j[i], A_col_i[0], thread_c.real);
    if constexpr(sizeof(real_t) < sizeof(matrix_t))
      A_col_i[j] = conj(A_col_j[i] = A[i]);
    else
      A_col_i[j] = A_col_j[i] = A[i];
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  if (threadIdx.x == 0) {
    idx[blockIdx.x] = thread_x;
    if (blockIdx.x == 0)
    { A[0] = sq; int32_t p = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = p; }
  }
}

template <int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void gemv_pp_nopiv_kernel(int32_t N, matrix_t sq, real_t rsq, matrix_t* __restrict__ A, int64_t lda, real_t* __restrict__ D, idx_t* __restrict__ idx) {
  constexpr int32_t elements = GRID_BLOCKS * BLOCK_THREADS;
  int32_t offset = int32_t(blockIdx.x) * BLOCK_THREADS + int32_t(threadIdx.x) + 1;

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  real_max cmp_max;
  idx_t thread_x = idx_t();

  for (int32_t i = offset; i < N; i += elements) {
    idx_t thread_c = idx_t({ D[i], i });
    pp_func(rsq, A[i], A[int64_t(i) * lda], thread_c.real);
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  if (threadIdx.x == 0) {
    idx[blockIdx.x] = thread_x;
    if (blockIdx.x == 0) { A[0] = sq; }
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

void internal::Cholesky::gemv_pp_f64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* sq, double* A, int32_t lda, int32_t* jpiv, double* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sq[0], sq[1], A, int64_t(lda), jpiv, D, (double_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sq[0], sq[1], &A[M], int64_t(lda), D, (double_idx*)sq);
  imax_f64_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_f32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* sq, float* A, int32_t lda, int32_t* jpiv, float* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sq[0], sq[1], A, int64_t(lda), jpiv, D, (float_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sq[0], sq[1], &A[M], int64_t(lda), D, (float_idx*)sq);
  imax_f32_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_f128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* sq, double2* A, int32_t lda, int32_t* jpiv, double2* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sq[0], sq[1], A, int64_t(lda), jpiv, D, (double2_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sq[0], sq[1], &A[M], int64_t(lda), D, (double2_idx*)sq);
  imax_f128_dd_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_f128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* sq, float4* A, int32_t lda, int32_t* jpiv, float4* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sq[0], sq[1], A, int64_t(lda), jpiv, D, (float4_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sq[0], sq[1], &A[M], int64_t(lda), D, (float4_idx*)sq);
  imax_f128_qf_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_cf64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* sq, std::complex<double>* A, int32_t lda, int32_t* jpiv, double* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  cuDoubleComplex sqc = make_cuDoubleComplex(*sq, 0.);
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sqc, sq[1], (cuDoubleComplex*)A, int64_t(lda), jpiv, D, (double_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sqc, sq[1], (cuDoubleComplex*)&A[M], int64_t(lda), D, (double_idx*)sq);
  imax_f64_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_cf32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* sq, std::complex<float>* A, int32_t lda, int32_t* jpiv, float* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  cuComplex sqc = make_cuComplex(*sq, 0.f);
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sqc, sq[1], (cuComplex*)A, int64_t(lda), jpiv, D, (float_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sqc, sq[1], (cuComplex*)&A[M], int64_t(lda), D, (float_idx*)sq);
  imax_f32_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_cf128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* sq, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  complex_double2 sqc = device::dd::make_complex_double2(*sq, make_double2(0., 0.));
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sqc, sq[1], A, int64_t(lda), jpiv, D, (double2_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sqc, sq[1], &A[M], int64_t(lda), D, (double2_idx*)sq);
  imax_f128_dd_host_sync(stream, N, grid_n, sq);
}

void internal::Cholesky::gemv_pp_cf128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* sq, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D) {
  int32_t grid_n = std::min(grid_blocks, (N + block_threads - 2) / block_threads);
  int32_t grid = std::max(grid_n, std::min(grid_blocks, (M + block_threads - 1) / block_threads));
  complex_float4 sqc = device::qf::make_complex_float4(*sq, make_float4(0.f, 0.f, 0.f, 0.f));
  if (j) gemv_pp_kernel<grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (j, M, N, sqc, sq[1], A, int64_t(lda), jpiv, D, (float4_idx*)sq);
  else gemv_pp_nopiv_kernel<grid_blocks, block_threads>
    <<< grid_n, block_threads, 0, stream >>> (N, sqc, sq[1], &A[M], int64_t(lda), D, (float4_idx*)sq);
  imax_f128_qf_host_sync(stream, N, grid_n, sq);
}
