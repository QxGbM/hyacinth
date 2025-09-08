
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::make_complex_double2(f.real, device::dd::negate(f.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::make_complex_float4(f.real, device::qf::negate(f.imag)); }
};

struct gemv_pp_fused {
  __device__ __forceinline__ void operator()(double c, double& c_conj, double& d) {
    c_conj = c; d = fma(-c, c, d);
  }
  __device__ __forceinline__ void operator()(float c, float& c_conj, float& d) {
    c_conj = c; d = fmaf(-c, c, d);
  }
  __device__ __forceinline__ void operator()(double2 c, double2& c_conj, double2& d) {
    c_conj = c; d = device::dd::add(device::dd::mul(device::dd::negate(c), c), d);
  }
  __device__ __forceinline__ void operator()(float4 c, float4& c_conj, float4& d) {
    c_conj = c; d = device::qf::add(device::qf::mul(device::qf::negate(c), c), d);
  }
  __device__ __forceinline__ void operator()(cuDoubleComplex c, cuDoubleComplex& c_conj, double& d) {
    c_conj = make_cuDoubleComplex(c.x, -c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d));
  }
  __device__ __forceinline__ void operator()(cuComplex c, cuComplex& c_conj, float& d) {
    c_conj = make_cuComplex(c.x, -c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d));
  }
  __device__ __forceinline__ void operator()(complex_double2 c, complex_double2& c_conj, double2& d) {
    using device::dd::add, device::dd::mul, device::dd::negate;
    c_conj = device::dd::make_complex_double2(c.real, negate(c.imag));
    d = add(mul(negate(c.real), c.real), add(mul(negate(c.imag), c.imag), d));
  }
  __device__ __forceinline__ void operator()(complex_float4 c, complex_float4& c_conj, float4& d) {
    using device::qf::add, device::qf::mul, device::qf::negate;
    c_conj = device::qf::make_complex_float4(c.real, negate(c.imag));
    d = add(mul(negate(c.real), c.real), add(mul(negate(c.imag), c.imag), d));
  }
};

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) { return device::cmp::double_max(a, b); }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) { return device::cmp::float_max(a, b); }
  __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) { return device::cmp::double2_max(a, b); }
  __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) { return device::cmp::float4_max(a, b); }

  __host__ __device__ __forceinline__ void init(double_idx& a) { a = double_idx({ 0., -1 }); }
  __host__ __device__ __forceinline__ void init(float_idx& a) { a = float_idx({ 0.f, -1 }); }
  __host__ __device__ __forceinline__ void init(double2_idx& a) { a = double2_idx({ make_double2(0., 0.), -1 }); }
  __host__ __device__ __forceinline__ void init(float4_idx& a) { a = float4_idx({ make_float4(0.f, 0.f, 0.f, 0.f), -1 }); }
};

template <class real_t, class real_ptr, class matrix_ptr, class idx_t, class idx_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, class matrix_t>
__global__ void gemv_pp_kernel(int32_t j, int32_t M, int32_t N, matrix_t sq, matrix_ptr A, int64_t lda, real_ptr D, idx_ptr idx) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
  constexpr int32_t elements = GRID_BLOCKS * BLOCK_THREADS;
  constexpr int32_t block_mask = ~(BLOCK_THREADS - 1) & (elements - 1);
  int32_t block_offset = int32_t(blockIdx.x) * BLOCK_THREADS;
  int32_t N2 = N & (BLOCK_THREADS - 1), N1 = N - N2;
  matrix_ptr A_i = &A[M], A_col_j = &A[int64_t(M) + int64_t(j) * lda], A_row_j = &A[j + M];
  idx_t thread_x;

  __shared__ matrix_t Aij[2];
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  conj conj_func; gemv_pp_fused pp_func; real_max cmp_max;
  cmp_max.init(thread_x);

  if (threadIdx.x == 0 && block_offset == 0) {
    if constexpr(COMPLEX) Aij[0] = conj_func(A_col_j[0]);
      else Aij[0] = A_col_j[0];
    Aij[1] = sq;
  }
  __syncthreads();

  for (int32_t k = block_offset; k < N1; k += elements) {
    int32_t i = k + int32_t(threadIdx.x);
    matrix_t thread_i = A_i[i], thread_j = A_col_j[i];
    real_t thread_c = D[i];
    A_col_j[i] = thread_i;
    pp_func(thread_j, thread_j, thread_c);

    if (i != j) {
      if (0 < i) {
        int64_t col_idx = int64_t(i) * lda;
        if constexpr(COMPLEX) A_row_j[col_idx] = conj_func(thread_i);
          else A_row_j[col_idx] = thread_i;
        A_i[col_idx] = thread_j;
      }
      D[i] = thread_c;
    }
    idx_t thread_y = idx_t({ (i == j) ? real_t() : thread_c, i });
    thread_x = (k == block_offset) ? thread_y : cmp_max(thread_x, thread_y);
  }

  if (threadIdx.x < N2 && block_offset == (N1 & block_mask)) {
    int32_t i = N1 + int32_t(threadIdx.x);
    matrix_t thread_i = A_i[i], thread_j = A_col_j[i];
    real_t thread_c = D[i];
    A_col_j[i] = thread_i;
    pp_func(thread_j, thread_j, thread_c);

    if (i != j) {
      if (0 < i) {
        int64_t col_idx = int64_t(i) * lda;
        if constexpr(COMPLEX) A_row_j[col_idx] = conj_func(thread_i);
          else A_row_j[col_idx] = thread_i;
        A_i[col_idx] = thread_j;
      }
      D[i] = thread_c;
    }
    idx_t thread_y = idx_t({ (i == j) ? real_t() : thread_c, i });
    thread_x = (N1 == block_offset) ? thread_y : cmp_max(thread_x, thread_y);
  }

  if (block_offset < N)
    thread_x = block_reduce.Reduce(thread_x, cmp_max);

  if (threadIdx.x == 0) {
    idx[blockIdx.x] = idx_t({ thread_x.real, (thread_x.idx == 0 ? j : thread_x.idx) - 1 });
    if (blockIdx.x == 0)
    { A_col_j[0] = Aij[0]; A_i[0] = Aij[1]; D[j] = D[0]; }
  }

  A_col_j = &A[int64_t(j) * lda];
  N2 = M & (BLOCK_THREADS - 1); N1 = M - N2;

  for (int32_t k = block_offset; k < N1; k += elements) {
    int32_t i = k + int32_t(threadIdx.x);
    matrix_t a = A[i]; A[i] = A_col_j[i]; A_col_j[i] = a;
  }

  if (threadIdx.x < N2 && block_offset == (N1 & block_mask)) {
    int32_t i = N1 + int32_t(threadIdx.x);
    matrix_t a = A[i]; A[i] = A_col_j[i]; A_col_j[i] = a;
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

void internal::Cholesky::gemv_pp_f64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* sq, double* A, int32_t lda, double* D) {
  if (0 < j) {
    gemv_pp_kernel <double, double* __restrict__, double* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, *sq, A, lda, D, (double_idx*)sq);
    imax_f64_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_f64(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_f32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* sq, float* A, int32_t lda, float* D) {
  if (0 < j) {
    gemv_pp_kernel <float, float* __restrict__, float* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, *sq, A, lda, D, (float_idx*)sq);
    imax_f32_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_f32(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_f128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* sq, double2* A, int32_t lda, double2* D) {
  if (0 < j) {
    gemv_pp_kernel <double2, double2* __restrict__, double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, *sq, A, lda, D, (double2_idx*)sq);
    imax_f128_dd_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_f128_dd(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_f128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* sq, float4* A, int32_t lda, float4* D) {
  if (0 < j) {
    gemv_pp_kernel <float4, float4* __restrict__, float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, *sq, A, lda, D, (float4_idx*)sq);
    imax_f128_qf_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_f128_qf(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_cf64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* sq, std::complex<double>* A, int32_t lda, double* D) {
  if (0 < j) {
    cuDoubleComplex sqc = make_cuDoubleComplex(*sq, 0.);
    gemv_pp_kernel <double, double* __restrict__, cuDoubleComplex* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, (cuDoubleComplex*)A, lda, D, (double_idx*)sq);
    imax_f64_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_cf64(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_cf32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* sq, std::complex<float>* A, int32_t lda, float* D) {
  if (0 < j) {
    cuComplex sqc = make_cuComplex(*sq, 0.f);
    gemv_pp_kernel <float, float* __restrict__, cuComplex* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, (cuComplex*)A, lda, D, (float_idx*)sq);
    imax_f32_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_cf32(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_cf128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* sq, complex_double2* A, int32_t lda, double2* D) {
  if (0 < j) {
    complex_double2 sqc = device::dd::make_complex_double2(*sq, make_double2(0., 0.));
    gemv_pp_kernel <double2, double2* __restrict__, complex_double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, A, lda, D, (double2_idx*)sq);
    imax_f128_dd_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_cf128_dd(stream, M, N, sq, A, lda, D);
}

void internal::Cholesky::gemv_pp_cf128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* sq, complex_float4* A, int32_t lda, float4* D) {
  if (0 < j) {
    complex_float4 sqc = device::qf::make_complex_float4(*sq, make_float4(0.f, 0.f, 0.f, 0.f));
    gemv_pp_kernel <float4, float4* __restrict__, complex_float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, A, lda, D, (float4_idx*)sq);
    imax_f128_qf_host_sync(stream, N - 1, std::min(grid_blocks, (N + block_threads - 1) / block_threads), sq);
  }
  else
    gemv_pp_nopiv_cf128_qf(stream, M, N, sq, A, lda, D);
}
