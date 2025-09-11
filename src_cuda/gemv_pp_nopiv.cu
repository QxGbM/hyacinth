
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct gemv_pp_fused {
  __device__ __forceinline__ void operator()(double c, double& c_conj, double& d) {
    d = fma(-c, c, d); c_conj = c;
  }
  __device__ __forceinline__ void operator()(float c, float& c_conj, float& d) {
    d = fmaf(-c, c, d); c_conj = c;
  }
  __device__ __forceinline__ void operator()(double2 c, double2& c_conj, double2& d) {
    d = device::dd::add(d, device::dd::negate(device::dd::square(c))); c_conj = c;
  }
  __device__ __forceinline__ void operator()(float4 c, float4& c_conj, float4& d) {
    d = device::qf::add(d, device::qf::negate(device::qf::square(c))); c_conj = c;
  }
  __device__ __forceinline__ void operator()(cuDoubleComplex c, cuDoubleComplex& c_conj, double& d) {
    d = fma(-c.x, c.x, fma(-c.y, c.y, d)); c_conj = make_cuDoubleComplex(c.x, -c.y);
  }
  __device__ __forceinline__ void operator()(cuComplex c, cuComplex& c_conj, float& d) {
    d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d)); c_conj = make_cuComplex(c.x, -c.y);
  }
  __device__ __forceinline__ void operator()(complex_double2 c, complex_double2& c_conj, double2& d) {
    using device::dd::add, device::dd::square, device::dd::negate;
    d = add(d, negate(add(square(c.real), square(c.imag))));
    c_conj = device::dd::make_complex_double2(c.real, negate(c.imag));
  }
  __device__ __forceinline__ void operator()(complex_float4 c, complex_float4& c_conj, float4& d) {
    using device::qf::add, device::qf::square, device::qf::negate;
    d = add(d, negate(add(square(c.real), square(c.imag))));
    c_conj = device::qf::make_complex_float4(c.real, negate(c.imag));
  }
};

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) { return device::cmp::double_max(a, b); }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) { return device::cmp::float_max(a, b); }
  __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) { return device::cmp::double2_max(a, b); }
  __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) { return device::cmp::float4_max(a, b); }
};

template <class real_ptr, class matrix_t, class matrix_ptr, class idx_t, class idx_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS>
__global__ void gemv_pp_nopiv_kernel(int32_t N, matrix_t sq, matrix_ptr A, int64_t lda, real_ptr D, idx_ptr idx) {
  constexpr int32_t elements = GRID_BLOCKS * BLOCK_THREADS;
  int32_t block_offset = int32_t(blockIdx.x) * BLOCK_THREADS;

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  gemv_pp_fused pp_func; real_max cmp_max;
  idx_t thread_x = idx_t();

  for (int32_t i = block_offset + int32_t(threadIdx.x); i < N; i += elements) {
    int32_t i1 = i + 1;
    idx_t thread_c = idx_t({ D[i], i1 });
    pp_func(A[i], A[int64_t(i1) * lda - 1], thread_c.real);
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  if (block_offset < N)
    thread_x = block_reduce.Reduce(thread_x, cmp_max);

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) A[-1] = sq;
    idx[blockIdx.x] = thread_x;
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

void internal::Cholesky::gemv_pp_nopiv_f64(cudaStream_t stream, int32_t M, int32_t N, double* sq, double* A, int32_t lda, double* D) {
  gemv_pp_nopiv_kernel <double* __restrict__, double, double* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, *sq, &A[M + 1], lda, &D[1], (double_idx*)sq);
  imax_f64_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_f32(cudaStream_t stream, int32_t M, int32_t N, float* sq, float* A, int32_t lda, float* D) {
  gemv_pp_nopiv_kernel <float* __restrict__, float, float* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, *sq, &A[M + 1], lda, &D[1], (float_idx*)sq);
  imax_f32_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_f128_dd(cudaStream_t stream, int32_t M, int32_t N, double2* sq, double2* A, int32_t lda, double2* D) {
  gemv_pp_nopiv_kernel <double2* __restrict__, double2, double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, *sq, &A[M + 1], lda, &D[1], (double2_idx*)sq);
  imax_f128_dd_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_f128_qf(cudaStream_t stream, int32_t M, int32_t N, float4* sq, float4* A, int32_t lda, float4* D) {
  gemv_pp_nopiv_kernel <float4* __restrict__, float4, float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, *sq, &A[M + 1], lda, &D[1], (float4_idx*)sq);
  imax_f128_qf_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_cf64(cudaStream_t stream, int32_t M, int32_t N, double* sq, std::complex<double>* A, int32_t lda, double* D) {
  cuDoubleComplex sqc = make_cuDoubleComplex(*sq, 0.);
  gemv_pp_nopiv_kernel <double* __restrict__, cuDoubleComplex, cuDoubleComplex* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, (cuDoubleComplex*)&A[M + 1], lda, &D[1], (double_idx*)sq);
  imax_f64_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_cf32(cudaStream_t stream, int32_t M, int32_t N, float* sq, std::complex<float>* A, int32_t lda, float* D) {
  cuComplex sqc = make_cuComplex(*sq, 0.f);
  gemv_pp_nopiv_kernel <float* __restrict__, cuComplex, cuComplex* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, (cuComplex*)&A[M + 1], lda, &D[1], (float_idx*)sq);
  imax_f32_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_cf128_dd(cudaStream_t stream, int32_t M, int32_t N, double2* sq, complex_double2* A, int32_t lda, double2* D) {
  complex_double2 sqc = device::dd::make_complex_double2(*sq, make_double2(0., 0.));
  gemv_pp_nopiv_kernel <double2* __restrict__, complex_double2, complex_double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, &A[M + 1], lda, &D[1], (double2_idx*)sq);
  imax_f128_dd_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}

void internal::Cholesky::gemv_pp_nopiv_cf128_qf(cudaStream_t stream, int32_t M, int32_t N, float4* sq, complex_float4* A, int32_t lda, float4* D) {
  complex_float4 sqc = device::qf::make_complex_float4(*sq, make_float4(0.f, 0.f, 0.f, 0.f));
  gemv_pp_nopiv_kernel <float4* __restrict__, complex_float4, complex_float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, &A[M + 1], lda, &D[1], (float4_idx*)sq);
  imax_f128_qf_host_sync(stream, N, std::min(grid_blocks, (N + block_threads - 2) / block_threads), sq);
}
