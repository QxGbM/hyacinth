
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

template <class matrix_t, class matrix_ptr, class matrix_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
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

template <class matrix_t, class matrix_ptr, class matrix_const_ptr>
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
  gemv_kernel <matrix_t, matrix_ptr, matrix_const_ptr, block_threads, items_per_thread>
    <<< dim3(grid_x, grid_y, 1), dim3(32, block_warps, 1), 0, stream >>> (M, N, split_N, A, lda, &B[offset]);
  return grid_y + 1;
}

void internal::Cholesky::gemv_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, double2* A, int32_t lda, double2* D) {
  double2* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<double2, double2* __restrict__, const double2* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_f128_dd(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, float4* A, int32_t lda, float4* D) {
  float4* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<float4, float4* __restrict__, const float4* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_f128_qf(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D) {
  complex_double2* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_cf128_dd(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D) {
  complex_float4* B = &A[N];
  int32_t reduce = 1;
  if (1 <= N && 2 <= M)
    reduce = gemv_dispatcher<complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__>(stream, M, N, A, lda, B);
  reduce_scal_cf128_qf(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_scal_f128_dd_f64(cudaStream_t stream, cublasHandle_t handle, double2* scale, int32_t M, int32_t Nq, int32_t Nd, double2* A, int32_t lda, double2* D) {
  int32_t reduce = 1, ld_f64 = 2 * lda;
  double* A_f64 = (double*)&A[Nq], *B_f64 = &A_f64[Nd];
  double2* B = &A[Nq + Nd];

  if (1 <= Nd && 2 <= M) {
    double minus_one = -1., zero = 0.;
    cublasDgemv(handle, CUBLAS_OP_T, Nd, M - 1, &minus_one, &A_f64[ld_f64], ld_f64, A_f64, 1, &zero, (double*)&B[1 - lda], 1);
    device::convert_and_copy(stream, M - 1, 1, &B[1 - lda], M - 1, device::Precision::FP64, 1, &B[1], M - 1, device::Precision::FP128_DD);
  }

  if (1 <= Nq && 2 <= M)
    reduce = gemv_dispatcher<double2, double2* __restrict__, const double2* __restrict__>(stream, M, Nq, A, lda, B);
  reduce_scal_f128_dd_f64(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B_f64, ld_f64, &D[1]);
}

void internal::Cholesky::gemv_scal_f128_qf_f64(cudaStream_t stream, cublasHandle_t handle, float4* scale, int32_t M, int32_t Nq, int32_t Nd, float4* A, int32_t lda, float4* D) {
  int32_t reduce = 1, ld_f64 = 2 * lda;
  double* A_f64 = (double*)&A[Nq], *B_f64 = &A_f64[Nd];
  float4* B = &A[Nq + Nd];

  if (1 <= Nd && 2 <= M) {
    double minus_one = -1., zero = 0.;
    cublasDgemv(handle, CUBLAS_OP_T, Nd, M - 1, &minus_one, &A_f64[ld_f64], ld_f64, A_f64, 1, &zero, (double*)&B[1 - lda], 1);
    device::convert_and_copy(stream, M - 1, 1, &B[1 - lda], M - 1, device::Precision::FP64, 1, &B[1], M - 1, device::Precision::FP128_QF);
  }

  if (1 <= Nq && 2 <= M)
    reduce = gemv_dispatcher<float4, float4* __restrict__, const float4* __restrict__>(stream, M, Nq, A, lda, B);
  reduce_scal_f128_qf_f64(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B_f64, ld_f64, &D[1]);
}

void internal::Cholesky::gemv_scal_cf128_dd_cf64(cudaStream_t stream, cublasHandle_t handle, double2* scale, int32_t M, int32_t Nq, int32_t Nd, complex_double2* A, int32_t lda, double2* D) {
  int32_t reduce = 1, ld_f64 = 2 * lda;
  std::complex<double>* A_f64 = (std::complex<double>*)&A[Nq], *B_f64 = &A_f64[Nd];
  complex_double2* B = &A[Nq + Nd];

  if (1 <= Nd && 2 <= M) {
    std::complex<double> minus_one(-1., 0.), zero(0., 0.);
    cublasZgemv(handle, CUBLAS_OP_C, Nd, M - 1, (cuDoubleComplex*)&minus_one, (const cuDoubleComplex*)&A[lda], lda, (const cuDoubleComplex*)A, 1, (cuDoubleComplex*)&zero, (cuDoubleComplex*)&B[1 - lda], 1);
    device::convert_and_copy(stream, 2 * M - 2, 1, &B[1 - lda], 2 * M - 2, device::Precision::FP64, 1, &B[1], 2 * M - 2, device::Precision::FP128_DD);
  }

  if (1 <= Nq && 2 <= M)
    reduce = gemv_dispatcher<complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__>(stream, M, Nq, A, lda, B);
  reduce_scal_cf128_dd_cf64(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B_f64, ld_f64, &D[1]);
}

void internal::Cholesky::gemv_scal_cf128_qf_cf64(cudaStream_t stream, cublasHandle_t handle, float4* scale, int32_t M, int32_t Nq, int32_t Nd, complex_float4* A, int32_t lda, float4* D) {
  int32_t reduce = 1, ld_f64 = 2 * lda;
  std::complex<double>* A_f64 = (std::complex<double>*)&A[Nq], *B_f64 = &A_f64[Nd];
  complex_float4* B = &A[Nq + Nd];

  if (1 <= Nd && 2 <= M) {
    std::complex<double> minus_one(-1., 0.), zero(0., 0.);
    cublasZgemv(handle, CUBLAS_OP_C, Nd, M - 1, (cuDoubleComplex*)&minus_one, (const cuDoubleComplex*)&A[lda], lda, (const cuDoubleComplex*)A, 1, (cuDoubleComplex*)&zero, (cuDoubleComplex*)&B[1 - lda], 1);
    device::convert_and_copy(stream, 2 * M - 2, 1, &B[1 - lda], 2 * M - 2, device::Precision::FP64, 1, &B[1], 2 * M - 2, device::Precision::FP128_QF);
  }

  if (1 <= Nq && 2 <= M)
    reduce = gemv_dispatcher<complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__>(stream, M, Nq, A, lda, B);
  reduce_scal_cf128_qf_cf64(stream, scale, M, reduce, &B[int64_t(1) + int64_t(1 - reduce) * int64_t(lda)], lda, B_f64, ld_f64, &D[1]);
}
