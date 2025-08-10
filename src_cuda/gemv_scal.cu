
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct minus_a_fma_real {
  __device__ __forceinline__ double operator()(double a, double b) { return -a * b; }
  __device__ __forceinline__ float operator()(float a, float b) { return -a * b; }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::mul(device::dd::negate(a), b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::mul(device::qf::negate(a), b); }
  __device__ __forceinline__ double operator()(double a, double b, double c) { return fma(-a, b, c); }
  __device__ __forceinline__ float operator()(float a, float b, float c) { return fmaf(-a, b, c); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 c) { return device::dd::fma(device::dd::negate(a), b, c); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) { return device::qf::fma(device::qf::negate(a), b, c); }
};

struct minus_conj_a_fma_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(fma(-a.x, b.x, -a.y * b.y), fma(-a.x, b.y, a.y * b.x)); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) {
    return make_cuComplex(fmaf(-a.x, b.x, -a.y * b.y), fmaf(-a.x, b.y, a.y * b.x)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { 
    return device::dd::mul(device::dd::make_complex_double2(device::dd::negate(a.real), a.imag), b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { 
    return device::qf::mul(device::qf::make_complex_float4(device::qf::negate(a.real), a.imag), b); }
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    return make_cuDoubleComplex(fma(-a.x, b.x, fma(-a.y, b.y, c.x)), fma(-a.x, b.y, fma(a.y, b.x, c.y))); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b, cuComplex c) {
    return make_cuComplex(fmaf(-a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(-a.x, b.y, fmaf(a.y, b.x, c.y))); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, complex_double2 c) { 
    return device::dd::fma(device::dd::make_complex_double2(device::dd::negate(a.real), a.imag), b, c); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) { 
    return device::qf::fma(device::qf::make_complex_float4(device::qf::negate(a.real), a.imag), b, c); }
};

template <int32_t FMA, int32_t COMPLEX, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ void array_fma(matrix_t const (&a)[ITEMS_PER_THREAD], matrix_t const (&b)[ITEMS_PER_THREAD], matrix_t (&c)[ITEMS_PER_THREAD]) {
  if constexpr(FMA && COMPLEX) {
    minus_conj_a_fma_complex fma_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      c[i] = fma_func(a[i], b[i], c[i]);
  }
  else if constexpr(COMPLEX) {
    minus_conj_a_fma_complex fma_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      c[i] = fma_func(a[i], b[i]);
  }
  else if constexpr(FMA) {
    minus_a_fma_real fma_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      c[i] = fma_func(a[i], b[i], c[i]);
  }
  else {
    minus_a_fma_real fma_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      c[i] = fma_func(a[i], b[i]);
  }
}

struct add_real {
  __device__ __forceinline__ double operator()(double a, double b) { return a + b; }
  __device__ __forceinline__ float operator()(float a, float b) { return a + b; }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
};

struct add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) { return make_cuComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::qf::add(a, b); }
};

struct scal_add_function {
  __device__ __forceinline__ void operator()(double s, double a, double b, double& c, double& c_conj) { c_conj = c = s * (a + b); }
  __device__ __forceinline__ void operator()(float s, float a, float b, float& c, float& c_conj) { c_conj = c = s * (a + b); }
  __device__ __forceinline__ void operator()(double2 s, double2 a, double2 b, double2& c, double2& c_conj) { c_conj = c = device::dd::mul(s, device::dd::add(a, b)); }
  __device__ __forceinline__ void operator()(float4 s, float4 a, float4 b, float4& c, float4& c_conj) { c_conj = c = device::qf::mul(s, device::qf::add(a, b)); }

  __device__ __forceinline__ void operator()(double s, cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex& c, cuDoubleComplex& c_conj) {
    cuDoubleComplex e = c = make_cuDoubleComplex(s * (a.x + b.x), s * (a.y + b.y));
    c_conj = make_cuDoubleComplex(e.x, -e.y);
  }
  __device__ __forceinline__ void operator()(float s, cuComplex a, cuComplex b, cuComplex& c, cuComplex& c_conj) {
    cuComplex e = c = make_cuComplex(s * (a.x + b.x), s * (a.y + b.y));
    c_conj = make_cuComplex(e.x, -e.y);
  }
  __device__ __forceinline__ void operator()(double2 s, complex_double2 a, complex_double2 b, complex_double2& c, complex_double2& c_conj) {
    complex_double2 e = c = device::dd::make_complex_double2(device::dd::mul(s, device::dd::add(a.real, b.real)), device::dd::mul(s, device::dd::add(a.imag, b.imag)));
    c_conj = device::dd::conj(e);
  }
  __device__ __forceinline__ void operator()(float4 s, complex_float4 a, complex_float4 b, complex_float4& c, complex_float4& c_conj) {
    complex_float4 e = c = device::qf::make_complex_float4(device::qf::mul(s, device::qf::add(a.real, b.real)), device::qf::mul(s, device::qf::add(a.imag, b.imag)));
    c_conj = device::qf::conj(e);
  }
};

template <class real_t, class matrix_t, class matrix_ptr, class matrix_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t FUSE_SCALE>
__global__ void gemv_kernel(real_t scale, int32_t M, int32_t N, int32_t split_N, matrix_const_ptr A, int32_t lda, matrix_ptr B) {
  constexpr int32_t COMPLEX = (sizeof(real_t) < sizeof(matrix_t));
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

  for (int32_t i = (block_warps * blockIdx.x + threadIdx.y); i < M; i += inc_row) {
    matrix_const_ptr A_i = &A[uint64_t(i) * uint64_t(lda)];

    warp_load.Load(A_i, threadA, N1, matrix_t());
    warp_load.Load(A, threadX, N1, matrix_t());
    array_fma<0, COMPLEX>(threadA, threadX, threadB);

    for (int32_t k = elements; k < N1; k += elements) {
      warp_load.Load(&A_i[k], threadA);
      warp_load.Load(&A[k], threadX);
      array_fma<1, COMPLEX>(threadA, threadX, threadB);
    }

    if (0 < N2) {
      warp_load.Load(&A_i[N1], threadA, N2, matrix_t());
      warp_load.Load(&A[N1], threadX, N2, matrix_t());
      array_fma<1, COMPLEX>(threadA, threadX, threadB);
    }

    matrix_t block_res;
    if constexpr(COMPLEX)
      block_res = block_reduce.Reduce(threadB, add_complex());
    else
      block_res = block_reduce.Reduce(threadB, add_real());

    if (threadIdx.x == 0)
      if constexpr(FUSE_SCALE) {
        scal_add_function scal_func;
        scal_func(scale, block_res, B[i], B[i], B[uint64_t(i) * uint64_t(lda)]);
      }
      else
        B[i] = block_res;
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t block_threads = block_warps * 32;
constexpr int32_t thread_bytes = 32;

template <class real_t, class matrix_t, class matrix_ptr, class matrix_const_ptr>
inline int64_t gemv_dispatcher(cudaStream_t stream, real_t scale, int32_t M, int32_t N, matrix_const_ptr A, int32_t lda, matrix_ptr B) {
  constexpr int32_t gridy_max = 6; // 2^6 = 64;
  constexpr int32_t target_blocks = 10; // 2^10 = 1024;
  constexpr int32_t minimal_k = 8; // 2^8 = 256;
  constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);

  int32_t grid_x = int32_t(M + 7) >> 3;
  int32_t log2_gridx = std::floor(std::log2f(grid_x));
  int32_t log2_N = std::floor(std::log2f(std::max(N, 1)));

  int32_t gridy_occu = 1 << std::max(target_blocks - log2_gridx, 0);
  int32_t gridy_size = 1 << std::max(std::min(gridy_max, log2_N - minimal_k), 0);
  int32_t grid_y = std::min(gridy_occu, gridy_size) - 1;

  if (grid_y <= 1) {
    gemv_kernel <real_t, matrix_t, matrix_ptr, matrix_const_ptr, block_threads, items_per_thread, 1>
      <<< dim3(grid_x, 1, 1), dim3(32, block_warps, 1), 0, stream >>> (scale, M, N, N, A, lda, B);
    return 0;
  }
  else {
    int64_t offset = int64_t(-grid_y) * int64_t(lda);
    int32_t split_N = N / grid_y;
    gemv_kernel <real_t, matrix_t, matrix_ptr, matrix_const_ptr, block_threads, items_per_thread, 0>
      <<< dim3(grid_x, grid_y, 1), dim3(32, block_warps, 1), 0, stream >>> (scale, M, N, split_N, A, lda, &B[offset]);
    return grid_y + 1;
  }
}

void internal::Cholesky::minus_transAx_plusB_scale_double(cudaStream_t stream, double scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<double, double, double* __restrict__, const double* __restrict__>(stream, scale, M, N, A, lda, B);
  if (reduce)
    reduce_scal_f64(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_transAx_plusB_scale_float(cudaStream_t stream, float scale, int32_t M, int32_t N, const float* A, int32_t lda, float* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<float, float, float* __restrict__, const float* __restrict__>(stream, scale, M, N, A, lda, B);
  if (reduce)
    reduce_scal_f32(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_transAx_plusB_scale_double2(cudaStream_t stream, double2 scale, int32_t M, int32_t N, const double2* A, int32_t lda, double2* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<double2, double2, double2* __restrict__, const double2* __restrict__>(stream, scale, M, N, A, lda, B);
  if (reduce)
    reduce_scal_f128_dd(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_transAx_plusB_scale_float4(cudaStream_t stream, float4 scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<float4, float4, float4* __restrict__, const float4* __restrict__>(stream, scale, M, N, A, lda, B);
  if (reduce)
    reduce_scal_f128_qf(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, double scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<double, cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__>(stream, scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B);
  if (reduce)
    reduce_scal_cf64(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_adjAx_plusB_scale_float_complex(cudaStream_t stream, float scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<float, cuComplex, cuComplex* __restrict__, const cuComplex* __restrict__>(stream, scale, M, N, (const cuComplex*)A, lda, (cuComplex*)B);
  if (reduce)
    reduce_scal_cf32(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(cudaStream_t stream, double2 scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<double2, complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__>(stream, scale, M, N, A, lda, B);
  if (reduce)
    reduce_scal_cf128_dd(stream, scale, M, reduce, B, lda);
}

void internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(cudaStream_t stream, float4 scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B) {
  int64_t reduce = N < 1 ? 1 : gemv_dispatcher<float4, complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__>(stream, scale, M, N, A, lda, B);
  if (reduce)
    reduce_scal_cf128_qf(stream, scale, M, reduce, B, lda);
}
