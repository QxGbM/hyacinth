
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

__device__ __forceinline__ bool cmp_fl(double a, double b, bool& less) { less = a < b; return a == b; }
__device__ __forceinline__ bool cmp_fl(float a, float b, bool& less) { less = a < b; return a == b; }
__device__ __forceinline__ bool cmp_fl(double2 a, double2 b, bool& less) {
  bool l1 = a.x < b.x, l2 = a.y < b.y, p1 = a.x == b.x; less = l1 || (p1 && l2); return p1 && (a.y == b.y);
}
__device__ __forceinline__ bool cmp_fl(float4 a, float4 b, bool& less) {
  bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, l4 = a.w < b.w;
  bool p1 = a.x == b.x, p2 = p1 && (a.y == b.y), p3 = p2 && (a.z == b.z);
  less = l1 || (p1 && l2) || (p2 && l3) || (p3 && l4); return p3 && (a.w == b.w);
}

struct __align__(16) float_idx { float real; int32_t idx; int32_t p; };
struct __align__(16) double_idx { double real; int32_t idx; int32_t p; };
struct __align__(32) double2_idx { double2 real; int32_t idx; int32_t p; };
struct __align__(32) float4_idx { float4 real; int32_t idx; int32_t p; };
template <class idx_t> struct idx_max {
  __device__ __forceinline__ idx_t operator()(idx_t a, idx_t b) {
    bool less, par = cmp_fl(a.real, b.real, less); 
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx, idx_ab = less ? b.idx : a.idx;
    return idx_t({ less ? b.real : a.real, par ? idx_min : idx_ab, 0 });
  }
};

template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(double epi, double_idx x, double_idx& r) {
  double sqx = sqrt(x.real); int32_t i = r.p - int32_t(x.real < epi);
  r = double_idx({ 1. / sqx, x.idx - 1, (x.p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(sqx, 0.); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(float epi, float_idx x, float_idx& r) {
  float sqx = sqrtf(x.real); int32_t i = r.p - int32_t(x.real < epi);
  r = float_idx({ 1.f / sqx, x.idx - 1, (x.p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(sqx, 0.f); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(double2 epi, double2_idx x, double2_idx& r) {
  double2 sqx, rsqx; device::dd::frsqrt(x.real, sqx, rsqx); bool less; cmp_fl(x.real, epi, less); int32_t i = r.p - int32_t(less);
  r = double2_idx({ rsqx, x.idx - 1, (x.p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(sqx, make_double2(0., 0.)); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(float4 epi, float4_idx x, float4_idx& r) {
  float4 sqx, rsqx; device::qf::frsqrt(x.real, sqx, rsqx); bool less; cmp_fl(x.real, epi, less); int32_t i = r.p - int32_t(less);
  r = float4_idx({ rsqx, x.idx - 1, (x.p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(sqx, make_float4(0.f, 0.f, 0.f, 0.f)); } else { return sqx; }
}

__device__ __forceinline__ double mul_(double a, double b) { return a * b; }
__device__ __forceinline__ float mul_(float a, float b) { return a * b; }
__device__ __forceinline__ double2 mul_(double2 a, double2 b) { return device::dd::mul(a, b); }
__device__ __forceinline__ float4 mul_(float4 a, float4 b) { return device::qf::mul(a, b); }

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void potrf_init_kernel(real_t epi, int32_t p, int32_t N, matrix_t* __restrict__ A, int64_t lda_p1, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work) {
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce; idx_max<idx_t> cmp_max;
  auto grid = cooperative_groups::this_grid(); const int32_t nthreads = (grid.num_threads());

  idx_t thread_x = idx_t();
  for (int32_t i = int32_t(grid.thread_rank()); i < N; i += nthreads) {
    real_t x; int64_t loc = int64_t(i) * lda_p1;
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { x = A[loc].x; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { x = A[loc].x; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { x = A[loc].real; } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { x = A[loc].real; } else { x = A[loc]; }
    thread_x = cmp_max(thread_x, idx_t({ D[i] = x, jpiv[i] = i + 1, 0 }));
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  idx_t r = idx_t({ real_t(), 0, p });
  if (BLOCK_THREADS == nthreads) {
    cooperative_groups::this_thread_block().sync();
    if (int32_t(threadIdx.x) == 0) {
      thread_x.p = 1 < N; A[0] = real_sqrt<matrix_t>(D[N] = mul_(epi, thread_x.real), thread_x, r); *work = r;
      if (0 <= r.p && 0 < r.idx) { matrix_t* A_jj = &A[int64_t(r.idx) * lda_p1]; *A_jj = A_jj[-r.idx]; D[r.idx] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[r.idx]; jpiv[r.idx] = t; }
    }
  } else {
    if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS)
      { thread_x = cmp_max(thread_x, work[i]); }
      thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
      if (int32_t(threadIdx.x) == 0) {
        thread_x.p = 1 < N; A[0] = real_sqrt<matrix_t>(D[N] = mul_(epi, thread_x.real), thread_x, r); *work = r;
        if (0 <= r.p && 0 < r.idx) { matrix_t* A_jj = &A[int64_t(r.idx) * lda_p1]; *A_jj = A_jj[-r.idx]; D[r.idx] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[r.idx]; jpiv[r.idx] = t; }
      }
    }
  }
}

__device__ __forceinline__ double pp_func(double r, double c, double& d) {
  c = r * c; d = fma(-c, c, d); return c;
}
__device__ __forceinline__ float pp_func(float r, float c, float& d) {
  c = r * c; d = fmaf(-c, c, d); return c;
}
__device__ __forceinline__ double2 pp_func(double2 r, double2 c, double2& d) {
  c = device::dd::mul(r, c); d = device::dd::add(d, device::dd::negate(device::dd::square(c))); return c;
}
__device__ __forceinline__ float4 pp_func(float4 r, float4 c, float4& d) {
  c = device::qf::mul(r, c); d = device::qf::add(d, device::qf::negate(device::qf::square(c))); return c;
}
__device__ __forceinline__ cuDoubleComplex pp_func(double r, cuDoubleComplex c, double& d) {
  c = make_cuDoubleComplex(r * c.x, -r * c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d)); return c;
}
__device__ __forceinline__ cuComplex pp_func(float r, cuComplex c, float& d) {
  c = make_cuComplex(r * c.x, -r * c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d)); return c;
}
__device__ __forceinline__ complex_double2 pp_func(double2 r, complex_double2 c, double2& d) {
  using device::dd::add, device::dd::mul, device::dd::square, device::dd::negate;
  c = device::dd::make_complex_double2(mul(r, c.real), negate(mul(r, c.imag)));
  d = add(d, negate(add(square(c.real), square(c.imag)))); return c;
}
__device__ __forceinline__ complex_float4 pp_func(float4 r, complex_float4 c, float4& d) {
  using device::qf::add, device::qf::mul, device::qf::square, device::qf::negate;
  c = device::qf::make_complex_float4(mul(r, c.real), negate(mul(r, c.imag)));
  d = add(d, negate(add(square(c.real), square(c.imag)))); return c;
}

template <class real_t, class matrix_t> struct add_fl {
  __device__ __forceinline__ matrix_t operator()(matrix_t a, matrix_t b) {
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return device::dd::add(a, b); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return device::qf::add(a, b); } else
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(device::dd::add(a.real, b.real), device::dd::add(a.imag, b.imag)); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(device::qf::add(a.real, b.real), device::qf::add(a.imag, b.imag)); } else
    { return matrix_t(); }
  }
};

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t fma_(matrix_t a, matrix_t b, matrix_t c) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return fma(-a, b, c); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return fmaf(-a, b, c); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return device::dd::add(c, device::dd::mul(device::dd::negate(a), b)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return device::qf::add(c, device::qf::mul(device::qf::negate(a), b)); } else
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) {
    return make_cuDoubleComplex(fma(-a.x, b.x, fma(-a.y, b.y, c.x)), fma(-a.x, b.y, fma(a.y, b.x, c.y)));
  } else if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) {
    return make_cuComplex(fmaf(-a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(-a.x, b.y, fmaf(a.y, b.x, c.y)));
  } else if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) {
    using device::dd::add, device::dd::mul, device::dd::negate, device::dd::make_complex_double2;
    /*c.real = add(negate(add(mul(a.real, b.real), mul(a.imag, b.imag))), c.real);
    c.imag = add(add(mul(negate(a.real), b.imag), mul(a.imag, b.real)), c.imag);
    return c;*/
    double2 p1 = mul(a.real, b.real), p2 = mul(a.imag, b.imag), p3 = mul(add(negate(a.real), a.imag), add(b.real, b.imag));
    return make_complex_double2(add(negate(add(p1, p2)), c.real), add(add(p1, negate(p2)), add(p3, c.imag)));
  } else if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) {
    using device::qf::add, device::qf::mul, device::qf::negate, device::qf::make_complex_float4;
    /*c.real = add(negate(add(mul(a.real, b.real), mul(a.imag, b.imag))), c.real);
    c.imag = add(add(mul(negate(a.real), b.imag), mul(a.imag, b.real)), c.imag);
    return c;*/
    float4 p1 = mul(a.real, b.real), p2 = mul(a.imag, b.imag), p3 = mul(add(negate(a.real), a.imag), add(b.real, b.imag));
    return make_complex_float4(add(negate(add(p1, p2)), c.real), add(add(p1, negate(p2)), add(p3, c.imag)));
  } else { return matrix_t(); }
}

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t conj(matrix_t a) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); } else
  { return a; }
}

template <int32_t BLOCK_THREADS_EXP, class real_t, class matrix_t, class idx_t>
__global__ void potrf_iter_kernel(int32_t iterMax, int32_t N, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, int32_t* __restrict__ out) {
  constexpr int32_t BLOCK_THREADS = 1 << BLOCK_THREADS_EXP;
  __shared__ idx_t r, shm_x[BLOCK_THREADS]; __shared__ real_t e; add_fl<real_t, matrix_t> add_; idx_max<idx_t> cmp_max; 
  __shared__ typename cub::BlockReduce<matrix_t, BLOCK_THREADS>::TempStorage temp_gemv;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  auto grid = cooperative_groups::this_grid(); const int32_t tid = int32_t(grid.thread_rank()), nthreads = int32_t(grid.num_threads());
  if (int32_t(threadIdx.x) == 0) { r = *work; e = D[N]; }
  cooperative_groups::this_thread_block().sync();

  int32_t M = 0;
  while (0 <= r.p) {
    matrix_t* A_col_j = &A[int64_t(r.idx) * lda]; shm_x[threadIdx.x] = idx_t();
    if (0 < M) {
      for (int32_t i = int32_t(blockIdx.x) + 1; i < N; i += int32_t(gridDim.x)) {
        matrix_t threadB = threadIdx.x ? matrix_t() : A_col_j[i], *A_col_i = &A[int64_t(i != r.idx ? i : 0) * lda];
        for (int32_t k = int32_t(threadIdx.x) - M; k < 0; k += BLOCK_THREADS)
        { threadB = fma_<real_t>(A_col_i[k], A_col_j[k], threadB); }
        threadB = cub::BlockReduce<matrix_t, BLOCK_THREADS>(temp_gemv).Reduce(threadB, add_);
        cooperative_groups::this_thread_block().sync();
        if (int32_t(threadIdx.x) == 0) { A_col_j[i] = threadB; }
      }
      grid.sync();
      if (0 < r.idx) for (int32_t i = tid - M; i < 0; i += nthreads)
      { matrix_t t = A[i]; A[i] = A_col_j[i]; A_col_j[i] = t; }
    }
    for (int32_t i = tid + 1; i < N; i += nthreads) {
      matrix_t* A_col_i = &A[int64_t(i) * lda]; real_t D_i = D[i];
      A_col_i[0] = pp_func(r.real, A_col_j[i], D_i);
      shm_x[threadIdx.x] = cmp_max(shm_x[threadIdx.x], idx_t({ D[i] = D_i, i, 0 }));
      if (i != r.idx) { A_col_i[r.idx] = conj<real_t>(A_col_j[i] = A[i]); }
    }

    A = &(++A)[lda]; ++M; --N; ++jpiv; ++D;
    idx_t thread_c = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(shm_x[threadIdx.x], cmp_max);
    if (int32_t(threadIdx.x) == 0 && tid < N) { work[blockIdx.x] = thread_c; } else { thread_c = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      const int32_t active_blocks = int32_t(min(uint32_t(gridDim.x), uint32_t(1) + (uint32_t(N) >> BLOCK_THREADS_EXP)));
      for (int32_t i = int32_t(threadIdx.x) + 1; i < active_blocks; i += BLOCK_THREADS)
      { thread_c = cmp_max(thread_c, work[i]); }
      thread_c = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_c, cmp_max);
      if (int32_t(threadIdx.x) == 0) {
        thread_c.p = int32_t(1 < N && M < iterMax); *A = real_sqrt<matrix_t>(e, thread_c, r); *work = r;
        if (0 <= r.p && 0 < r.idx) { A_col_j = &A[int64_t(r.idx) * lda]; A_col_j[r.idx] = *A_col_j; D[r.idx] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[r.idx]; jpiv[r.idx] = t; }
      }
    }
    grid.sync();
    if (int32_t(threadIdx.x) == 0) { r = *work; }
    cooperative_groups::this_thread_block().sync();
  }

  if (tid == 0) { *out = M + int32_t(N == 1); }
}

template <char mode, class real_t, class matrix_t>
__global__ void matrix_fill_upper_to_full(matrix_t* __restrict__ A, int64_t lda) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  bool pred; if constexpr(mode == 'U') { pred = y < x; } else if constexpr(mode == 'L') { pred = x < y; } else { pred = false; }
  if (pred) { A[x + y * lda] = conj<real_t>(A[y + x * lda]); }
}

template <class idx_t, class real_t, class matrix_t>
inline int32_t potrfp_dispatcher(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, int32_t* rank) {
  if (fillmode == 'U' || fillmode == 'u')
    matrix_fill_upper_to_full<'U', real_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, lda);
  else if (fillmode == 'L' || fillmode == 'l')
    matrix_fill_upper_to_full<'L', real_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, lda);

  k = std::min(N, std::max(0, k)); k = k ? k : N; p = std::max(0, p); epi = std::min(1., std::max(0., std::pow(epi, 2)));
  real_t epi_f = real_t();
  if constexpr(std::is_same_v<real_t, double>) { epi_f = epi; }
  if constexpr(std::is_same_v<real_t, float>) { epi_f = float(epi); } else
  if constexpr(std::is_same_v<real_t, double2>) { epi_f = device::dd::double2dd(epi); } else
  if constexpr(std::is_same_v<real_t, float4>) { epi_f = device::qf::double2qf(epi); }

  constexpr int32_t grid_blocks = 2048, block_threads_exp = 7, block_threads = 1 << block_threads_exp;
  int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
  uint8_t* diag = &((uint8_t*)D)[65536]; int64_t lda_p1 = lda + int64_t(1);
  void* initArgs[]{ &epi_f, &p, &N, &A, &lda_p1, &jpiv, &diag, &D };
  cudaLaunchCooperativeKernel(potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, initArgs, 0, stream);

  if (1 < N) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_iter_kernel<block_threads_exp, real_t, matrix_t, idx_t>, block_threads, 0);
    grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), N);
    void* kernelArgs[]{ &k, &N, &A, &lda, &jpiv, &diag, &D, &rank };
    cudaLaunchCooperativeKernel(potrf_iter_kernel<block_threads_exp, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  }
  cudaStreamSynchronize(stream); return *rank;
}

namespace internal::Cholesky {

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

};