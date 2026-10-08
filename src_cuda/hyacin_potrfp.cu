
#include <internal.hpp>
#include <ext_arith.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

__device__ __forceinline__ bool cmp_fl(double a, double b) { return a < b; }
__device__ __forceinline__ bool cmp_fl(float a, float b) { return a < b; }
__device__ __forceinline__ bool cmp_fl(double2 a, double2 b) {
  bool l1 = a.x < b.x, l2 = a.y < b.y, p1 = a.x == b.x; return l1 || (p1 && l2);
}
__device__ __forceinline__ bool cmp_fl(float4 a, float4 b) {
  bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, l4 = a.w < b.w;
  bool p1 = a.x == b.x, p2 = p1 && (a.y == b.y), p3 = p2 && (a.z == b.z);
  return l1 || (p1 && l2) || (p2 && l3) || (p3 && l4);
}

struct __align__(16) float_idx { float real; int32_t idx; int32_t p; };
struct __align__(16) double_idx { double real; int32_t idx; int32_t p; };
struct __align__(32) double2_idx { double2 real; int32_t idx; int32_t p; };
struct __align__(32) float4_idx { float4 real; int32_t idx; int32_t p; };
template <class idx_t> struct idx_max {
  __device__ __forceinline__ idx_t operator()(idx_t a, idx_t b) { return cmp_fl(a.real, b.real) ? b : a; }
};

__device__ __forceinline__ double2 operator-(const double2& a, const double2& b) { double sum, err, a2 = a.y - b.y; device::two_sum(a.x, -b.x, sum, err); return device::renormalize(make_double2(sum, a2 + err)); }
__device__ __forceinline__ double2 operator*(const double2& a, const double2& b) { double p = a.x * b.x; return device::renormalize(make_double2(p, fma(a.x, b.y, a.y * b.x) + fma(a.x, b.x, -p))); }
__device__ __forceinline__ double2 square(const double2& a) { double p = a.x * a.x; return device::renormalize(make_double2(p, fma(a.x + a.x, a.y, fma(a.x, a.x, -p)))); }
__device__ __forceinline__ double2 increment(double2 x) { double c = 1.5; device::two_sum(x.x, c, x.x, c); x.y += c; return device::renormalize(x); }

__device__ __forceinline__ float4 operator-(const float4& a, const float4& b) {
  float2 a0, a1, a2;
  device::two_sum(make_float2(a.x, a.y), make_float2(-b.x, -b.y), a0, a1); // 1122 - 1223
  device::two_sum(a.z, -b.z, a2.x, a2.y); // 33 - 34, 4@a2.y

  float r0 = a0.x; a2.y += a.w - b.w;
  device::two_sum(make_float2(a0.y, a2.x), a1, a0, a1); // 2233 - 2334, 4@a1.y
  device::two_sum(a0.y, a1.x, a0.y, a1.x); // 33 - 34, 4@a1.x
  return device::renormalize(make_float4(r0, a0.x, a0.y, a2.y + (a1.y + a1.x)));
}

__device__ __forceinline__ float4 operator*(const float4& a, const float4& b) {
  float2 prod = make_float2(a.x * b.x, a.y * b.y), err = make_float2(fmaf(a.x, b.x, -prod.x), fmaf(a.y, b.y, -prod.y));
  float c1 = prod.x, c4 = fmaf(a.w, b.x, a.z * b.y) + fmaf(a.y, b.z, fmaf(a.x, b.w, err.y));
  float2 c23 = make_float2(err.x, prod.y);

  prod = make_float2(a.y * b.x, a.z * b.x); err = make_float2(fmaf(a.y, b.x, -prod.x), fmaf(a.z, b.x, -prod.y));
  device::two_sum(c23, prod, c23, prod); device::two_sum(prod.x, err.x, prod.x, err.x); device::two_sum(c23.y, prod.x, c23.y, prod.x);
  c4 += (prod.x + prod.y) + (err.x + err.y);

  prod = make_float2(a.x * b.y, a.x * b.z); err = make_float2(fmaf(a.x, b.y, -prod.x), fmaf(a.x, b.z, -prod.y));
  device::two_sum(c23, prod, c23, prod); device::two_sum(prod.x, err.x, prod.x, err.x); device::two_sum(c23.y, prod.x, c23.y, prod.x);
  c4 += (prod.x + prod.y) + (err.x + err.y);
  return device::renormalize(make_float4(c1, c23.x, c23.y, c4));
}

__device__ __forceinline__ float4 square(const float4& a) {
  float2 prod = make_float2(a.x * a.x, a.y * a.y), err = make_float2(fmaf(a.x, a.x, -prod.x), fmaf(a.y, a.y, -prod.y)), c23 = make_float2(err.x, prod.y);
  float a1 = a.x + a.x, c1 = prod.x, c4 = fmaf(a.w, a1, a.z * (a.y + a.y)) + err.y;

  prod = make_float2(a.y * a1, a.z * a1); err = make_float2(fmaf(a.y, a1, -prod.x), fmaf(a.z, a1, -prod.y));
  device::two_sum(c23, prod, c23, prod); device::two_sum(prod.x, err.x, prod.x, err.x); device::two_sum(c23.y, prod.x, c23.y, prod.x);
  c4 += (prod.x + prod.y) + (err.x + err.y);
  return device::renormalize(make_float4(c1, c23.x, c23.y, c4));
}
__device__ __forceinline__ float4 increment(float4 x)
{ float c = 1.5f; device::two_sum(x.x, c, x.x, c); device::two_sum(x.y, c, x.y, c); device::two_sum(x.z, c, x.z, c); x.w += c; return device::renormalize(x); }

__device__ __forceinline__ double real_sqrt(double epi, double_idx x, double_idx& r)
{ x.real = sqrt(x.real); r = double_idx({ 1. / x.real, x.idx - 1, r.p - int32_t(x.real < epi) }); return x.real; }
__device__ __forceinline__ float real_sqrt(float epi, float_idx x, float_idx& r)
{ x.real = sqrtf(x.real); r = float_idx({ 1.f / x.real, x.idx - 1, r.p - int32_t(x.real < epi) }); return x.real; }

__device__ __forceinline__ double2 real_sqrt(double2 epi, double2_idx x, double2_idx& r) {
  int32_t e; double2 rx = make_double2(frexp(rsqrt(x.real.x), &e), 0.);
  int32_t e2 = e + e - 1; double2 s = make_double2(scalbn(-x.real.x, e2), scalbn(-x.real.y, e2));

  rx = rx * increment(s * square(rx));
  rx = rx * increment(s * square(rx));
  rx = make_double2(scalbn(rx.x, e), scalbn(rx.y, e)); x.real = x.real * rx;
  r = double2_idx({ rx, x.idx - 1, r.p - int32_t(cmp_fl(x.real, epi)) });
  return x.real;
}

__device__ __forceinline__ float4 real_sqrt(float4 epi, float4_idx x, float4_idx& r) {
  int32_t e; float4 rx = make_float4(frexpf(rsqrtf(x.real.x), &e), 0.f, 0.f, 0.f);
  int32_t e2 = e + e - 1; float4 s = make_float4(scalbnf(-x.real.x, e2), scalbnf(-x.real.y, e2), scalbnf(-x.real.z, e2), scalbnf(-x.real.w, e2));

  rx = rx * increment(s * square(rx));
  rx = rx * increment(s * square(rx));
  rx = rx * increment(s * square(rx));
  rx = make_float4(scalbnf(rx.x, e), scalbnf(rx.y, e), scalbnf(rx.z, e), scalbnf(rx.w, e)); x.real = x.real * rx;
  r = float4_idx({ rx, x.idx - 1, r.p - int32_t(cmp_fl(x.real, epi)) });
  return x.real;
}

template <class matrix_t, class real_t> __device__ __forceinline__ matrix_t ext(real_t x) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(x, 0.); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(x, 0.f); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(x, make_double2(0., 0.)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(x, make_float4(0.f, 0.f, 0.f, 0.f)); } else
  { return x; }
}

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
  if (BLOCK_THREADS == nthreads) {
    cooperative_groups::this_thread_block().sync();
    if (int32_t(threadIdx.x) == 0)
    { idx_t r = idx_t({ real_t(), 0, thread_x.idx <= 0 ? -1 : p }); real_t d = real_sqrt(real_t(), thread_x, r); *work = r; D[N] = epi * d; *A = ext<matrix_t>(d); }
  } else {
    if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS)
      { thread_x = cmp_max(thread_x, work[i]); }
      thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
      if (int32_t(threadIdx.x) == 0)
      { idx_t r = idx_t({ real_t(), 0, thread_x.idx <= 0 ? -1 : p }); real_t d = real_sqrt(real_t(), thread_x, r); *work = r; D[N] = epi * d; *A = ext<matrix_t>(d); }
    }
  }
}

__device__ __forceinline__ double pp_func(double r, double c, double& d)
{ c = r * c; d = fma(-c, c, d); return c; }
__device__ __forceinline__ float pp_func(float r, float c, float& d)
{ c = r * c; d = fmaf(-c, c, d); return c; }
__device__ __forceinline__ double2 pp_func(double2 r, double2 c, double2& d)
{ c = r * c; d = d - square(c); return c; }
__device__ __forceinline__ float4 pp_func(float4 r, float4 c, float4& d)
{ c = r * c; d = d - square(c); return c; }
__device__ __forceinline__ cuDoubleComplex pp_func(double r, cuDoubleComplex c, double& d)
{ c = make_cuDoubleComplex(r * c.x, -r * c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d)); return c; }
__device__ __forceinline__ cuComplex pp_func(float r, cuComplex c, float& d)
{ c = make_cuComplex(r * c.x, -r * c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d)); return c; }
__device__ __forceinline__ complex_double2 pp_func(double2 r, complex_double2 c, double2& d) {
  c = device::make_complex_double2(r * c.real, r * make_double2(-c.imag.x, -c.imag.y));
  d = d - device::add(square(c.real), square(c.imag)); return c;
}
__device__ __forceinline__ complex_float4 pp_func(float4 r, complex_float4 c, float4& d) {
  c = device::make_complex_float4(r * c.real, r * make_float4(-c.imag.x, -c.imag.y, -c.imag.z, -c.imag.w));
  d = d - device::add(square(c.real), square(c.imag)); return c;
}

template <class real_t, class matrix_t> struct add_fl {
  __device__ __forceinline__ matrix_t operator()(matrix_t a, matrix_t b) {
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return device::add(a, b); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return device::add(a, b); } else
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(device::add(a.real, b.real), device::add(a.imag, b.imag)); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(device::add(a.real, b.real), device::add(a.imag, b.imag)); } else
    { return matrix_t(); }
  }
};

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t fma_(matrix_t a, matrix_t b, matrix_t c) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return fma(-a, b, c); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return fmaf(-a, b, c); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return c - (a * b); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return c - (a * b); } else
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) {
    return make_cuDoubleComplex(fma(-a.x, b.x, fma(-a.y, b.y, c.x)), fma(-a.x, b.y, fma(a.y, b.x, c.y)));
  } else if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) {
    return make_cuComplex(fmaf(-a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(-a.x, b.y, fmaf(a.y, b.x, c.y)));
  } else if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) {
    double2 p1 = a.real * b.real, p2 = a.imag * b.imag, p3 = (a.imag - a.real) * device::add(b.real, b.imag);
    return device::make_complex_double2(c.real - device::add(p1, p2), device::add(p1 - p2, device::add(p3, c.imag)));
  } else if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) {
    float4 p1 = a.real * b.real, p2 = a.imag * b.imag, p3 = (a.imag - a.real) * device::add(b.real, b.imag);
    return device::make_complex_float4(c.real - device::add(p1, p2), device::add(p1 - p2, device::add(p3, c.imag)));
  } else { return matrix_t(); }
}

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t conj(matrix_t a) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(a.real, make_double2(-a.imag.x, -a.imag.y)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(a.real, make_float4(-a.imag.x, -a.imag.y, -a.imag.z, -a.imag.w)); } else
  { return a; }
}

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void potrf_iter_kernel(int32_t iterN, int32_t N, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, int32_t* __restrict__ out) {
  __shared__ idx_t r; __shared__ real_t e; add_fl<real_t, matrix_t> add_; idx_max<idx_t> cmp_max; 
  __shared__ typename cub::BlockReduce<matrix_t, BLOCK_THREADS>::TempStorage temp_gemv;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  if (int32_t(threadIdx.x) == 0) { r = *work; e = D[N]; }
  cooperative_groups::this_thread_block().sync();

  auto grid = cooperative_groups::this_grid();
  int32_t M = 0; const int32_t tid = int32_t(grid.thread_rank()), nthreads = int32_t(grid.num_threads());
  while (0 <= r.p) {
    int32_t j = r.idx; matrix_t* A_col_j = &A[int64_t(j) * lda];
    if (0 < M) {
      for (int32_t i = int32_t(blockIdx.x) + 1; i < N; i += int32_t(gridDim.x)) {
        int32_t l = i - int32_t(i <= j); matrix_t threadB = threadIdx.x ? matrix_t() : A_col_j[l], *A_col_i = &A[int64_t(l) * lda];
        for (int32_t k = int32_t(threadIdx.x) - M; k < 0; k += BLOCK_THREADS)
        { threadB = fma_<real_t>(A_col_i[k], A_col_j[k], threadB); }
        threadB = cub::BlockReduce<matrix_t, BLOCK_THREADS>(temp_gemv).Reduce(threadB, add_);
        cooperative_groups::this_thread_block().sync();
        if (int32_t(threadIdx.x) == 0) { A_col_j[l] = threadB; }
      }
      grid.sync();
    }

    ++M; --N; idx_t thread_c = idx_t();
    if (0 < j) {
      if (tid == 0) {
        int32_t t = jpiv[j]; jpiv[j] = jpiv[0]; jpiv[0] = t; real_t D0 = D[0];
        *A_col_j = pp_func(r.real, *A_col_j, D0); thread_c = cmp_max(thread_c, idx_t({ D[j] = D0, j, 0 }));
      }
      for (int32_t i = tid + 1; i < N; i += nthreads) {
        int32_t l = i + int32_t(j <= i); real_t D_i = D[l]; matrix_t* A_col_i = &A[int64_t(l) * lda];
        *A_col_i = pp_func(r.real, A_col_j[l], D_i); A_col_i[j] = conj<real_t>(A_col_j[l] = A[l]);
        thread_c = cmp_max(thread_c, idx_t({ D[l] = D_i, l, 0 }));
      }
      for (int32_t i = nthreads - tid - M; i < 0; i += nthreads)
      { matrix_t t = A_col_j[i]; A_col_j[i] = A[i]; A[i] = t; }
    } else for (int32_t i = tid + 1; i <= N; i += nthreads) {
      real_t D_i = D[i]; A[int64_t(i) * lda] = pp_func(r.real, A_col_j[i], D_i);
      thread_c = cmp_max(thread_c, idx_t({ D[i] = D_i, i, 0 }));
    }

    A = &(++A)[lda]; ++jpiv; ++D;
    thread_c = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_c, cmp_max);
    if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_c; } else { thread_c = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS)
      { thread_c = cmp_max(thread_c, work[i]); }
      thread_c = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_c, cmp_max);
      if (int32_t(threadIdx.x) == 0) { *A = ext<matrix_t>(real_sqrt(e, thread_c, r)); *work = r; }
    }
    grid.sync();
    if (int32_t(threadIdx.x) == 0) { r = *work; if (r.idx < 0 || N <= iterN) { r.p = -1; }}
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
inline int32_t potrfp_dispatcher(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, int32_t* rank) {
  if (fillmode == 'U' || fillmode == 'u')
    matrix_fill_upper_to_full<'U', real_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, lda);
  else if (fillmode == 'L' || fillmode == 'l')
    matrix_fill_upper_to_full<'L', real_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, lda);

  K = (K == N) ? 1 : (N - K); p = std::max(0, p); epi = std::min(1., std::max(0., std::abs(epi)));
  real_t epi_f;
  if constexpr(std::is_same_v<real_t, double>) { epi_f = epi; } else
  if constexpr(std::is_same_v<real_t, float>) { epi_f = float(epi); } else
  if constexpr(std::is_same_v<real_t, double2>) { epi_f = make_double2(epi, 0.); } else
  if constexpr(std::is_same_v<real_t, float4>) { float e0 = float(epi); epi -= double(e0); float e1 = float(epi); epi_f = make_float4(e0, e1, epi - double(e1), 0.f); } else
  { epi_f = real_t(); }

  constexpr int32_t grid_blocks = 2048, block_threads = 128;
  int32_t device_sms = *rank, maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
  uint8_t* diag = &((uint8_t*)D)[65536]; int64_t lda_p1 = lda + int64_t(1);
  void* initArgs[]{ &epi_f, &p, &N, &A, &lda_p1, &jpiv, &diag, &D };
  cudaLaunchCooperativeKernel(potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, initArgs, 0, stream);

  if (K < N) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_iter_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
    grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), N);
    void* kernelArgs[]{ &K, &N, &A, &lda, &jpiv, &diag, &D, &rank };
    cudaLaunchCooperativeKernel(potrf_iter_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
    cudaStreamSynchronize(stream); return *rank;
  } else { return int32_t(N == 1); }
}

namespace internal::Cholesky {

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, epi, K, p, N, A, lda, jpiv, dev_work, pinned_work); }

};
