
#include <internal.hpp>
#include <ext_arith.hpp>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

static __device__ __forceinline__ double2 operator+(const double2& a, const double2& b) { return device::add(a, b); }
static __device__ __forceinline__ double2 operator-(const double2& a, const double2& b) { double s, e; device::two_sum(a.x, -b.x, s, e); return device::renormalize(make_double2(s, (a.y - b.y) + e)); }
static __device__ __forceinline__ float3 operator+(const float3& a, const float3& b) { return device::add(a, b); }
static __device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
  float2 s, e; device::two_sum(a.x, a.y, -b.x, -b.y, s.x, s.y, e.x, e.y); device::two_sum(s.y, e.x, s.y, e.x);
  return device::renormalize(make_float3(s.x, s.y, (a.z - b.z) + (e.x + e.y)));
}
static __device__ __forceinline__ float4 operator+(const float4& a, const float4& b) { return device::add(a, b); }
static __device__ __forceinline__ float4 operator-(const float4& a, const float4& b) {
  float3 s, e;
  device::two_sum(a.x, a.y, a.z, -b.x, -b.y, -b.z, s.x, s.y, s.z, e.x, e.y, e.z);
  device::two_sum(s.y, s.z, e.x, e.y, s.y, s.z, e.x, e.y);
  device::two_sum(s.z, e.x, s.z, e.x);
  return device::renormalize(make_float4(s.x, s.y, s.z, (a.w - b.w) + (e.x + e.y + e.z)));
}

static __device__ __forceinline__ double2 operator*(const double2& a, const double2& b) { double p = a.x * b.x; return device::renormalize(make_double2(p, fma(a.x, b.y, a.y * b.x) + fma(a.x, b.x, -p))); }
static __device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
  float r0 = a.x * b.x, r1 = a.x * b.y, p = a.y * b.x, e = fmaf(a.x, b.x, -r0);
  float r2 = fmaf(a.z, b.x, fmaf(a.y, b.y, fmaf(a.x, b.z, fmaf(a.x, b.y, -r1) + fmaf(a.y, b.x, -p))));
  device::two_sum(r1, e, r1, e); device::two_sum(r1, p, r1, p);
  return device::renormalize(make_float3(r0, r1, r2 + e + p));
}
static __device__ __forceinline__ float4 operator*(const float4& a, const float4& b) {
  float r0 = a.x * b.x, r1 = fmaf(a.x, b.x, -r0), r2 = a.y * b.y;
  float r3 = fmaf(a.w, b.x, fmaf(a.z, b.y, fmaf(a.y, b.z, fmaf(a.x, b.w, fmaf(a.y, b.y, -r2)))));

  float p0 = a.y * b.x, p1 = a.z * b.x, p2 = a.x * b.y, p3 = a.x * b.z;
  float e0 = fmaf(a.y, b.x, -p0), e1 = fmaf(a.z, b.x, -p1), e2 = fmaf(a.x, b.y, -p2), e3 = fmaf(a.x, b.z, -p3);
  device::two_sum(r1, r2, p0, p1, r1, r2, p0, p1); device::two_sum(r1, r2, p2, p3, r1, r2, p2, p3);
  device::two_sum(p0, p2, e0, e2, p0, p2, e0, e2); device::two_sum(r2, p0, r2, p0); device::two_sum(r2, p2, r2, p2);
  return device::renormalize(make_float4(r0, r1, r2, r3 + p0 + p1 + p2 + p3 + e0 + e1 + e2 + e3));
}

static __device__ __forceinline__ double2 sqr(const double2& a) { double p = a.x * a.x; return device::renormalize(make_double2(p, fma(a.x + a.x, a.y, fma(a.x, a.x, -p)))); }
static __device__ __forceinline__ float3 sqr(const float3& a) {
  float a1 = a.x + a.x, r0 = a.x * a.x, r1 = a.y * a1, e = fmaf(a.x, a.x, -r0), r2 = fmaf(a.y, a.y, fmaf(a1, a.z, fmaf(a.y, a1, -r1)));
  device::two_sum(r1, e, r1, e); return device::renormalize(make_float3(r0, r1, r2 + e));
}
static __device__ __forceinline__ float4 sqr(const float4& a) {
  float r0 = a.x * a.x, r2 = a.y * a.y, r1 = fmaf(a.x, a.x, -r0);
  float a1 = a.x + a.x, r3 = fmaf(a.w, a1, fmaf(a.z, a.y + a.y, fmaf(a.y, a.y, -r2)));

  float p0 = a.y * a1, p1 = a.z * a1, e0 = fmaf(a.y, a1, -p0), e1 = fmaf(a.z, a1, -p1);
  device::two_sum(r1, r2, p0, p1, r1, r2, p0, p1); device::two_sum(p0, e0, p0, e0); device::two_sum(r2, p0, r2, p0);
  return device::renormalize(make_float4(r0, r1, r2, r3 + p0 + p1 + e0 + e1));
}

static __device__ __forceinline__ double sqrt_fl(double& r) { double x = sqrt(r); r = 1. / x; return x; }
static __device__ __forceinline__ float sqrt_fl(float& r) { float x = sqrtf(r); r = 1.f / x; return x; }

static __device__ __forceinline__ double2 increment(const double2& x) { double2 y = device::renormalize(make_double2(1.5, x.x)); y.y += x.y; return y; }
static __device__ __forceinline__ double2 sqrt_fl(double2& r) {
  int32_t e; double2 rx = make_double2(frexp(rsqrt(r.x), &e), 0.);
  int32_t e2 = e + e - 1; double2 x = make_double2(scalbn(-r.x, e2), scalbn(-r.y, e2));
  rx = rx * increment(x * sqr(rx)); rx = rx * increment(x * sqr(rx));
  x = r; return x * (r = make_double2(scalbn(rx.x, e), scalbn(rx.y, e)));
}

static __device__ __forceinline__ float3 increment(const float3& x) { float3 y = device::renormalize(make_float3(1.5f, x.x, x.y)); y.z += x.z; return y; }
static __device__ __forceinline__ float3 sqrt_fl(float3& r) {
  int32_t e; float3 rx = make_float3(frexpf(rsqrtf(r.x), &e), 0.f, 0.f);
  int32_t e2 = e + e - 1; float3 x = make_float3(scalbnf(-r.x, e2), scalbnf(-r.y, e2), scalbnf(-r.z, e2));
  rx = rx * increment(x * sqr(rx)); rx = rx * increment(x * sqr(rx)); rx = rx * increment(x * sqr(rx));
  x = r; return x * (r = make_float3(scalbnf(rx.x, e), scalbnf(rx.y, e), scalbnf(rx.z, e)));
}

static __device__ __forceinline__ float4 increment(const float4& x) { float4 y = device::renormalize(make_float4(1.5f, x.x, x.y, x.z)); y.w += x.w; return y; }
static __device__ __forceinline__ float4 sqrt_fl(float4& r) {
  int32_t e; float4 rx = make_float4(frexpf(rsqrtf(r.x), &e), 0.f, 0.f, 0.f);
  int32_t e2 = e + e - 1; float4 x = make_float4(scalbnf(-r.x, e2), scalbnf(-r.y, e2), scalbnf(-r.z, e2), scalbnf(-r.w, e2));
  rx = rx * increment(x * sqr(rx)); rx = rx * increment(x * sqr(rx)); rx = rx * increment(x * sqr(rx));
  x = r; return x * (r = make_float4(scalbnf(rx.x, e), scalbnf(rx.y, e), scalbnf(rx.z, e), scalbnf(rx.w, e)));
}

template <class matrix_t, class real_t> static __device__ __forceinline__ matrix_t ext(const real_t& x) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(x, 0.); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(x, 0.f); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(x, make_double2(0., 0.)); } else
  if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, complex_float3>) { return device::make_complex_float3(x, make_float3(0.f, 0.f, 0.f)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(x, make_float4(0.f, 0.f, 0.f, 0.f)); } else
  { return x; }
}

template <class real_t> static __device__ __forceinline__ double to_double(const real_t& x) {
  if constexpr(std::is_same_v<real_t, double>) { return x; } else
  if constexpr(std::is_same_v<real_t, float>) { return double(x); } else
  if constexpr(std::is_same_v<real_t, double2>) { return x.x; } else
  if constexpr(std::is_same_v<real_t, float3>) { return double(x.x) + double(x.y) + double(x.z); } else
  if constexpr(std::is_same_v<real_t, float4>) { return double(x.x) + double(x.y) + double(x.z); } else
  { return 0.; }
}

static __device__ __forceinline__ bool cmp_fl(const double& a, const double& b) { return a < b; }
static __device__ __forceinline__ bool cmp_fl(const float& a, const float& b) { return a < b; }
static __device__ __forceinline__ bool cmp_fl(const double2& a, const double2& b) { bool l1 = a.x < b.x, l2 = a.y < b.y, p1 = a.x == b.x; return l1 || (p1 && l2); }
static __device__ __forceinline__ bool cmp_fl(const float3& a, const float3& b)
{ bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, p1 = a.x == b.x, p2 = p1 && (a.y == b.y); return l1 || (p1 && l2) || (p2 && l3); }
static __device__ __forceinline__ bool cmp_fl(const float4& a, const float4& b)
{ bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, l4 = a.w < b.w, p1 = a.x == b.x, p2 = p1 && (a.y == b.y), p3 = p2 && (a.z == b.z); return l1 || (p1 && l2) || (p2 && l3) || (p3 && l4); }

struct __align__(8) float_idx { double trace; float real; int32_t idx; };
struct __align__(8) double_idx { double trace, real; int32_t idx; };
struct __align__(16) double2_idx { double2 real; double trace; int32_t idx; };
struct __align__(8) float3_idx { double trace; float3 real; int32_t idx; };
struct __align__(16) float4_idx { float4 real; double trace; int32_t idx; };
template <class real_t, class idx_t> struct idx_max {
  __device__ __forceinline__ idx_t operator()(idx_t a, const real_t& x, const int32_t i)
  { bool p = cmp_fl(a.real, x); a.idx = p ? i : a.idx; a.real = p ? x : a.real; a.trace += to_double(x); return a; }
  __device__ __forceinline__ idx_t operator()(idx_t a, idx_t b)
  { bool p = cmp_fl(a.real, b.real); a.idx = p ? b.idx : a.idx; a.real = p ? b.real : a.real; a.trace += b.trace; return a; }
};

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void potrf_init_kernel(double eps, int32_t N, matrix_t* __restrict__ A, int64_t lda_p1, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work) {
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce; idx_max<real_t, idx_t> cmp_max;
  auto grid = cooperative_groups::this_grid(); const int32_t nthreads = (grid.num_threads());
  constexpr bool type_cuComplex = !std::is_same_v<real_t, matrix_t> && (std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex>);
  constexpr bool type_stuComplex = !std::is_same_v<real_t, matrix_t> && (std::is_same_v<matrix_t, complex_double2> || std::is_same_v<matrix_t, complex_float3> || std::is_same_v<matrix_t, complex_float4>);

  idx_t thread_x = idx_t();
  for (int32_t i = int32_t(grid.thread_rank()); i < N; i += nthreads) {
    real_t x; int64_t loc = int64_t(i) * lda_p1;
    if constexpr(type_cuComplex) { x = A[loc].x; } else if constexpr(type_stuComplex) { x = A[loc].real; } else { x = A[loc]; }
    thread_x = cmp_max(thread_x, D[i] = x, jpiv[i] = i + 1);
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  if (BLOCK_THREADS == nthreads) {
    cooperative_groups::this_thread_block().sync();
    if (int32_t(threadIdx.x) == 0)
    { --thread_x.idx; thread_x.trace = eps * thread_x.trace; *A = ext<matrix_t>(sqrt_fl(thread_x.real)); *work = thread_x; }
  } else {
    if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS)
      { thread_x = cmp_max(thread_x, work[i]); }
      thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
      if (int32_t(threadIdx.x) == 0)
      { --thread_x.idx; thread_x.trace = eps * thread_x.trace; *A = ext<matrix_t>(sqrt_fl(thread_x.real)); *work = thread_x; }
    }
  }
}

template <class real_t, class matrix_t> static __device__ __forceinline__ matrix_t fma_fl(const matrix_t& a, const matrix_t& b, const matrix_t& c) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return fma(-a, b, c); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return fmaf(-a, b, c); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return c - (a * b); } else
  if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, float3>) { return c - (a * b); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return c - (a * b); } else
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) {
    return make_cuDoubleComplex(fma(-a.x, b.x, fma(-a.y, b.y, c.x)), fma(-a.x, b.y, fma(a.y, b.x, c.y)));
  } else if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) {
    return make_cuComplex(fmaf(-a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(-a.x, b.y, fmaf(a.y, b.x, c.y)));
  } else if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) {
    double2 p1 = a.real * b.real, p2 = a.imag * b.imag, p3 = (a.imag - a.real) * (b.real + b.imag);
    return device::make_complex_double2(c.real - (p1 + p2), (p1 - p2) + (p3 + c.imag));
  } else if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, complex_float3>) {
    float3 p1 = a.real * b.real, p2 = a.imag * b.imag, p3 = (a.imag - a.real) * (b.real + b.imag);
    return device::make_complex_float3(c.real - (p1 + p2), (p1 - p2) + (p3 + c.imag));
  } else if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) {
    float4 p1 = a.real * b.real, p2 = a.imag * b.imag, p3 = (a.imag - a.real) * (b.real + b.imag);
    return device::make_complex_float4(c.real - (p1 + p2), (p1 - p2) + (p3 + c.imag));
  } else { return matrix_t(); }
}

static __device__ __forceinline__ double process_diag(double& d, const double& r, const double c) { double rc = r * c; d = fma(-rc, rc, d); return rc; }
static __device__ __forceinline__ float process_diag(float& d, const float& r, const float c) { float rc = r * c; d = fmaf(-rc, rc, d); return rc; }
static __device__ __forceinline__ double2 process_diag(double2& d, const double2& r, const double2 c) { double2 rc = r * c; d = d - sqr(rc); return rc; }
static __device__ __forceinline__ float3 process_diag(float3& d, const float3& r, const float3 c) { float3 rc = r * c; d = d - sqr(rc); return rc; }
static __device__ __forceinline__ float4 process_diag(float4& d, const float4& r, const float4 c) { float4 rc = r * c; d = d - sqr(rc); return rc; }
static __device__ __forceinline__ cuDoubleComplex process_diag(double& d, const double& r, const cuDoubleComplex c)
{ cuDoubleComplex rc = make_cuDoubleComplex(r * c.x, -r * c.y); d = fma(-rc.x, rc.x, fma(-rc.y, rc.y, d)); return rc; }
static __device__ __forceinline__ cuComplex process_diag(float& d, const float& r, const cuComplex c)
{ cuComplex rc = make_cuComplex(r * c.x, -r * c.y); d = fmaf(-rc.x, rc.x, fmaf(-rc.y, rc.y, d)); return rc; }
static __device__ __forceinline__ complex_double2 process_diag(double2& d, const double2& r, const complex_double2 c)
{ complex_double2 rc = device::make_complex_double2(r * c.real, r * make_double2(-c.imag.x, -c.imag.y)); d = d - (sqr(rc.real) + sqr(rc.imag)); return rc; }
static __device__ __forceinline__ complex_float3 process_diag(float3& d, const float3& r, const complex_float3 c)
{ complex_float3 rc = device::make_complex_float3(r * c.real, r * make_float3(-c.imag.x, -c.imag.y, -c.imag.z)); d = d - (sqr(rc.real) + sqr(rc.imag)); return rc; }
static __device__ __forceinline__ complex_float4 process_diag(float4& d, const float4& r, const complex_float4 c)
{ complex_float4 rc = device::make_complex_float4(r * c.real, r * make_float4(-c.imag.x, -c.imag.y, -c.imag.z, -c.imag.w)); d = d - (sqr(rc.real) + sqr(rc.imag)); return rc; }

template <class real_t, class matrix_t> static __device__ __forceinline__ matrix_t conj(const matrix_t& a) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(a.real, make_double2(-a.imag.x, -a.imag.y)); } else
  if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, complex_float3>) { return device::make_complex_float3(a.real, make_float3(-a.imag.x, -a.imag.y, -a.imag.z)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(a.real, make_float4(-a.imag.x, -a.imag.y, -a.imag.z, -a.imag.w)); } else
  { return a; }
}

template <class real_t, class matrix_t> struct add_fl {
  __device__ __forceinline__ matrix_t operator()(const matrix_t& a, const matrix_t& b) {
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, float3>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(a.real + b.real, a.imag + b.imag); } else
    if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, complex_float3>) { return device::make_complex_float3(a.real + b.real, a.imag + b.imag); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(a.real + b.real, a.imag + b.imag); } else
    { return matrix_t(); }
  }
};

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void potrf_iter_kernel(int32_t iterN, int32_t N, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, int4* __restrict__ out) {
  __shared__ idx_t r; __shared__ double e; __shared__ bool p; idx_max<real_t, idx_t> cmp_max; 
  __shared__ typename cub::BlockReduce<matrix_t, BLOCK_THREADS>::TempStorage temp_gemv;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  if (int32_t(threadIdx.x) == 0) { r = *work; e = r.trace; p = 0 <= r.idx; }
  cooperative_groups::this_thread_block().sync();

  auto grid = cooperative_groups::this_grid();
  int32_t M = 0; const int32_t tid = int32_t(grid.thread_rank()), nthreads = int32_t(grid.num_threads());
  while (p) {
    int32_t j = r.idx; matrix_t* A_col_j = &A[int64_t(j) * lda];
    if (0 < M) {
      for (int32_t i = int32_t(blockIdx.x) + 1; i < N; i += int32_t(gridDim.x)) {
        int32_t l = i - int32_t(i <= j); matrix_t threadB = threadIdx.x ? matrix_t() : A_col_j[l], *A_col_i = &A[int64_t(l) * lda];
        for (int32_t k = int32_t(threadIdx.x) - M; k < 0; k += BLOCK_THREADS)
        { threadB = fma_fl<real_t>(A_col_i[k], A_col_j[k], threadB); }
        threadB = cub::BlockReduce<matrix_t, BLOCK_THREADS>(temp_gemv).Reduce(threadB, add_fl<real_t, matrix_t>());
        cooperative_groups::this_thread_block().sync();
        if (int32_t(threadIdx.x) == 0) { A_col_j[l] = threadB; }
      }
      grid.sync();
    }

    ++M; --N; idx_t thread_c = idx_t();
    if (0 < j) {
      if (tid == 0) {
        int32_t t = jpiv[j]; jpiv[j] = jpiv[0]; jpiv[0] = t; real_t D0 = D[0];
        *A_col_j = process_diag(D0, r.real, *A_col_j); thread_c = cmp_max(thread_c, D[j] = D0, j);
      }
      for (int32_t i = tid + 1; i < N; i += nthreads) {
        int32_t l = i + int32_t(j <= i); real_t D_i = D[l]; matrix_t* A_col_i = &A[int64_t(l) * lda];
        *A_col_i = process_diag(D_i, r.real, A_col_j[l]); A_col_i[j] = conj<real_t>(A_col_j[l] = A[l]);
        thread_c = cmp_max(thread_c, D[l] = D_i, l);
      }
      for (int32_t i = nthreads - tid - M; i < 0; i += nthreads)
      { matrix_t t = A_col_j[i]; A_col_j[i] = A[i]; A[i] = t; }
    } else for (int32_t i = tid + 1; i <= N; i += nthreads) {
      real_t D_i = D[i]; A[int64_t(i) * lda] = process_diag(D_i, r.real, A_col_j[i]);
      thread_c = cmp_max(thread_c, D[i] = D_i, i);
    }

    A = &(++A)[lda]; ++jpiv; ++D;
    thread_c = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_c, cmp_max);
    if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_c; } else { thread_c = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      for (int32_t i = int32_t(threadIdx.x) + 1; i < int32_t(gridDim.x); i += BLOCK_THREADS)
      { thread_c = cmp_max(thread_c, work[i]); }
      thread_c = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_c, cmp_max);
      if (int32_t(threadIdx.x) == 0) { thread_c.idx = (e < thread_c.trace) ? (thread_c.idx - 1) : -1; *A = ext<matrix_t>(sqrt_fl(thread_c.real)); *work = thread_c; }
    }
    grid.sync();
    if (int32_t(threadIdx.x) == 0) { r = *work; if (r.idx < 0 || N <= iterN) { p = false; }}
    cooperative_groups::this_thread_block().sync();
  }

  if (tid == 0) { *out = make_int4(M + int32_t(N == 1), __double2loint(r.trace), __double2hiint(r.trace), 0); }
}

template <char mode, class real_t, class matrix_t>
__global__ void matrix_fill_upper_to_full(matrix_t* __restrict__ A, int64_t lda) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  bool pred; if constexpr(mode == 'U') { pred = y < x; } else if constexpr(mode == 'L') { pred = x < y; } else { pred = false; }
  if (pred) { A[x + y * lda] = conj<real_t>(A[y + x * lda]); }
}

template <class real_t, class matrix_t> static inline void write_log(double eps, int32_t K, int32_t N, int32_t rank, double trace, FILE* logger) { if (logger) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, real_t>)
  { std::fprintf(logger, "POTRFP-FP64 [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, real_t>)
  { std::fprintf(logger, "POTRFP-FP32 [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, real_t>)
  { std::fprintf(logger, "POTRFP-FP128-DD [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, real_t>)
  { std::fprintf(logger, "POTRFP-FP96-TF [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, real_t>)
  { std::fprintf(logger, "POTRFP-FP128-QF [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>)
  { std::fprintf(logger, "POTRFP-FP64-COMPLEX [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>)
  { std::fprintf(logger, "POTRFP-FP32-COMPLEX [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>)
  { std::fprintf(logger, "POTRFP-FP128-DD-COMPLEX [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, float3> && std::is_same_v<matrix_t, complex_float3>)
  { std::fprintf(logger, "POTRFP-FP96-TF-COMPLEX [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>)
  { std::fprintf(logger, "POTRFP-FP128-QF-COMPLEX [eps=%.1le] [N=%d] [K=%d] [Cholesky-rank=%d] [Residual^2=%le]\n", eps, N, K, rank, trace); }
}}

template <class idx_t, class real_t, class matrix_t>
static inline std::pair<int32_t, double> potrfp_dispatcher(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, int32_t* rank, void* logger) {
  constexpr int32_t grid_blocks = 2048, block_threads = 128;
  if (fillmode == 'U' || fillmode == 'u')
    matrix_fill_upper_to_full<'U', real_t> <<< dim3(uint32_t(N + 127) >> 7, uint32_t(N)), block_threads, 0, stream >>> (A, lda);
  else if (fillmode == 'L' || fillmode == 'l')
    matrix_fill_upper_to_full<'L', real_t> <<< dim3(uint32_t(N + 127) >> 7, uint32_t(N)), block_threads, 0, stream >>> (A, lda);

  int32_t device_sms = *rank, maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
  uint8_t* diag = &((uint8_t*)D)[98304]; int64_t lda_p1 = lda + int64_t(1);
  double eps2 = eps * eps; void* initArgs[]{ &eps2, &N, &A, &lda_p1, &jpiv, &diag, &D };
  cudaLaunchCooperativeKernel(potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, initArgs, 0, stream);

  int32_t iters = (K == N) ? 1 : (N - K); double trace = 0.0;
  if (iters < N) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_iter_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
    grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), N);
    void* kernelArgs[]{ &iters, &N, &A, &lda, &jpiv, &diag, &D, &rank };
    cudaLaunchCooperativeKernel(potrf_iter_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  } else { iters = int32_t(N == 1); }

  cudaStreamSynchronize(stream); iters = rank[0]; std::memcpy(&trace, &rank[1], sizeof(double));
  write_log<real_t, matrix_t>(eps, K, N, iters, trace, (FILE*)logger);
  return std::make_pair(iters, trace);
}

namespace internal::Cholesky {

  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, float3* A, int32_t lda, int32_t* jpiv, float3* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<float3_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, complex_float3* A, int32_t lda, int32_t* jpiv, float3* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<float3_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }
  std::pair<int32_t, double> potrfp(cudaStream_t stream, char fillmode, double eps, int32_t K, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* r, void* logger)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, eps, K, N, A, lda, jpiv, dev_work, r, logger); }

};
