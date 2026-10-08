
#include <hyacin.h>
#include <internal.hpp>
#include <ext_arith.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
__device__ __forceinline__ double2 conv_i64_dd(uint64_t i) { return make_double2(scalbn(double(int32_t(i >> 32)), 32), double(uint32_t(i))); }
template <int32_t orderA> __device__ __forceinline__ double conv_a63_f64(uint64_t const (&a)[orderA], int32_t e) {
  double res = double(int64_t(a[orderA - 1]));
  if constexpr(2 < orderA) { double2 i = conv_i64_dd(a[1]); res = (scalbn(res, 63) + i.x) + i.y; }
  if constexpr(1 < orderA) { double2 i = conv_i64_dd(a[0]); res = (scalbn(res, 63) + i.x) + i.y; }
  return scalbn(res, e);
}

template <int32_t orderA> __device__ __forceinline__ double2 conv_a63_dd(uint64_t const (&a)[orderA], int32_t e) {
  double2 res = device::renormalize(conv_i64_dd(a[orderA - 1]));
  if constexpr(2 < orderA) { res = device::add(make_double2(scalbn(res.x, 63), scalbn(res.y, 63)), device::renormalize(conv_i64_dd(a[1]))); }
  if constexpr(1 < orderA) { res = device::add(make_double2(scalbn(res.x, 63), scalbn(res.y, 63)), device::renormalize(conv_i64_dd(a[0]))); }
  return make_double2(scalbn(res.x, e), scalbn(res.y, e));
}

__device__ __forceinline__ float4 conv_i64_qf_m126(uint64_t i) {
  constexpr uint32_t i24 = 0xffffff;
  float x = scalbnf(float(int16_t(i >> 48)), -78), y = scalbnf(float(uint32_t(i >> 24) & i24), -102), z = scalbnf(float(uint32_t(i) & i24), -126);
  device::two_sum(x, y, x, y); device::two_sum(x, z, x, z); device::two_sum(y, z, y, z);
  return make_float4(x, y, z, 0.f);
}

template <int32_t orderA> __device__ __forceinline__ float4 conv_a63_qf(uint64_t const (&a)[orderA], int32_t e) {
  float4 res = conv_i64_qf_m126(a[orderA - 1]);
  if constexpr(2 < orderA) { res = device::add(make_float4(scalbnf(res.x, 63), scalbnf(res.y, 63), scalbnf(res.z, 63), scalbnf(res.w, 63)), conv_i64_qf_m126(a[1])); }
  if constexpr(1 < orderA) { res = device::add(make_float4(scalbnf(res.x, 63), scalbnf(res.y, 63), scalbnf(res.z, 63), scalbnf(res.w, 63)), conv_i64_qf_m126(a[0])); }
  e += 126; return make_float4(scalbnf(res.x, e), scalbnf(res.y, e), scalbnf(res.z, e), scalbnf(res.w, e));
}

template <int32_t orderA, int32_t Complex, class matrix_t> __device__ __forceinline__ matrix_t deq_i(const uint64_t* A, int64_t stride, int32_t e) {
  if constexpr(Complex) {
    uint64_t r[orderA], i[orderA];
    if constexpr(0 < orderA) { r[0] = *A; } if constexpr(1 < orderA) { r[1] = *(A += stride); } if constexpr(2 < orderA) { r[2] = *(A += stride); }
    if constexpr(0 < orderA) { i[0] = *(A += stride); } if constexpr(1 < orderA) { i[1] = *(A += stride); } if constexpr(2 < orderA) { i[2] = *(A += stride); }
    if constexpr(std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(conv_a63_f64(r, e), conv_a63_f64(i, e)); } else
    if constexpr(std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(float(conv_a63_f64(r, e)), float(conv_a63_f64(i, e))); } else
    if constexpr(std::is_same_v<matrix_t, complex_double2>) { return device::make_complex_double2(conv_a63_dd(r, e), conv_a63_dd(i, e)); } else
    if constexpr(std::is_same_v<matrix_t, complex_float4>) { return device::make_complex_float4(conv_a63_qf(r, e), conv_a63_qf(i, e)); } else
    { return matrix_t(); }
  } else {
    uint64_t a[orderA];
    if constexpr(0 < orderA) { a[0] = *A; } if constexpr(1 < orderA) { a[1] = *(A += stride); } if constexpr(2 < orderA) { a[2] = *(A += stride); }
    if constexpr(std::is_same_v<matrix_t, double>) { return conv_a63_f64(a, e); } else
    if constexpr(std::is_same_v<matrix_t, float>) { return float(conv_a63_f64(a, e)); } else
    if constexpr(std::is_same_v<matrix_t, double2>) { return conv_a63_dd(a, e); } else
    if constexpr(std::is_same_v<matrix_t, float4>) { return conv_a63_qf(a, e); } else
    { return matrix_t(); }
  }
}

__device__ __forceinline__ cuDoubleComplex conj(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
__device__ __forceinline__ cuComplex conj(cuComplex a) { return make_cuComplex(a.x, -a.y); }
__device__ __forceinline__ complex_double2 conj(complex_double2 a) { return device::make_complex_double2(a.real, make_double2(-a.imag.x, -a.imag.y)); }
__device__ __forceinline__ complex_float4 conj(complex_float4 a) { return device::make_complex_float4(a.real, make_float4(-a.imag.x, -a.imag.y, -a.imag.z, -a.imag.w)); }

template<int32_t orderA, int32_t Complex, class matrix_t>
__global__ void triangle_unpack_dequantize_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t strideA, const int32_t* __restrict__ vexp, matrix_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    int32_t ex = vexp[x], ey = vexp[y];
    matrix_t f = (ex == int_max || ey == int_max) ? matrix_t() : deq_i<orderA, Complex, matrix_t>(&A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)], strideA, -(ex + ey));
    if constexpr(Complex) { B[x + (y * ldb)] = conj(B[y + (x * ldb)] = f); } else { B[x + (y * ldb)] = B[y + (x * ldb)] = f; }
  }
}

template<int32_t Complex, class matrix_t>
inline void tp_deq_dispatcher(cudaStream_t stream, int32_t N, int32_t orderA, const uint64_t* A, const int32_t* vexp, matrix_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t N64 = int64_t(N), strideA = (N64 * N64 + N64) / int64_t(2), ldb64 = int64_t(ldb);
  switch (orderA) {
    case 1: triangle_unpack_dequantize_kernel<1, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, vexp, B, ldb64); return;
    case 2: triangle_unpack_dequantize_kernel<2, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, vexp, B, ldb64); return;
    case 3: triangle_unpack_dequantize_kernel<3, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, vexp, B, ldb64); return;
    default: return;
  }
}

extern "C" void hyacinXdequantize(const hyacinHandle_t* handle, int32_t N, int32_t orderC, const uint64_t* C, const int32_t* vexp, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0) { return; }
  Timer::register_replicate_kernel(handle->cudaStream, handle->timer);
  switch (Gtype) {
    case HYACIN_F64: tp_deq_dispatcher<0>(handle->cudaStream, N, orderC, C, vexp, (double*)G, ldg); return;
    case HYACIN_F32: tp_deq_dispatcher<0>(handle->cudaStream, N, orderC, C, vexp, (float*)G, ldg); return;
    case HYACIN_DD: tp_deq_dispatcher<0>(handle->cudaStream, N, orderC, C, vexp, (double2*)G, ldg); return;
    case HYACIN_QF: tp_deq_dispatcher<0>(handle->cudaStream, N, orderC, C, vexp, (float4*)G, ldg); return;
    case HYACIN_F64_COMPLEX: tp_deq_dispatcher<1>(handle->cudaStream, N, orderC, C, vexp, (cuDoubleComplex*)G, ldg); return;
    case HYACIN_F32_COMPLEX: tp_deq_dispatcher<1>(handle->cudaStream, N, orderC, C, vexp, (cuComplex*)G, ldg); return;
    case HYACIN_DD_COMPLEX: tp_deq_dispatcher<1>(handle->cudaStream, N, orderC, C, vexp, (complex_double2*)G, ldg); return;
    case HYACIN_QF_COMPLEX: tp_deq_dispatcher<1>(handle->cudaStream, N, orderC, C, vexp, (complex_float4*)G, ldg); return;
    default: return;
  }
}
