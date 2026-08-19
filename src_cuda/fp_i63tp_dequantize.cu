
#include <hyacin.h>
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
template <int32_t orderA, int32_t Complex, class matrix_t>
__device__ __forceinline__ void deq_i(const uint64_t* A, int64_t stride, int32_t e, matrix_t* f) {
  if constexpr(Complex) {
    uint64_t r[orderA], i[orderA];
    if constexpr(0 < orderA) { r[0] = *A; } if constexpr(1 < orderA) { r[1] = *(A += stride); } if constexpr(2 < orderA) { r[2] = *(A += stride); }
    if constexpr(0 < orderA) { i[0] = *(A += stride); } if constexpr(1 < orderA) { i[1] = *(A += stride); } if constexpr(2 < orderA) { i[2] = *(A += stride); }
    if constexpr(std::is_same_v<matrix_t, cuDoubleComplex>) { *f = make_cuDoubleComplex(device::dd::conv_a63_f64(r, e), device::dd::conv_a63_f64(i, e)); } else
    if constexpr(std::is_same_v<matrix_t, cuComplex>) { *f = make_cuComplex(float(device::dd::conv_a63_f64(r, e)), float(device::dd::conv_a63_f64(i, e))); } else
    if constexpr(std::is_same_v<matrix_t, complex_double2>) { *f = device::dd::make_complex_double2(device::dd::conv_a63_dd(r, e), device::dd::conv_a63_dd(i, e)); } else
    if constexpr(std::is_same_v<matrix_t, complex_float4>) { *f = device::qf::make_complex_float4(device::qf::conv_a63_qf(r, e), device::qf::conv_a63_qf(i, e)); }
  }
  else {
    uint64_t a[orderA];
    if constexpr(0 < orderA) { a[0] = *A; } if constexpr(1 < orderA) { a[1] = *(A += stride); } if constexpr(2 < orderA) { a[2] = *(A += stride); }
    if constexpr(std::is_same_v<matrix_t, double>) { *f = device::dd::conv_a63_f64(a, e); } else
    if constexpr(std::is_same_v<matrix_t, float>) { *f = float(device::dd::conv_a63_f64(a, e)); } else
    if constexpr(std::is_same_v<matrix_t, double2>) { *f = device::dd::conv_a63_dd(a, e); } else
    if constexpr(std::is_same_v<matrix_t, float4>) { *f = device::qf::conv_a63_qf(a, e); }
  }
}

template<int32_t orderA, int32_t Complex, class matrix_t>
__global__ void triangle_unpack_dequantize_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t strideA, const int32_t* __restrict__ vexp, matrix_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    B = &B[y + (x * ldb)];
    int32_t ex = vexp[x], ey = vexp[y];
    if (ex == int_max || ey == int_max) { *B = matrix_t(); }
      else { deq_i<orderA, Complex>(&A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)], strideA, -(ex + ey), B); }
  }
}

template<int32_t Complex, class matrix_t>
inline void tp_deq_dispatcher(cudaStream_t stream, int32_t N, int32_t orderA, const uint64_t* A, const int32_t* vexp, matrix_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t N64 = int64_t(N), strideA = (N64 * N64 + N64) / int64_t(2), ldb64 = int64_t(ldb);
  switch(orderA) {
    case 1: triangle_unpack_dequantize_kernel<1, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, vexp, B, ldb64); return;
    case 2: triangle_unpack_dequantize_kernel<2, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, vexp, B, ldb64); return;
    case 3: triangle_unpack_dequantize_kernel<3, Complex> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, vexp, B, ldb64); return;
    default: return;
  }
}

extern "C" void hyacinXdequantize(hyacinHandle_t handle, int32_t N, int32_t orderC, const uint64_t* C, const int32_t* vexp, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0) { return; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch(Gtype) {
    case HYACIN_F64: tp_deq_dispatcher<0>(handle.cudaStream, N, orderC, C, vexp, (double*)G, ldg); return;
    case HYACIN_F32: tp_deq_dispatcher<0>(handle.cudaStream, N, orderC, C, vexp, (float*)G, ldg); return;
    case HYACIN_DD: tp_deq_dispatcher<0>(handle.cudaStream, N, orderC, C, vexp, (double2*)G, ldg); return;
    case HYACIN_QF: tp_deq_dispatcher<0>(handle.cudaStream, N, orderC, C, vexp, (float4*)G, ldg); return;
    case HYACIN_F64_COMPLEX: tp_deq_dispatcher<1>(handle.cudaStream, N, orderC, C, vexp, (cuDoubleComplex*)G, ldg); return;
    case HYACIN_F32_COMPLEX: tp_deq_dispatcher<1>(handle.cudaStream, N, orderC, C, vexp, (cuComplex*)G, ldg); return;
    case HYACIN_DD_COMPLEX: tp_deq_dispatcher<1>(handle.cudaStream, N, orderC, C, vexp, (complex_double2*)G, ldg); return;
    case HYACIN_QF_COMPLEX: tp_deq_dispatcher<1>(handle.cudaStream, N, orderC, C, vexp, (complex_float4*)G, ldg); return;
    default: return;
  }
}
