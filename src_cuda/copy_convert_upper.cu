
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

struct convert_fp {
  __device__ __forceinline__ void operator()(double a, double& b) { b = a; }
  __device__ __forceinline__ void operator()(float a, double& b) { b = double(a); }
  __device__ __forceinline__ void operator()(double2 a, double& b) { b = a.x + a.y; }
  __device__ __forceinline__ void operator()(float4 a, double& b) {
    b = double(a.x) + double(a.y) + double(a.z) + double(a.w); }

  __device__ __forceinline__ void operator()(double a, float& b) { b = float(a); }
  __device__ __forceinline__ void operator()(float a, float& b) { b = a; }

  __device__ __forceinline__ void operator()(double a, double2& b) { b = make_double2(a, 0.); }
  __device__ __forceinline__ void operator()(double2 a, double2& b) { b = a; }
  __device__ __forceinline__ void operator()(float4 a, double2& b) {
    b = device::dd::normalize(make_double2(double(a.x) + double(a.y), double(a.z) + double(a.w))); }

  __device__ __forceinline__ void operator()(double a, float4& b) {
    float a1 = float(a); double c = a - double(a1);
    float a2 = float(c), a3 = float(c - double(a2));
    b = make_float4(a1, a2, a3, 0.f);
  }
  __device__ __forceinline__ void operator()(double2 a, float4& b) {
    float4 c, d; operator()(a.x, c); operator()(a.y, d); b = device::qf::add(c, d); }
  __device__ __forceinline__ void operator()(float4 a, float4& b) { b = a; }
};

template <int32_t M, class typeA, class typeB> struct upper_tri_conv_copy {
  const typeA* A;
  typeB* B;
  uint64_t lda, ldb;
  upper_tri_conv_copy(const typeA* A, int32_t lda, typeB* B, int32_t ldb) :
    A(A), B(B), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(uint64_t i) {
    float r = sqrtf(__ull2float_rz((i << (4 - M)) + 1));
    uint64_t x = (uint64_t(r) - 1) >> 1;
    uint64_t base = (x * (x + 1)) >> (2 - M);
    uint64_t y = i - base;

    convert_fp conv;
    conv(A[y + x * lda], B[y + x * ldb]);
  }
};

template <class typeA, class typeB> struct rect_conv_copy {
  const typeA* A;
  typeB* B;
  uint64_t M, lda, ldb;
  rect_conv_copy(int32_t M, const typeA* A, int32_t lda, typeB* B, int32_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(uint64_t i) {
    uint64_t x = i / M, y = i - x * M;
    convert_fp conv;
    conv(A[y + x * lda], B[y + x * ldb]);
  }
};

template <class typeA, class typeB>
inline void upper_tri_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const typeA* A, int32_t lda, typeB* B, int32_t ldb) {
  uint64_t iter_items = uint64_t(N) * uint64_t(N + 1);
  thrust::counting_iterator<uint64_t> iter(0);

  if (M == 1) {
    upper_tri_conv_copy<1, typeA, typeB> conv(A, lda, B, ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items >> 1, conv);
  }
  else if (M == 2) {
    upper_tri_conv_copy<2, typeA, typeB> conv(A, lda, B, ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, conv);
  }
}

void device::copy_upper_triangular(cudaStream_t stream, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB) {
  if (precA == Precision::FP64) {
    if (precB == Precision::FP64)
      upper_tri_dispatcher<double, double>(stream, M, N, (const double*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP32)
      upper_tri_dispatcher<double, float>(stream, M, N, (const double*)A, lda, (float*)B, ldb);
    else if (precB == Precision::FP128_DD)
      upper_tri_dispatcher<double, double2>(stream, M, N, (const double*)A, lda, (double2*)B, ldb);
    else if (precB == Precision::FP128_QF)
      upper_tri_dispatcher<double, float4>(stream, M, N, (const double*)A, lda, (float4*)B, ldb);
  }
  else if (precA == Precision::FP32) {
    if (precB == Precision::FP64)
      upper_tri_dispatcher<float, double>(stream, M, N, (const float*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP32)
      upper_tri_dispatcher<float, float>(stream, M, N, (const float*)A, lda, (float*)B, ldb);
  }
  else if (precA == Precision::FP128_DD) {
    if (precB == Precision::FP64)
      upper_tri_dispatcher<double2, double>(stream, M, N, (const double2*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP128_DD)
      upper_tri_dispatcher<double2, double2>(stream, M, N, (const double2*)A, lda, (double2*)B, ldb);
    else if (precB == Precision::FP128_QF)
      upper_tri_dispatcher<double2, float4>(stream, M, N, (const double2*)A, lda, (float4*)B, ldb);
  }
  else if (precA == Precision::FP128_QF) {
    if (precB == Precision::FP64)
      upper_tri_dispatcher<float4, double>(stream, M, N, (const float4*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP128_DD)
      upper_tri_dispatcher<float4, double2>(stream, M, N, (const float4*)A, lda, (double2*)B, ldb);
    else if (precB == Precision::FP128_QF)
      upper_tri_dispatcher<float4, float4>(stream, M, N, (const float4*)A, lda, (float4*)B, ldb);
  }
}

template <class typeA, class typeB>
inline void rect_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const typeA* A, int32_t lda, typeB* B, int32_t ldb) {
  uint64_t iter_items = uint64_t(N) * uint64_t(M);
  thrust::counting_iterator<uint64_t> iter(0);

  rect_conv_copy<typeA, typeB> conv(M, A, lda, B, ldb);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, conv);
}

void device::copy_rectangle(cudaStream_t stream, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB) {
  if (precA == Precision::FP64) {
    if (precB == Precision::FP64)
      rect_dispatcher<double, double>(stream, M, N, (const double*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP32)
      rect_dispatcher<double, float>(stream, M, N, (const double*)A, lda, (float*)B, ldb);
    else if (precB == Precision::FP128_DD)
      rect_dispatcher<double, double2>(stream, M, N, (const double*)A, lda, (double2*)B, ldb);
    else if (precB == Precision::FP128_QF)
      rect_dispatcher<double, float4>(stream, M, N, (const double*)A, lda, (float4*)B, ldb);
  }
  else if (precA == Precision::FP32) {
    if (precB == Precision::FP64)
      rect_dispatcher<float, double>(stream, M, N, (const float*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP32)
      rect_dispatcher<float, float>(stream, M, N, (const float*)A, lda, (float*)B, ldb);
  }
  else if (precA == Precision::FP128_DD) {
    if (precB == Precision::FP64)
      rect_dispatcher<double2, double>(stream, M, N, (const double2*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP128_DD)
      rect_dispatcher<double2, double2>(stream, M, N, (const double2*)A, lda, (double2*)B, ldb);
    else if (precB == Precision::FP128_QF)
      rect_dispatcher<double2, float4>(stream, M, N, (const double2*)A, lda, (float4*)B, ldb);
  }
  else if (precA == Precision::FP128_QF) {
    if (precB == Precision::FP64)
      rect_dispatcher<float4, double>(stream, M, N, (const float4*)A, lda, (double*)B, ldb);
    else if (precB == Precision::FP128_DD)
      rect_dispatcher<float4, double2>(stream, M, N, (const float4*)A, lda, (double2*)B, ldb);
    else if (precB == Precision::FP128_QF)
      rect_dispatcher<float4, float4>(stream, M, N, (const float4*)A, lda, (float4*)B, ldb);
  }
}
