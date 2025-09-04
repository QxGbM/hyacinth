
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
  __device__ __forceinline__ void operator()(double2 a, double& b) { b = device::dd::dd2double(a); }
  __device__ __forceinline__ void operator()(float4 a, double& b) { b = device::qf::qf2double(a); }

  __device__ __forceinline__ void operator()(double a, float& b) { b = float(a); }
  __device__ __forceinline__ void operator()(float a, float& b) { b = a; }
  __device__ __forceinline__ void operator()(double2 a, float& b) { b = float(a.x) + float(a.y); }
  __device__ __forceinline__ void operator()(float4 a, float& b) { b = a.x + a.y + a.z + a.w; }

  __device__ __forceinline__ void operator()(double a, double2& b) { b = device::dd::double2dd(a); }
  __device__ __forceinline__ void operator()(float a, double2& b) { b = make_double2(double(a), 0.); }
  __device__ __forceinline__ void operator()(double2 a, double2& b) { b = a; }
  __device__ __forceinline__ void operator()(float4 a, double2& b) { b = device::dd::qf2dd(a); }

  __device__ __forceinline__ void operator()(double a, float4& b) { b = device::qf::double2qf(a); }
  __device__ __forceinline__ void operator()(float a, float4& b) { b = make_float4(a, 0.f, 0.f, 0.f); }
  __device__ __forceinline__ void operator()(double2 a, float4& b) { b = device::dd::dd2qf(a); }
  __device__ __forceinline__ void operator()(float4 a, float4& b) { b = a; }
};

template <int32_t beta, class typeA, class typeB> struct rect_conv_copy {
  const typeA* __restrict__ A;
  typeB* __restrict__ B;
  uint64_t M, lda, ldb;
  rect_conv_copy(int32_t M, const typeA* A, int32_t lda, typeB* B, int32_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}

  __device__ __forceinline__ void add(double a, double& b) { b = a + b; }
  __device__ __forceinline__ void add(float a, float& b) { b = a + b; }
  __device__ __forceinline__ void add(double2 a, double2& b) { b = device::dd::add(a, b); }
  __device__ __forceinline__ void add(float4 a, float4& b) { b = device::qf::add(a, b); }
  
  __device__ __forceinline__ void operator()(uint64_t i) {
    uint64_t x = i / M, y = i - x * M;
    convert_fp conv;
    if constexpr(beta) 
    { typeB a; conv(A[y + x * lda], a); add(a, B[y + x * ldb]); }
    else
    conv(A[y + x * lda], B[y + x * ldb]);
  }
};

template <class typeA, class typeB>
inline void conv_copy_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const typeA* A, int32_t lda, int32_t beta, typeB* B, int32_t ldb) {
  thrust::counting_iterator<uint64_t> iter(0);
  if (beta)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(M) * uint64_t(N), rect_conv_copy<1, typeA, typeB>(M, A, lda, B, ldb));
  else
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(M) * uint64_t(N), rect_conv_copy<0, typeA, typeB>(M, A, lda, B, ldb));
}

template <class typeA>
inline void conv_copy_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const typeA* A, int32_t lda, int32_t beta, void* B, int32_t ldb, device::Precision precB) {
  switch (precB) {
    case device::Precision::FP64: conv_copy_dispatcher<typeA, double>(stream, M, N, A, lda, beta, (double*)B, ldb); break;
    case device::Precision::FP32: conv_copy_dispatcher<typeA, float>(stream, M, N, A, lda, beta, (float*)B, ldb); break;
    case device::Precision::FP128_DD: conv_copy_dispatcher<typeA, double2>(stream, M, N, A, lda, beta, (double2*)B, ldb); break;
    case device::Precision::FP128_QF: conv_copy_dispatcher<typeA, float4>(stream, M, N, A, lda, beta, (float4*)B, ldb); break;
    default: break;
  }
}

void device::convert_and_copy(cudaStream_t stream, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, int32_t beta, void* B, int32_t ldb, Precision precB) {
  switch (precA) {
    case Precision::FP64: conv_copy_dispatcher<double>(stream, M, N, (const double*)A, lda, beta, B, ldb, precB); break;
    case Precision::FP32: conv_copy_dispatcher<float>(stream, M, N, (const float*)A, lda, beta, B, ldb, precB); break;
    case Precision::FP128_DD: conv_copy_dispatcher<double2>(stream, M, N, (const double2*)A, lda, beta, B, ldb, precB); break;
    case Precision::FP128_QF: conv_copy_dispatcher<float4>(stream, M, N, (const float4*)A, lda, beta, B, ldb, precB); break;
    default: break;
  }
}

template <int32_t sc0ga1, class real_t> struct permute_copy {
  const real_t* __restrict__ A;
  real_t* __restrict__ B;
  const int32_t* jpiv;
  uint64_t M, lda, ldb;
  permute_copy(int32_t M, const int32_t* jpiv, const real_t* A, int32_t lda, real_t* B, int32_t ldb) :
    A(A), B(B), jpiv(jpiv), M(M), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(uint64_t i) {
    uint64_t x = i / M, y = i - x * M;
    uint64_t px = uint64_t(jpiv[x] - 1);
    if constexpr(sc0ga1)
      B[y + x * ldb] = A[y + px * lda];
    else
      B[y + px * ldb] = A[y + x * lda];
  }
};

void device::copy_permute(cudaStream_t stream, int32_t sc0ga1, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec) {
  uint64_t iter_items = uint64_t(N) * uint64_t(M);
  thrust::counting_iterator<uint64_t> iter(0);

  if (prec == Precision::FP64 && !sc0ga1) {
    permute_copy<0, double> perm(M, jpiv, (const double*)A, lda, (double*)B, ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, perm);
  }
  else if (prec == Precision::FP64 && sc0ga1) {
    permute_copy<1, double> perm(M, jpiv, (const double*)A, lda, (double*)B, ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, perm);
  }
  else if (prec == Precision::FP32 && !sc0ga1) {
    permute_copy<0, float> perm(M, jpiv, (const float*)A, lda, (float*)B, ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, perm);
  }
  else if (prec == Precision::FP32 && sc0ga1) {
    permute_copy<1, float> perm(M, jpiv, (const float*)A, lda, (float*)B, ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, perm);
  }
}

template <class real_t> struct identity {
  real_t* __restrict__ A;
  real_t zero, one;
  uint64_t M, lda, strideD;
  identity(real_t zero, real_t one, int32_t M, real_t* A, int32_t lda, int32_t strideD) :
    A(A), zero(zero), one(one), M(M), lda(lda), strideD(strideD) {}
  
  __device__ __forceinline__ void operator()(uint64_t i) {
    uint64_t x = i / M, y = i - x * M;
    int32_t diag = int32_t(i % strideD == uint64_t(0));
    A[y + x * lda] = diag ? one : zero;
  }
};

void device::strided_identity(cudaStream_t stream, int32_t M, int32_t N, int32_t strideD, void* A, int32_t lda, Precision prec) {
  uint64_t iter_items = uint64_t(N) * uint64_t(M);
  thrust::counting_iterator<uint64_t> iter(0);

  if (prec == Precision::FP64) {
    identity<double> id(0., 1., M, (double*)A, lda, strideD);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
  else if (prec == Precision::FP32) {
    identity<float> id(0.f, 1.f, M, (float*)A, lda, strideD);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
}
