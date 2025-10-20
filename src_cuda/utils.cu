
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
  int64_t M, lda, ldb;
  rect_conv_copy(int64_t M, const typeA* A, int64_t lda, typeB* B, int64_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}

  __device__ __forceinline__ void add(double a, double& b) { b = a + b; }
  __device__ __forceinline__ void add(float a, float& b) { b = a + b; }
  __device__ __forceinline__ void add(double2 a, double2& b) { b = device::dd::add(a, b); }
  __device__ __forceinline__ void add(float4 a, float4& b) { b = device::qf::add(a, b); }
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    convert_fp conv;
    if constexpr(beta) { typeB a; conv(A[y + x * lda], a); add(a, B[y + x * ldb]); }
      else conv(A[y + x * lda], B[y + x * ldb]);
  }
};

template <class typeA, class typeB>
inline void conv_copy_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const typeA* A, int32_t lda, int32_t beta, typeB* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  if (beta)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(M) * int64_t(N), rect_conv_copy<1, typeA, typeB>(M, A, lda, B, ldb));
  else
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(M) * int64_t(N), rect_conv_copy<0, typeA, typeB>(M, A, lda, B, ldb));
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

const int32_t clen = 16384;
__constant__ int32_t cpiv[clen];

template <int32_t sc0ga1, class elem_t> struct permute_copy {
  const elem_t* __restrict__ A;
  elem_t* __restrict__ B;
  int64_t M, lda, ldb;
  permute_copy(int64_t M, const elem_t* A, int64_t lda, elem_t* B, int64_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int64_t px = int64_t(cpiv[x] - 1);
    if constexpr(sc0ga1) B[y + x * ldb] = A[y + px * lda];
      else B[y + px * ldb] = A[y + x * lda];
  }
};

void device::copy_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec) {
  for (int32_t i = 0; i < N; i += clen) {
    int32_t iter_items = std::min(N - i, clen);
    int64_t elements = int64_t(iter_items) * int64_t(M);
    thrust::counting_iterator<int64_t> iter(0);
    cudaMemcpyToSymbolAsync(cpiv, &jpiv[i], int64_t(iter_items) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);

    if (prec == Precision::FP64) {
      permute_copy<1, int64_t> perm(M, (const int64_t*)A, lda, &((int64_t*)B)[int64_t(i) * int64_t(ldb)], ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
    else if (prec == Precision::FP32) {
      permute_copy<1, int32_t> perm(M, (const int32_t*)A, lda, &((int32_t*)B)[int64_t(i) * int64_t(ldb)], ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
  }
}

void device::copy_scatter(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec) {
  for (int32_t i = 0; i < N; i += clen) {
    int32_t iter_items = std::min(N - i, clen);
    int64_t elements = int64_t(iter_items) * int64_t(M);
    thrust::counting_iterator<int64_t> iter(0);
    cudaMemcpyToSymbolAsync(cpiv, &jpiv[i], int64_t(iter_items) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);

    if (prec == Precision::FP64) {
      permute_copy<0, int64_t> perm(M, &((const int64_t*)A)[int64_t(i) * int64_t(lda)], lda, (int64_t*)B, ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
    else if (prec == Precision::FP32) {
      permute_copy<0, int32_t> perm(M, &((const int32_t*)A)[int64_t(i) * int64_t(lda)], lda, (int32_t*)B, ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
  }
}

template <class real_t> struct identity {
  real_t* __restrict__ A;
  real_t zero, one;
  int64_t M, lda, strideD;
  identity(real_t zero, real_t one, int64_t M, real_t* A, int64_t lda, int64_t strideD) :
    A(A), zero(zero), one(one), M(M), lda(lda), strideD(strideD) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int32_t diag = int32_t(i % strideD == int64_t(0));
    A[y + x * lda] = diag ? one : zero;
  }
};

void device::strided_identity(cudaStream_t stream, int32_t M, int32_t N, int32_t strideD, void* A, int32_t lda, Precision prec) {
  int64_t iter_items = int64_t(N) * int64_t(M);
  thrust::counting_iterator<int64_t> iter(0);

  if (prec == Precision::FP64) {
    identity<double> id(0., 1., M, (double*)A, lda, strideD);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
  else if (prec == Precision::FP32) {
    identity<float> id(0.f, 1.f, M, (float*)A, lda, strideD);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
}
