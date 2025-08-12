
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <class real_t, class complex_t> struct convert_func {
  const real_t* A;
  complex_t* B;
  int64_t N, lda, strideA, ldb;
  convert_func(int32_t N, const real_t* A, int32_t lda, int64_t strideA, complex_t* B, int32_t ldb) :
    A(A), B(B), N(N), lda(lda), strideA(strideA), ldb(ldb) {}

  __device__ __forceinline__ cuDoubleComplex conv(double real, double imagA, double imagAT) {
    return make_cuDoubleComplex(real, imagA - imagAT);
  }
  __device__ __forceinline__ cuComplex conv(float real, float imagA, float imagAT) {
    return make_cuComplex(real, imagA - imagAT);
  }
  __device__ __forceinline__ complex_double2 conv(double2 real, double2 imagA, double2 imagAT) {
    return device::dd::make_complex_double2(real, device::dd::add(imagA, device::dd::negate(imagAT)));
  }
  __device__ __forceinline__ complex_float4 conv(float4 real, float4 imagA, float4 imagAT) {
    return device::qf::make_complex_float4(real, device::qf::add(imagA, device::qf::negate(imagAT)));
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / N, y = i - N * x;
    real_t real = A[y + x * lda];
    real_t imagA = A[strideA + y + x * lda];
    real_t imagAT = A[strideA + x + y * lda];
    B[y + x * ldb] = conv(real, imagA, imagAT);
  }
};

void internal::int8::planar_to_interleave_f64(cudaStream_t stream, int32_t N, double* A, int32_t lda, int64_t strideA, std::complex<double>* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N) * uint64_t(N), convert_func(N, A, lda, strideA, (cuDoubleComplex*)B, ldb));
}

void internal::int8::planar_to_interleave_f32(cudaStream_t stream, int32_t N, float* A, int32_t lda, int64_t strideA, std::complex<float>* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N) * uint64_t(N), convert_func(N, A, lda, strideA, (cuComplex*)B, ldb));
}

void internal::int8::planar_to_interleave_f128_dd(cudaStream_t stream, int32_t N, double2* A, int32_t lda, int64_t strideA, complex_double2* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N) * uint64_t(N), convert_func(N, A, lda, strideA, B, ldb));
}

void internal::int8::planar_to_interleave_f128_qf(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int64_t strideA, complex_float4* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N) * uint64_t(N), convert_func(N, A, lda, strideA, B, ldb));
}
