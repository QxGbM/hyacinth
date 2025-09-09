
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <class real_t> struct scalbn_func {
  real_t* __restrict__ A;
  const int32_t* __restrict__ vec_expon;
  int32_t gemm_expon;
  int64_t N, lda;
  scalbn_func(int64_t N, real_t* A, int64_t lda, int32_t gemm_expon, const int32_t* vec_expon) :
    A(A), vec_expon(vec_expon), gemm_expon(gemm_expon), N(N), lda(lda) {}

  __device__ __forceinline__ double scal(int32_t expon, double f) {
    return scalbn(f, expon);
  }
  __device__ __forceinline__ float scal(int32_t expon, float f) {
    return scalbnf(f, expon);
  }
  __device__ __forceinline__ double2 scal(int32_t expon, double2 f) {
    return device::dd::fscalbn(f, expon);
  }
  __device__ __forceinline__ float4 scal(int32_t expon, float4 f) {
    return device::qf::fscalbn(f, expon);
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / N, y = i - N * x;
    real_t f = A[y + x * lda];
    int32_t expon = gemm_expon + vec_expon[x] + vec_expon[y];
    A[y + x * lda] = scal(expon, f);
  }
};

void internal::int8::scal_exponent_f64(cudaStream_t stream, int32_t N, double* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon) {
  thrust::counting_iterator<int64_t> iter(0);
  scalbn_func<double> conv(N, A, lda, gemm_expon, vec_expon);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

void internal::int8::scal_exponent_f32(cudaStream_t stream, int32_t N, float* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon) {
  thrust::counting_iterator<int64_t> iter(0);
  scalbn_func<float> conv(N, A, lda, gemm_expon, vec_expon);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

void internal::int8::scal_exponent_f128_dd(cudaStream_t stream, int32_t N, double2* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon) {
  thrust::counting_iterator<int64_t> iter(0);
  scalbn_func<double2> conv(N, A, lda, gemm_expon, vec_expon);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

void internal::int8::scal_exponent_f128_qf(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon) {
  thrust::counting_iterator<int64_t> iter(0);
  scalbn_func<float4> conv(N, A, lda, gemm_expon, vec_expon);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

template <class real_t, class complex_t> struct convert_func {
  const real_t* __restrict__ A;
  const int32_t* __restrict__ vec_expon;
  complex_t* __restrict__ B;
  int32_t gemm_expon;
  int64_t N, lda, strideA, ldb;
  convert_func(int64_t N, const real_t* A, int64_t lda, int32_t gemm_expon, const int32_t* vec_expon, complex_t* B, int64_t ldb) :
    A(A), vec_expon(vec_expon), B(B), gemm_expon(gemm_expon), N(N), lda(lda), strideA(N * lda), ldb(ldb) {}

  __device__ __forceinline__ cuDoubleComplex conv(int32_t expon, double real, double imagA, double imagAT) {
    return make_cuDoubleComplex(scalbn(real, expon), scalbn(imagA - imagAT, expon));
  }
  __device__ __forceinline__ cuComplex conv(int32_t expon, float real, float imagA, float imagAT) {
    return make_cuComplex(scalbnf(real, expon), scalbnf(imagA - imagAT, expon));
  }
  __device__ __forceinline__ complex_double2 conv(int32_t expon, double2 real, double2 imagA, double2 imagAT) {
    return device::dd::make_complex_double2(device::dd::fscalbn(real, expon), device::dd::fscalbn(device::dd::add(imagA, device::dd::negate(imagAT)), expon));
  }
  __device__ __forceinline__ complex_float4 conv(int32_t expon, float4 real, float4 imagA, float4 imagAT) {
    return device::qf::make_complex_float4(device::qf::fscalbn(real, expon), device::qf::fscalbn(device::qf::add(imagA, device::qf::negate(imagAT)), expon));
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / N, y = i - N * x;
    real_t real = A[y + x * lda];
    real_t imagA = A[strideA + y + x * lda];
    real_t imagAT = A[strideA + x + y * lda];
    int32_t expon = gemm_expon + vec_expon[x] + vec_expon[y];
    B[y + x * ldb] = conv(expon, real, imagA, imagAT);
  }
};

void internal::int8::planar_to_interleave_f64(cudaStream_t stream, int32_t N, const double* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, std::complex<double>* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  convert_func<double, cuDoubleComplex> conv(N, A, lda, gemm_expon, vec_expon, (cuDoubleComplex*)B, ldb);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

void internal::int8::planar_to_interleave_f32(cudaStream_t stream, int32_t N, const float* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, std::complex<float>* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  convert_func<float, cuComplex> conv(N, A, lda, gemm_expon, vec_expon, (cuComplex*)B, ldb);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

void internal::int8::planar_to_interleave_f128_dd(cudaStream_t stream, int32_t N, const double2* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, complex_double2* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  convert_func<double2, complex_double2> conv(N, A, lda, gemm_expon, vec_expon, B, ldb);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}

void internal::int8::planar_to_interleave_f128_qf(cudaStream_t stream, int32_t N, const float4* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, complex_float4* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  convert_func<float4, complex_float4> conv(N, A, lda, gemm_expon, vec_expon, B, ldb);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(N), conv);
}
