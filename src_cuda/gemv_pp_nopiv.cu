
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <cuComplex.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <class real_t, class matrix_t> struct gemv_pp_nopiv {
  real_t* __restrict__ D;
  matrix_t* __restrict__ X;
  matrix_t x0; int64_t incx;
  gemv_pp_nopiv(matrix_t x0, matrix_t* X, int64_t incx, real_t* D) : D(D), X(X), x0(x0), incx(incx) {}

  __device__ __forceinline__ void update_x(double c, double& c_conj, double& d) const {
    c_conj = c; d = fma(-c, c, d);
  }
  __device__ __forceinline__ void update_x(float c, float& c_conj, float& d) const {
    c_conj = c; d = fmaf(-c, c, d);
  }
  __device__ __forceinline__ void update_x(double2 c, double2& c_conj, double2& d) const {
    c_conj = c; d = device::dd::add(device::dd::mul(device::dd::negate(c), c), d);
  }
  __device__ __forceinline__ void update_x(float4 c, float4& c_conj, float4& d) const {
    c_conj = c; d = device::qf::add(device::qf::mul(device::qf::negate(c), c), d);
  }
  __device__ __forceinline__ void update_x(cuDoubleComplex c, cuDoubleComplex& c_conj, double& d) const {
    c_conj = make_cuDoubleComplex(c.x, -c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d));
  }
  __device__ __forceinline__ void update_x(cuComplex c, cuComplex& c_conj, float& d) const {
    c_conj = make_cuComplex(c.x, -c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d));
  }
  __device__ __forceinline__ void update_x(complex_double2 c, complex_double2& c_conj, double2& d) const {
    using device::dd::add, device::dd::mul, device::dd::negate;
    c_conj = device::dd::make_complex_double2(c.real, negate(c.imag));
    d = add(mul(negate(c.real), c.real), add(mul(negate(c.imag), c.imag), d));
  }
  __device__ __forceinline__ void update_x(complex_float4 c, complex_float4& c_conj, float4& d) const {
    using device::qf::add, device::qf::mul, device::qf::negate;
    c_conj = device::qf::make_complex_float4(c.real, negate(c.imag));
    d = add(mul(negate(c.real), c.real), add(mul(negate(c.imag), c.imag), d));
  }

  __device__ __forceinline__ void operator()(int64_t i) const {
    if (0 < i) update_x(X[i], X[i * incx], D[i]);
      else X[0] = x0;
  }
};

void internal::Cholesky::gemv_pp_nopiv_f64(cudaStream_t stream, int32_t M, int32_t N, double sq, double* A, int32_t lda, double* D) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<double, double>(sq, &A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_f32(cudaStream_t stream, int32_t M, int32_t N, float sq, float* A, int32_t lda, float* D) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<float, float>(sq, &A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_f128_dd(cudaStream_t stream, int32_t M, int32_t N, double2 sq, double2* A, int32_t lda, double2* D) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<double2, double2>(sq, &A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_f128_qf(cudaStream_t stream, int32_t M, int32_t N, float4 sq, float4* A, int32_t lda, float4* D) {
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<float4, float4>(sq, &A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_cf64(cudaStream_t stream, int32_t M, int32_t N, double sq, std::complex<double>* A, int32_t lda, double* D) {
  cuDoubleComplex sqc = make_cuDoubleComplex(sq, 0.);
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<double, cuDoubleComplex>(sqc, (cuDoubleComplex*)&A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_cf32(cudaStream_t stream, int32_t M, int32_t N, float sq, std::complex<float>* A, int32_t lda, float* D) {
  cuComplex sqc = make_cuComplex(sq, 0.f);
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<float, cuComplex>(sqc, (cuComplex*)&A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_cf128_dd(cudaStream_t stream, int32_t M, int32_t N, double2 sq, complex_double2* A, int32_t lda, double2* D) {
  complex_double2 sqc = device::dd::make_complex_double2(sq, make_double2(0., 0.));
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<double2, complex_double2>(sqc, &A[M], lda, D));
}

void internal::Cholesky::gemv_pp_nopiv_cf128_qf(cudaStream_t stream, int32_t M, int32_t N, float4 sq, complex_float4* A, int32_t lda, float4* D) {
  complex_float4 sqc = device::qf::make_complex_float4(sq, make_float4(0.f, 0.f, 0.f, 0.f));
  thrust::counting_iterator<int64_t> iter(0);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(N), gemv_pp_nopiv<float4, complex_float4>(sqc, &A[M], lda, D));
}
