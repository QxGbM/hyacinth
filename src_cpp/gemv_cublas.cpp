
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

void internal::Cholesky::gemv_cublas_f64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, double* D) {
  if (2 <= M) {
    double rsq = scale[1], minus_rsq = -rsq;
    if (1 <= N)
      cublasDgemv(handle, CUBLAS_OP_T, N, M, &minus_rsq, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &rsq, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    else
      cublasDscal(handle, M, &rsq, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_f64(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(double), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_cublas_f32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, float* D) {
  if (2 <= M) {
    float rsq = scale[1], minus_rsq = -rsq;
    if (1 <= N)
      cublasSgemv(handle, CUBLAS_OP_T, N, M, &minus_rsq, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &rsq, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    else
      cublasSscal(handle, M, &rsq, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_f32(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_cublas_cf64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, double* D) {
  if (2 <= M) {
    std::complex<double> rsq(scale[1], 0.), minus_rsq(-rsq.real(), 0.);
    if (1 <= N)
      cublasZgemv(handle, CUBLAS_OP_C, N, M, (cuDoubleComplex*)&minus_rsq, (cuDoubleComplex*)A, lda, 
        (cuDoubleComplex*)&A[int64_t(j) * int64_t(lda)], 1, (cuDoubleComplex*)&rsq, (cuDoubleComplex*)&A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    else
      cublasZdscal(handle, M, (double*)&rsq, (cuDoubleComplex*)&A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_cf64(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M) {
    scale[1] = 0.;
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(double), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::gemv_cublas_cf32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, float* D) {
  if (2 <= M) {
    std::complex<float> rsq(scale[1], 0.), minus_rsq(-rsq.real(), 0.);
    if (1 <= N)
      cublasCgemv(handle, CUBLAS_OP_C, N, M, (cuComplex*)&minus_rsq, (cuComplex*)A, lda, 
        (cuComplex*)&A[int64_t(j) * int64_t(lda)], 1, (cuComplex*)&rsq, (cuComplex*)&A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    else
      cublasCsscal(handle, M, (float*)&rsq, (cuComplex*)&A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_cf32(stream, j, N, M, scale, A, lda, D);
  }
  else if (1 == M) {
    scale[1] = 0.f;
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
  }
}
