
#include <internal.hpp>
#include <cuComplex.h>

void internal::Cholesky::gemv_cublas_f64(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D) {
  if (2 <= M) {
    double one = 1., minus_one = -1.;
    if (1 <= N)
      cublasDgemv(handle, CUBLAS_OP_T, N, M, &minus_one, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &one, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_f64(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(double), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_cublas_f32(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D) {
  if (2 <= M) {
    float one = 1.f, minus_one = -1.f;
    if (1 <= N)
      cublasSgemv(handle, CUBLAS_OP_T, N, M, &minus_one, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &one, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_f32(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_cublas_cf64(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, double* D) {
  if (2 <= M) {
    std::complex<double> one(1., 0.), minus_one(-1., 0.);
    if (1 <= N)
      cublasZgemv(handle, CUBLAS_OP_C, N, M, (cuDoubleComplex*)&minus_one, (cuDoubleComplex*)A, lda, 
        (cuDoubleComplex*)&A[int64_t(j) * int64_t(lda)], 1, (cuDoubleComplex*)&one, (cuDoubleComplex*)&A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_cf64(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M) {
    cudaMemsetAsync(&((double*)scale)[1], 0, sizeof(double), stream);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(double), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::gemv_cublas_cf32(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, float* D) {
  if (2 <= M) {
    std::complex<float> one(1.f, 0.f), minus_one(-1.f, 0.f);
    if (1 <= N)
      cublasCgemv(handle, CUBLAS_OP_C, N, M, (cuComplex*)&minus_one, (cuComplex*)A, lda, 
        (cuComplex*)&A[int64_t(j) * int64_t(lda)], 1, (cuComplex*)&one, (cuComplex*)&A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp_cf32(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M) {
    cudaMemsetAsync(&((float*)scale)[1], 0, sizeof(float), stream);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
  }
}
