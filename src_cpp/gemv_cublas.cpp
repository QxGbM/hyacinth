
#include <internal.hpp>
#include <cuComplex.h>

void internal::Cholesky::gemv_scal_f64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B, double* D) {
  double minus_one = -1., one = 1.;
  if (1 <= N && 2 <= M)
    cublasDgemv(handle, CUBLAS_OP_T, N, M - 1, &minus_one, &A[lda], lda, A, 1, &one, &B[1], 1);
  reduce_scal_f64(stream, scale, M, 1, B, lda, D);
}

void internal::Cholesky::gemv_scal_f32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t M, int32_t N, const float* A, int32_t lda, float* B, float* D) {
  float minus_one = -1.f, one = 1.f;
  if (1 <= N && 2 <= M)
    cublasSgemv(handle, CUBLAS_OP_T, N, M - 1, &minus_one, &A[lda], lda, A, 1, &one, &B[1], 1);
  reduce_scal_f32(stream, scale, M, 1, B, lda, D);
}

void internal::Cholesky::gemv_scal_cf64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B, double* D) {
  std::complex<double> minus_one(-1., 0.), one(1., 0.);
  if (1 <= N && 2 <= M)
    cublasZgemv(handle, CUBLAS_OP_C, N, M - 1, (cuDoubleComplex*)&minus_one, (const cuDoubleComplex*)&A[lda], lda, (const cuDoubleComplex*)A, 1, (cuDoubleComplex*)&one, (cuDoubleComplex*)&B[1], 1);
  reduce_scal_cf64(stream, scale, M, 1, B, lda, D);
}

void internal::Cholesky::gemv_scal_cf32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B, float* D) {
  std::complex<float> minus_one(-1., 0.), one(1.f, 0.);
  if (1 <= N && 2 <= M)
    cublasCgemv(handle, CUBLAS_OP_C, N, M - 1, (cuComplex*)&minus_one, (const cuComplex*)&A[lda], lda, (const cuComplex*)A, 1, (cuComplex*)&one, (cuComplex*)&B[1], 1);
  reduce_scal_cf32(stream, scale, M, 1, B, lda, D);
}
