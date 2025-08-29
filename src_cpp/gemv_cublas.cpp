
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>

void internal::Cholesky::gemv_cublas_f64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t M, int32_t N, double* A, int32_t lda, double* D) {
  double* B = &A[N];
  if (1 <= N && 2 <= M) {
    double minus_one = -1., one = 1.;
    cublasDgemv(handle, CUBLAS_OP_T, N, M - 1, &minus_one, &A[lda], lda, A, 1, &one, &B[1], 1);
  }
  reduce_scal_f64(stream, scale, M, 1, &B[1], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_cublas_f32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t M, int32_t N, float* A, int32_t lda, float* D) {
  float* B = &A[N];
  if (1 <= N && 2 <= M) {
    float minus_one = -1.f, one = 1.f;
    cublasSgemv(handle, CUBLAS_OP_T, N, M - 1, &minus_one, &A[lda], lda, A, 1, &one, &B[1], 1);
  }
  reduce_scal_f32(stream, scale, M, 1, &B[1], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_cublas_cf64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, double* D) {
  std::complex<double>* B = &A[N];
  if (1 <= N && 2 <= M) {
    std::complex<double> minus_one(-1., 0.), one(1., 0.);
    cublasZgemv(handle, CUBLAS_OP_C, N, M - 1, (cuDoubleComplex*)&minus_one, (const cuDoubleComplex*)&A[lda], lda, (const cuDoubleComplex*)A, 1, (cuDoubleComplex*)&one, (cuDoubleComplex*)&B[1], 1);
  }
  reduce_scal_cf64(stream, scale, M, 1, &B[1], lda, B, lda, &D[1]);
}

void internal::Cholesky::gemv_cublas_cf32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, float* D) {
  std::complex<float>* B = &A[N];
  if (1 <= N && 2 <= M) {
    std::complex<float> minus_one(-1., 0.), one(1.f, 0.);
    cublasCgemv(handle, CUBLAS_OP_C, N, M - 1, (cuComplex*)&minus_one, (const cuComplex*)&A[lda], lda, (const cuComplex*)A, 1, (cuComplex*)&one, (cuComplex*)&B[1], 1);
  }
  reduce_scal_cf32(stream, scale, M, 1, &B[1], lda, B, lda, &D[1]);
}
