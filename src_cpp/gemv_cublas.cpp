
#include <internal.hpp>
#include <cuComplex.h>

void internal::Cholesky::gemv_scal(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D) {
  if (2 <= M) {
    double one = 1., minus_one = -1.;
    if (1 <= N)
      cublasDgemv(handle, CUBLAS_OP_T, N, M, &minus_one, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &one, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(double), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_scal(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D) {
  if (2 <= M) {
    float one = 1.f, minus_one = -1.f;
    if (1 <= N)
      cublasSgemv(handle, CUBLAS_OP_T, N, M, &minus_one, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &one, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M)
    cudaMemcpyAsync(&A[N], scale, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void internal::Cholesky::gemv_scal(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* D) {
  if (2 <= M) {
    cuDoubleComplex one = make_cuDoubleComplex(1., 0.), minus_one = make_cuDoubleComplex(-1., 0.);
    if (1 <= N)
      cublasZgemv(handle, CUBLAS_OP_C, N, M, &minus_one, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &one, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M) {
    cudaMemsetAsync(&((double*)scale)[1], 0, sizeof(double), stream);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(double), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::gemv_scal(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* D) {
  if (2 <= M) {
    cuComplex one = make_cuComplex(1.f, 0.f), minus_one = make_cuComplex(-1.f, 0.f);
    if (1 <= N)
      cublasCgemv(handle, CUBLAS_OP_C, N, M, &minus_one, A, lda, &A[int64_t(j) * int64_t(lda)], 1, &one, &A[int64_t(N) + int64_t(j) * int64_t(lda)], 1);
    gemv_pp(stream, scale, j, N, M, A, lda, jpiv, D);
  }
  else if (1 == M) {
    cudaMemsetAsync(&((float*)scale)[1], 0, sizeof(float), stream);
    cudaMemcpyAsync(&A[N], scale, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
  }
}
