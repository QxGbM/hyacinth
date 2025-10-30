
#include <internal.hpp>
#include <hyacin.hpp>
#include <cuComplex.h>

inline void workspace_realloc(cudaStream_t stream, void** ptr, int64_t* bytes_old, int64_t bytes_required) {
  if (*bytes_old < bytes_required) {
    void* workspace = nullptr;
    cudaStreamSynchronize(stream);
    cudaMalloc(&workspace, bytes_required);
    if (*ptr)
      cudaFree(*ptr);
    *ptr = workspace;
    *bytes_old = bytes_required;
  }
}

constexpr int64_t ws_rows = 8192;

void internal::Orthogonalize::qr_pp_f64(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, double* A, int32_t lda, const int32_t* ipiv, double* tau, void** Workspace, int64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  double* R = &A[int64_t(K) * int64_t(lda)];
  cusolverDnDgeqrf_bufferSize(cusolverH, M, K, A, lda, &l1);
  if (K < N)
    cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, &l2);
  int64_t bytes_required = std::max(ws_rows * int64_t(N), int64_t(1 + std::max(l1, l2))) * int64_t(sizeof(double));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  double* work = (double*)*Workspace;
  device::inplace_gather(stream, M, N, (int32_t*)tau, A, lda, work, *Lwork, device::Precision::FP64);
  cusolverDnDgeqrf(cusolverH, M, K, A, lda, tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(double), stream);
  }
}

void internal::Orthogonalize::qr_pp_f32(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, float* A, int32_t lda, const int32_t* ipiv, float* tau, void** Workspace, int64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  float* R = &A[int64_t(K) * int64_t(lda)];
  cusolverDnSgeqrf_bufferSize(cusolverH, M, K, A, lda, &l1);
  if (K < N)
    cusolverDnSormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, &l2);
  int64_t bytes_required = std::max(ws_rows * int64_t(N), int64_t(1 + std::max(l1, l2))) * int64_t(sizeof(float));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  float* work = (float*)*Workspace;
  device::inplace_gather(stream, M, N, (int32_t*)tau, A, lda, work, *Lwork, device::Precision::FP32);
  cusolverDnSgeqrf(cusolverH, M, K, A, lda, tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnSormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(float), stream);
  }
}

void internal::Orthogonalize::qr_pp_cf64(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* tau, void** Workspace, int64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  cuDoubleComplex* R = (cuDoubleComplex*)&A[int64_t(K) * int64_t(lda)];
  cusolverDnZgeqrf_bufferSize(cusolverH, M, K, (cuDoubleComplex*)A, lda, &l1);
  if (K < N)
    cusolverDnZunmqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, R, lda, &l2);
  int64_t bytes_required = std::max(ws_rows * int64_t(N), int64_t(1 + std::max(l1, l2))) * int64_t(sizeof(cuDoubleComplex));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  cuDoubleComplex* work = (cuDoubleComplex*)*Workspace;
  device::inplace_gather(stream, 2 * M, N, (int32_t*)tau, A, 2 * lda, work, *Lwork, device::Precision::FP64);
  cusolverDnZgeqrf(cusolverH, M, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnZunmqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(cuDoubleComplex), stream);
  }
}

void internal::Orthogonalize::qr_pp_cf32(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* tau, void** Workspace, int64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  cuComplex* R = (cuComplex*)&A[int64_t(K) * int64_t(lda)];
  cusolverDnCgeqrf_bufferSize(cusolverH, M, K, (cuComplex*)A, lda, &l1);
  if (K < N)
    cusolverDnCunmqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, (cuComplex*)A, lda, (cuComplex*)tau, R, lda, &l2);
  int64_t bytes_required = std::max(ws_rows * int64_t(N), int64_t(1 + std::max(l1, l2))) * int64_t(sizeof(cuComplex));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  cuComplex* work = (cuComplex*)*Workspace;
  device::inplace_gather(stream, 2 * M, N, (int32_t*)tau, A, 2 * lda, work, *Lwork, device::Precision::FP32);
  cusolverDnCgeqrf(cusolverH, M, K, (cuComplex*)A, lda, (cuComplex*)tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnCunmqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, (cuComplex*)A, lda, (cuComplex*)tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(cuComplex), stream);
  }
}
