
#include <hyacinth.hpp>
#include <internal.hpp>
#include <cuComplex.h>
#include <limits>

void device::check_interp_decomp_f64(cudaStream_t stream, cublasHandle_t handle, int32_t rank,
  int32_t M, int32_t N, const double* A, int32_t lda, const int32_t* jpiv, const double* X, int32_t ldx, double* rel_err) {

  if (rank <= 0)
  { *rel_err = std::numeric_limits<double>::quiet_NaN(); return; }

  double* dA = nullptr, *dC = nullptr;
  int64_t strideA = int64_t(M) * int64_t(N);
  cudaMalloc((void**)(&dA), strideA * sizeof(double));
  cudaMalloc((void**)(&dC), int64_t(M) * int64_t(rank) * sizeof(double));

  double zero = 0., one = 1., minus_one = -1.;
  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, dA, M, dA, M);
  copy_gather(stream, M, rank, jpiv, dA, M, dC, M, Precision::FP64);

  double nrm = 0., err = 0.;
  cublasDnrm2_64(handle, strideA, dA, int64_t(1), &nrm);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, &minus_one, dC, M, X, ldx, &one, dA, M);
  cublasDnrm2_64(handle, strideA, dA, int64_t(1), &err);

  cudaStreamSynchronize(stream);
  *rel_err = err / nrm;
  cudaFree(dA);
  cudaFree(dC);
}

void device::check_interp_decomp_f32(cudaStream_t stream, cublasHandle_t handle, int32_t rank,
  int32_t M, int32_t N, const float* A, int32_t lda, const int32_t* jpiv, const float* X, int32_t ldx, double* rel_err) {

  if (rank <= 0)
  { *rel_err = std::numeric_limits<double>::quiet_NaN(); return; }

  float* dA = nullptr, *dC = nullptr;
  int64_t strideA = int64_t(M) * int64_t(N);
  cudaMalloc((void**)(&dA), strideA * sizeof(float));
  cudaMalloc((void**)(&dC), int64_t(M) * int64_t(rank) * sizeof(float));

  float zero = 0.f, one = 1.f, minus_one = -1.f;
  cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, dA, M, dA, M);
  copy_gather(stream, M, rank, jpiv, dA, M, dC, M, Precision::FP32);

  float nrm = 0.f, err = 0.f;
  cublasSnrm2_64(handle, strideA, dA, int64_t(1), &nrm);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, &minus_one, dC, M, X, ldx, &one, dA, M);
  cublasSnrm2_64(handle, strideA, dA, int64_t(1), &err);

  cudaStreamSynchronize(stream);
  *rel_err = double(err) / double(nrm);
  cudaFree(dA);
  cudaFree(dC);
}

void device::check_interp_decomp_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t rank,
  int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const int32_t* jpiv, const std::complex<double>* X, int32_t ldx, double* rel_err) {

  if (rank <= 0)
  { *rel_err = std::numeric_limits<double>::quiet_NaN(); return; }

  cuDoubleComplex* dA = nullptr, *dC = nullptr;
  int64_t strideA = int64_t(M) * int64_t(N);
  cudaMalloc((void**)(&dA), strideA * sizeof(cuDoubleComplex));
  cudaMalloc((void**)(&dC), int64_t(M) * int64_t(rank) * sizeof(cuDoubleComplex));

  std::complex<double> zero(0., 0.), one(1., 0.), minus_one(-1., 0.);
  cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, (cuDoubleComplex*)&one, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)&zero, dA, M, dA, M);
  copy_gather(stream, 2 * M, rank, jpiv, dA, 2 * M, dC, 2 * M, Precision::FP64);

  double nrm = 0., err = 0.;
  cublasDznrm2_64(handle, strideA, dA, int64_t(1), &nrm);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, (cuDoubleComplex*)&minus_one, dC, M, (const cuDoubleComplex*)X, ldx, (cuDoubleComplex*)&one, dA, M);
  cublasDznrm2_64(handle, strideA, dA, int64_t(1), &err);

  cudaStreamSynchronize(stream);
  *rel_err = err / nrm;
  cudaFree(dA);
  cudaFree(dC);
}

void device::check_interp_decomp_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t rank,
  int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, const int32_t* jpiv, const std::complex<float>* X, int32_t ldx, double* rel_err) {
  
  if (rank <= 0)
  { *rel_err = std::numeric_limits<double>::quiet_NaN(); return; }

  cuComplex* dA = nullptr, *dC = nullptr;
  int64_t strideA = int64_t(M) * int64_t(N);
  cudaMalloc((void**)(&dA), strideA * sizeof(cuComplex));
  cudaMalloc((void**)(&dC), int64_t(M) * int64_t(rank) * sizeof(cuComplex));

  std::complex<float> zero(0.f, 0.f), one(1.f, 0.f), minus_one(-1.f, 0.f);
  cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, (cuComplex*)&one, (const cuComplex*)A, lda, (cuComplex*)&zero, dA, M, dA, M);
  copy_gather(stream, 2 * M, rank, jpiv, dA, 2 * M, dC, 2 * M, Precision::FP32);

  float nrm = 0., err = 0.;
  cublasScnrm2_64(handle, strideA, dA, int64_t(1), &nrm);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, rank, (cuComplex*)&minus_one, dC, M, (const cuComplex*)X, ldx, (cuComplex*)&one, dA, M);
  cublasScnrm2_64(handle, strideA, dA, int64_t(1), &err);

  cudaStreamSynchronize(stream);
  *rel_err = double(err) / double(nrm);
  cudaFree(dA);
  cudaFree(dC);
}
