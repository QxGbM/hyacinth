
#include <hyacinth.hpp>
#include <cuComplex.h>

void device::cublas_preload_real(cublasHandle_t handle) {
  double* A = nullptr;
  cudaMalloc((void**)&A, 8 * sizeof(double));
  double one = 1.;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &one, A, 2, A, 2, &one, &A[4], 2);
  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, &one, A, 2, &A[4], 2);

  float onef = 1.f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &onef, (float*)A, 2, (float*)A, 2, &onef, (float*)&A[4], 2);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, &onef, (float*)A, 2, (float*)&A[4], 2);

  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);
  cudaStreamSynchronize(stream);
  cudaFree(A);
}

void device::cublas_preload_complex(cublasHandle_t handle) {
  cuDoubleComplex* A = nullptr;
  cudaMalloc((void**)&A, 8 * sizeof(cuDoubleComplex));
  std::complex<double> one(1., 0.);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, (cuDoubleComplex*)&one, (cuDoubleComplex*)A, 2, (cuDoubleComplex*)A, 2, (cuDoubleComplex*)&one, (cuDoubleComplex*)&A[4], 2);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, (cuDoubleComplex*)&one, (cuDoubleComplex*)A, 2, (cuDoubleComplex*)&A[4], 2);

  std::complex<float> onef(1.f, 0.f);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, (cuComplex*)&onef, (cuComplex*)A, 2, (cuComplex*)A, 2, (cuComplex*)&onef, (cuComplex*)&A[4], 2);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 2, 2, (cuComplex*)&onef, (cuComplex*)A, 2, (cuComplex*)&A[4], 2);

  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);
  cudaStreamSynchronize(stream);
  cudaFree(A);
}
