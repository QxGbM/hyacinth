
#include <hyacinth.hpp>
#include <internal.hpp>
#include <cuComplex.h>
#include <double_double.hpp>
#include <quad_float.hpp>

void device::cublas_preload_real(cublasHandle_t handle) {
  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);

  double* A = nullptr;
  constexpr int32_t N = 2048;
  cudaMalloc((void**)&A, N * N * sizeof(double) * 2);

  double2 scale[2]{};
  device::convert_and_copy(stream, N, N, A, N, device::Precision::FP32, 0, A, N, device::Precision::FP32);
  internal::Cholesky::imax_f32(stream, N, (float*)A, (int32_t*)A, (float*)A);
  internal::Cholesky::gemv_scal_f128_dd(stream, scale, 1, 2, 2, &((double2*)A)[8], 4, (double2*)A);
  internal::Cholesky::gemv_scal_f128_dd(stream, scale, 1, 2, 0, &((double2*)A)[8], 4, (double2*)A);

  internal::int8::vexp_f32(stream, 1, N, N, (float*)A, N, (int32_t*)A);
  internal::int8::encode_f32(stream, 1, N, N, (float*)A, N, (int32_t*)A, (int8_t*)A, N);
  internal::int8::decode_f32_strided_i32(stream, 0, 1, N, (float*)A, (int32_t*)A, N);
  internal::int8::i32_normalization(stream, N, 1, 0, (int32_t*)A);
  internal::int8::scal_exponent_f32(stream, N, (float*)A, N, 0, (int32_t*)A);

  int32_t onei = 1;
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &onei, 
    A, CUDA_R_8I, N, A, CUDA_R_8I, N, &onei, A, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  double oned = 1.;
  cublasDgemv(handle, CUBLAS_OP_T, N, N, &oned, A, N, A, 1, &oned, A, 1);
  cublasDgemv(handle, CUBLAS_OP_T, N, 1, &oned, A, N, A, 1, &oned, A, 1);
  cublasDgemv(handle, CUBLAS_OP_T, 1, N, &oned, A, N, A, 1, &oned, A, 1);

  float onef = 1.f;
  cublasSgemv(handle, CUBLAS_OP_T, N, N, &onef, (float*)A, N, (float*)A, 1, &onef, (float*)A, 1);
  cublasSgemv(handle, CUBLAS_OP_T, N, 1, &onef, (float*)A, N, (float*)A, 1, &onef, (float*)A, 1);
  cublasSgemv(handle, CUBLAS_OP_T, 1, N, &onef, (float*)A, N, (float*)A, 1, &onef, (float*)A, 1);

  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, &oned, A, N, A, N);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, &onef, (float*)A, N, (float*)A, N);

  cudaStreamSynchronize(stream);
  cudaFree(A);
}

void device::cublas_preload_complex(cublasHandle_t handle) {
  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);

  cuDoubleComplex* A = nullptr;
  constexpr int32_t N = 2048;
  cudaMalloc((void**)&A, N * N * sizeof(cuDoubleComplex));

  double2 scale[2]{};
  device::convert_and_copy(stream, N, N, A, N, device::Precision::FP32, 0, A, N, device::Precision::FP32);
  internal::Cholesky::imax_f32(stream, N, (float*)A, (int32_t*)A, (float*)A);
  internal::Cholesky::gemv_scal_f128_dd(stream, scale, 1, 2, 2, &((double2*)A)[8], 4, (double2*)A);
  internal::Cholesky::gemv_scal_f128_dd(stream, scale, 1, 2, 0, &((double2*)A)[8], 4, (double2*)A);

  internal::int8::vexp_f32(stream, 1, N, N, (float*)A, N, (int32_t*)A);
  internal::int8::encode_f32(stream, 1, N, N, (float*)A, N, (int32_t*)A, (int8_t*)A, N);
  internal::int8::decode_f32_strided_i32(stream, 0, 1, N, (float*)A, (int32_t*)A, N);
  internal::int8::i32_normalization(stream, N, 1, 0, (int32_t*)A);
  internal::int8::planar_to_interleave_f32(stream, N, (float*)A, N, 0, (int32_t*)A, (std::complex<float>*)A, N);

  int32_t onei = 1;
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &onei, 
    A, CUDA_R_8I, N, A, CUDA_R_8I, N, &onei, A, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  std::complex<double> oned(1., 0.);
  cublasZgemv(handle, CUBLAS_OP_C, N, N, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, N, (cuDoubleComplex*)A, 1, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, 1);
  cublasZgemv(handle, CUBLAS_OP_C, N, 1, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, N, (cuDoubleComplex*)A, 1, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, 1);
  cublasZgemv(handle, CUBLAS_OP_C, 1, N, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, N, (cuDoubleComplex*)A, 1, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, 1);

  std::complex<float> onef(1.f, 0.f);
  cublasCgemv(handle, CUBLAS_OP_C, N, N, (cuComplex*)&onef, (cuComplex*)A, N, (cuComplex*)A, 1, (cuComplex*)&onef, (cuComplex*)A, 1);
  cublasCgemv(handle, CUBLAS_OP_C, N, 1, (cuComplex*)&onef, (cuComplex*)A, N, (cuComplex*)A, 1, (cuComplex*)&onef, (cuComplex*)A, 1);
  cublasCgemv(handle, CUBLAS_OP_C, 1, N, (cuComplex*)&onef, (cuComplex*)A, N, (cuComplex*)A, 1, (cuComplex*)&onef, (cuComplex*)A, 1);

  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, (cuDoubleComplex*)&oned, (cuDoubleComplex*)A, N, (cuDoubleComplex*)A, N);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, (cuComplex*)&onef, (cuComplex*)A, N, (cuComplex*)A, N);

  cudaStreamSynchronize(stream);
  cudaFree(A);
}
