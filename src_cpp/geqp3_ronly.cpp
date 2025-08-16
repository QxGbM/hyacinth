
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

int32_t device::QR::dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, double* A, int32_t lda, int32_t* jpiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA;
  int32_t elem_bytes = params.elem_bytes;
  int8_t* matA = (int8_t*)(workspace), *iA = &matA[uint64_t(algnN) * uint64_t(N + 1) * uint64_t(elem_bytes)];
  int32_t* scratch = (int32_t*)(&matA[params.scratchpad]);
  int32_t* vexp = (int32_t*)(&matA[params.v_exp]);
  int32_t ret = 0;

  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  internal::int8::vexp_f64(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_f64(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  if (elem_bytes == 4) {
    float* mat_f32 = (float*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f32, scratch);
    internal::int8::scal_exponent_f32(stream, N, mat_f32, algnN, orderA, vexp);

    ret = device::Cholesky::rpotrfp_f32(stream, N, mat_f32, algnN, jpiv);
    device::copy_upper_triangular(stream, 1, N, mat_f32, algnN, Precision::FP32, A, lda, Precision::FP64);
  }
  else if (elem_bytes == 8) {
    double* mat_f64 = (double*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f64, scratch);
    internal::int8::scal_exponent_f64(stream, N, mat_f64, algnN, orderA, vexp);

    ret = device::Cholesky::rpotrfp_f64(stream, N, mat_f64, algnN, jpiv);
    device::copy_upper_triangular(stream, 1, N, mat_f64, algnN, Precision::FP64, A, lda, Precision::FP64);
  }
  else if (elem_bytes == 16 && params.use_fp64_over_32 == 1) {
    double2* mat_dd = (double2*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f128_dd(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_dd, scratch);
    internal::int8::scal_exponent_f128_dd(stream, N, mat_dd, algnN, orderA, vexp);

    ret = device::Cholesky::rpotrfp_f128_dd(stream, N, mat_dd, algnN, jpiv);
    device::copy_upper_triangular(stream, 1, N, mat_dd, algnN, Precision::FP128_DD, A, lda, Precision::FP64);
  }
  else {
    float4* mat_qf = (float4*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f128_qf(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_qf, scratch);
    internal::int8::scal_exponent_f128_qf(stream, N, mat_qf, algnN, orderA, vexp);

    ret = device::Cholesky::rpotrfp_f128_qf(stream, N, mat_qf, algnN, jpiv);
    device::copy_upper_triangular(stream, 1, N, mat_qf, algnN, Precision::FP128_QF, A, lda, Precision::FP64);
  }

  return ret;
}

int32_t device::QR::sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, float* A, int32_t lda, int32_t* jpiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA;
  int32_t elem_bytes = params.elem_bytes;
  int8_t* matA = (int8_t*)(workspace), *iA = &matA[uint64_t(algnN) * uint64_t(N + 1) * uint64_t(elem_bytes)];
  int32_t* scratch = (int32_t*)(&matA[params.scratchpad]);
  int32_t* vexp = (int32_t*)(&matA[params.v_exp]);
  int32_t ret = 0;

  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  internal::int8::vexp_f32(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_f32(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  if (elem_bytes == 4) {
    float* mat_f32 = (float*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f32, scratch);
    internal::int8::scal_exponent_f32(stream, N, mat_f32, algnN, orderA, vexp);

    ret = device::Cholesky::rpotrfp_f32(stream, N, mat_f32, algnN, jpiv);
    device::copy_upper_triangular(stream, 1, N, mat_f32, algnN, Precision::FP32, A, lda, Precision::FP32);
  }
  else {
    double* mat_f64 = (double*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f64, scratch);
    internal::int8::scal_exponent_f64(stream, N, mat_f64, algnN, orderA, vexp);

    ret = device::Cholesky::rpotrfp_f64(stream, N, mat_f64, algnN, jpiv);
    device::copy_upper_triangular(stream, 1, N, mat_f64, algnN, Precision::FP64, A, lda, Precision::FP32);
  }

  return ret;
}

int32_t device::QR::zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<double>* A, int32_t lda, int32_t* jpiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA;
  int32_t elem_bytes = params.elem_bytes;
  int8_t* matA = (int8_t*)(workspace), *iA = &matA[uint64_t(algnN) * uint64_t(N + 1) * uint64_t(elem_bytes)];
  int32_t* scratch = (int32_t*)(&matA[params.scratchpad]);
  int32_t* vexp = (int32_t*)(&matA[params.v_exp]);
  int32_t ret = 0;

  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  internal::int8::vexp_cf64(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_cf64(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  const int8_t* A_im = &iA[uint64_t(orderA) * uint64_t(algnM) * uint64_t(N)];

  if (elem_bytes == 8) {
    float* mat_f32 = (float*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f32, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, A_im, A_im, orderA, mat_f32, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, A_im, orderA, &mat_f32[strideC], scratch);

    internal::int8::planar_to_interleave_f32(stream, N, mat_f32, algnN, orderA, vexp, (std::complex<float>*)iA, algnN);
    ret = device::Cholesky::cpotrfp_f32(stream, N, (std::complex<float>*)iA, algnN, jpiv);
    device::copy_upper_triangular(stream, 2, N, (float*)iA, algnN * 2, Precision::FP32, (double*)A, lda * 2, Precision::FP64);
  }
  else if (elem_bytes == 16) {
    double* mat_f64 = (double*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f64, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, A_im, A_im, orderA, mat_f64, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, A_im, orderA, &mat_f64[strideC], scratch);

    internal::int8::planar_to_interleave_f64(stream, N, mat_f64, algnN, orderA, vexp, (std::complex<double>*)iA, algnN);
    ret = device::Cholesky::cpotrfp_f64(stream, N, (std::complex<double>*)iA, algnN, jpiv);
    device::copy_upper_triangular(stream, 2, N, (double*)iA, algnN * 2, Precision::FP64, (double*)A, lda * 2, Precision::FP64);
  }
  else if (elem_bytes == 32 && params.use_fp64_over_32 == 1) {
    double2* mat_dd = (double2*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f128_dd(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_dd, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f128_dd(stream, handle, N, params.iter_k, algnN, algnM, A_im, A_im, orderA, mat_dd, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f128_dd(stream, handle, N, params.iter_k, algnN, algnM, iA, A_im, orderA, &mat_dd[strideC], scratch);

    internal::int8::planar_to_interleave_f128_dd(stream, N, mat_dd, algnN, orderA, vexp, (complex_double2*)iA, algnN);
    ret = device::Cholesky::cpotrfp_f128_dd(stream, N, (complex_double2*)iA, algnN, jpiv);
    device::copy_upper_triangular(stream, 2, N, (double2*)iA, algnN * 2, Precision::FP128_DD, (double*)A, lda * 2, Precision::FP64);
  }
  else {
    float4* mat_qf = (float4*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f128_qf(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_qf, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f128_qf(stream, handle, N, params.iter_k, algnN, algnM, A_im, A_im, orderA, mat_qf, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f128_qf(stream, handle, N, params.iter_k, algnN, algnM, iA, A_im, orderA, &mat_qf[strideC], scratch);

    internal::int8::planar_to_interleave_f128_qf(stream, N, mat_qf, algnN, orderA, vexp, (complex_float4*)iA, algnN);
    ret = device::Cholesky::cpotrfp_f128_qf(stream, N, (complex_float4*)iA, algnN, jpiv);
    device::copy_upper_triangular(stream, 2, N, (float4*)iA, algnN * 2, Precision::FP128_QF, (double*)A, lda * 2, Precision::FP64);
  }

  return ret;
}

int32_t device::QR::cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<float>* A, int32_t lda, int32_t* jpiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA;
  int32_t elem_bytes = params.elem_bytes;
  int8_t* matA = (int8_t*)(workspace), *iA = &matA[uint64_t(algnN) * uint64_t(N + 1) * uint64_t(elem_bytes)];
  int32_t* scratch = (int32_t*)(&matA[params.scratchpad]);
  int32_t* vexp = (int32_t*)(&matA[params.v_exp]);
  int32_t ret = 0;

  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  internal::int8::vexp_cf32(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_cf32(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  const int8_t* A_im = &iA[uint64_t(orderA) * uint64_t(algnM) * uint64_t(N)];

  if (elem_bytes == 8) {
    float* mat_f32 = (float*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f32, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, A_im, A_im, orderA, mat_f32, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, A_im, orderA, &mat_f32[strideC], scratch);

    internal::int8::planar_to_interleave_f32(stream, N, mat_f32, algnN, orderA, vexp, (std::complex<float>*)iA, algnN);
    ret = device::Cholesky::cpotrfp_f32(stream, N, (std::complex<float>*)iA, algnN, jpiv);
    device::copy_upper_triangular(stream, 2, N, (float*)iA, algnN * 2, Precision::FP32, (float*)A, lda * 2, Precision::FP32);
  }
  else if (elem_bytes == 16) {
    double* mat_f64 = (double*)matA;
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, iA, orderA, mat_f64, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, A_im, A_im, orderA, mat_f64, scratch);
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, A_im, orderA, &mat_f64[strideC], scratch);

    internal::int8::planar_to_interleave_f64(stream, N, mat_f64, algnN, orderA, vexp, (std::complex<double>*)iA, algnN);
    ret = device::Cholesky::cpotrfp_f64(stream, N, (std::complex<double>*)iA, algnN, jpiv);
    device::copy_upper_triangular(stream, 2, N, (double*)iA, algnN * 2, Precision::FP64, (float*)A, lda * 2, Precision::FP32);
  }

  return ret;
}
