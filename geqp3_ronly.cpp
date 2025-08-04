
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

int32_t device::QR::dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, double* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = 2 * orderA, order_lo = order_hi - orderC;
  int32_t strideA = algnM * algnN;

  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = &AHA[algnN * algnN * orderC];
  int32_t elem_bytes = params.elem_bytes;
  int32_t ret = 0;

  size_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, int_bytes);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_f64(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_f64(stream, orderA, M, N, A, lda, vexp, iA, algnM, strideA);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, params.iter_k, algnN, algnN, algnM, iA, strideA, orderA, iA, AHA, orderC);

  if (elem_bytes == 4) {
    float* mat_f32 = (float*)mat;
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f32, algnN);
    ret = device::Cholesky::spotrfp(stream, N, mat_f32, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f64(stream, 1, N, mat_f32, algnN, A, lda);
  }
  else if (elem_bytes == 8) {
    double* mat_f64 = (double*)mat;
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f64, algnN);
    ret = device::Cholesky::dpotrfp(stream, N, mat_f64, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f64(stream, 1, N, mat_f64, algnN, A, lda);
  }
  else if (elem_bytes == 16 && params.use_fp64_over_32 == 1) {
    double2* mat_dd = (double2*)mat;
    internal::int8::decode_dd_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_dd, algnN);
    ret = device::Cholesky::double_double_potrfp(stream, N, mat_dd, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_dd_f64(stream, 1, N, mat_dd, algnN, A, lda);
  }
  else {
    float4* mat_qf = (float4*)mat;
    internal::int8::decode_qf_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_qf, algnN);
    ret = device::Cholesky::quad_float_potrfp(stream, N, mat_qf, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_qf_f64(stream, 1, N, mat_qf, algnN, A, lda);
  }

  return ret;
}

int32_t device::QR::sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, float* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = 2 * orderA, order_lo = order_hi - orderC;
  int32_t strideA = algnM * algnN;


  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = &AHA[algnN * algnN * orderC];
  int32_t elem_bytes = params.elem_bytes;
  int32_t ret = 0;

  size_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, int_bytes);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_f32(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_f32(stream, orderA, M, N, A, lda, vexp, iA, algnM, strideA);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, params.iter_k, algnN, algnN, algnM, iA, strideA, orderA, iA, AHA, orderC);

  if (elem_bytes == 4) {
    float* mat_f32 = (float*)mat;
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f32, algnN);
    ret = device::Cholesky::spotrfp(stream, N, mat_f32, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f32(stream, 1, N, mat_f32, algnN, A, lda);
  }
  else {
    double* mat_f64 = (double*)mat;
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f64, algnN);
    ret = device::Cholesky::dpotrfp(stream, N, mat_f64, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f32(stream, 1, N, mat_f64, algnN, A, lda);
  }

  return ret;
}

int32_t device::QR::zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<double>* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = 2 * orderA, order_lo = order_hi - orderC;
  int32_t strideA = algnM * algnN;

  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = &AHA[2 * algnN * algnN * orderC];
  int32_t elem_bytes = params.elem_bytes;
  int32_t ret = 0;

  size_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, int_bytes);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_cf64(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_cf64(stream, orderA, M, N, A, lda, vexp, iA, algnM, strideA);
  internal::int8::c8i_HN_gemm_stridedA(stream, handle, params.iter_k, algnN, algnN, algnM, iA, strideA, orderA, iA, AHA, orderC);

  if (elem_bytes == 8) {
    std::complex<float>* mat_f32 = (std::complex<float>*)mat;
    internal::int8::decode_cf32_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f32, algnN);
    ret = device::Cholesky::cpotrfp(stream, N, mat_f32, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f64(stream, 2, N, (float*)mat_f32, algnN * 2, (double*)A, lda * 2);
  }
  else if (elem_bytes == 16) {
    std::complex<double>* mat_f64 = (std::complex<double>*)mat;
    internal::int8::decode_cf64_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f64, algnN);
    ret = device::Cholesky::zpotrfp(stream, N, mat_f64, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f64(stream, 2, N, (double*)mat_f64, algnN * 2, (double*)A, lda * 2);
  }
  else if (elem_bytes == 32 && params.use_fp64_over_32 == 1) {
    complex_double2* mat_dd = (complex_double2*)mat;
    internal::int8::decode_complex_dd_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_dd, algnN);
    ret = device::Cholesky::complex_double_double_potrfp(stream, N, mat_dd, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_dd_f64(stream, 2, N, (double2*)mat_dd, algnN * 2, (double*)A, lda * 2);
  }
  else {
    complex_float4* mat_qf = (complex_float4*)mat;
    internal::int8::decode_complex_qf_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_qf, algnN);
    ret = device::Cholesky::complex_quad_float_potrfp(stream, N, mat_qf, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_qf_f64(stream, 2, N, (float4*)mat_qf, algnN * 2, (double*)A, lda * 2);
  }

  return ret;
}

int32_t device::QR::cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<float>* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = 2 * orderA, order_lo = order_hi - orderC;
  int32_t strideA = algnM * algnN;

  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = &AHA[2 * algnN * algnN * orderC];
  int32_t elem_bytes = params.elem_bytes;
  int32_t ret = 0;

  size_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, int_bytes);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_cf32(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_cf32(stream, orderA, M, N, A, lda, vexp, iA, algnM, strideA);
  internal::int8::c8i_HN_gemm_stridedA(stream, handle, params.iter_k, algnN, algnN, algnM, iA, strideA, orderA, iA, AHA, orderC);

  if (elem_bytes == 8) {
    std::complex<float>* mat_f32 = (std::complex<float>*)mat;
    internal::int8::decode_cf32_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f32, algnN);
    ret = device::Cholesky::cpotrfp(stream, N, mat_f32, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f32(stream, 2, N, (float*)mat_f32, algnN * 2, (float*)A, lda * 2);
  }
  else {
    std::complex<double>* mat_f64 = (std::complex<double>*)mat;
    internal::int8::decode_cf64_strided_i32(stream, order_lo, order_hi, N, vexp, AHA, algnN, mat_f64, algnN);
    ret = device::Cholesky::zpotrfp(stream, N, mat_f64, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f32(stream, 2, N, (double*)mat_f64, algnN * 2, (float*)A, lda * 2);
  }

  return ret;
}
