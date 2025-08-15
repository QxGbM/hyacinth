
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

int32_t device::QR::dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, double* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA;
  int32_t elem_bytes = params.elem_bytes;
  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = (int32_t*)(&iA[params.work_bytes - 4 * uint64_t(algnN)]);
  int32_t ret = 0;

  uint64_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_f64(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_f64(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  if (elem_bytes == 4) {
    float* mat_f32 = (float*)mat;
    internal::int8::r8i_TN_gemm_stridedA_f32(stream, handle, N, params.iter_k, algnN, algnM, iA, orderA, mat_f32, AHA);
    internal::int8::scal_exponent_f32(stream, N, mat_f32, algnN, orderA, vexp);

    ret = device::Cholesky::spotrfp(stream, N, mat_f32, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f64(stream, 1, N, mat_f32, algnN, A, lda);
  }
  else if (elem_bytes == 8) {
    double* mat_f64 = (double*)mat;
    internal::int8::r8i_TN_gemm_stridedA_f64(stream, handle, N, params.iter_k, algnN, algnM, iA, orderA, mat_f64, AHA);
    internal::int8::scal_exponent_f64(stream, N, mat_f64, algnN, orderA, vexp);

    ret = device::Cholesky::dpotrfp(stream, N, mat_f64, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f64(stream, 1, N, mat_f64, algnN, A, lda);
  }
  else if (elem_bytes == 16 && params.use_fp64_over_32 == 1) {
    double2* mat_dd = (double2*)mat;
    internal::int8::r8i_TN_gemm_stridedA_f128_dd(stream, handle, N, params.iter_k, algnN, algnM, iA, orderA, mat_dd, AHA);
    internal::int8::scal_exponent_f128_dd(stream, N, mat_dd, algnN, orderA, vexp);

    ret = device::Cholesky::double_double_potrfp(stream, N, mat_dd, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_dd_f64(stream, 1, N, mat_dd, algnN, A, lda);
  }
  else {
    float4* mat_qf = (float4*)mat;
    internal::int8::r8i_TN_gemm_stridedA_f128_qf(stream, handle, N, params.iter_k, algnN, algnM, iA, orderA, mat_qf, AHA);
    internal::int8::scal_exponent_f128_qf(stream, N, mat_qf, algnN, orderA, vexp);

    ret = device::Cholesky::quad_float_potrfp(stream, N, mat_qf, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_qf_f64(stream, 1, N, mat_qf, algnN, A, lda);
  }

  return ret;
}

int32_t device::QR::sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, float* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = orderC / 2, order_lo = order_hi - orderC;
  int32_t gemm_expon = 2 * orderA - order_hi;

  int32_t elem_bytes = params.elem_bytes;
  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = (int32_t*)(&iA[params.work_bytes - 4 * uint64_t(algnN)]);
  int32_t ret = 0;

  uint64_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_f32(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_f32(stream, orderA, M, N, A, lda, vexp, iA, algnM);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, iA, iA, orderA, AHA, orderC);

  if (elem_bytes == 4) {
    float* mat_f32 = (float*)mat;
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, mat_f32, AHA, algnN);
    internal::int8::scal_exponent_f32(stream, N, mat_f32, algnN, gemm_expon, vexp);

    ret = device::Cholesky::spotrfp(stream, N, mat_f32, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f32(stream, 1, N, mat_f32, algnN, A, lda);
  }
  else {
    double* mat_f64 = (double*)mat;
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, mat_f64, AHA, algnN);
    internal::int8::scal_exponent_f64(stream, N, mat_f64, algnN, gemm_expon, vexp);

    ret = device::Cholesky::dpotrfp(stream, N, mat_f64, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f32(stream, 1, N, mat_f64, algnN, A, lda);
  }

  return ret;
}

int32_t device::QR::zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<double>* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = orderC / 2, order_lo = order_hi - orderC;
  int32_t gemm_expon = 2 * orderA - order_hi;

  int32_t elem_bytes = params.elem_bytes;
  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = (int32_t*)(&iA[params.work_bytes - 4 * uint64_t(algnN)]);
  int32_t ret = 0;

  uint64_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_cf64(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_cf64(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  const int8_t* A_im = &iA[uint64_t(orderA) * uint64_t(algnM) * uint64_t(N)];
  int32_t* AHA_im = &AHA[uint64_t(orderC) * strideC];

  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, iA, iA, orderA, AHA, orderC);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, A_im, A_im, orderA, AHA, orderC);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, iA, A_im, orderA, AHA_im, orderC);

  if (elem_bytes == 8) {
    float* mat_f32 = (float*)mat;
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, mat_f32, AHA, algnN);
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, &mat_f32[strideC], AHA_im, algnN);

    internal::int8::planar_to_interleave_f32(stream, N, mat_f32, algnN, gemm_expon, vexp, (std::complex<float>*)workspace, algnN);
    ret = device::Cholesky::cpotrfp(stream, N, (std::complex<float>*)workspace, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f64(stream, 2, N, (float*)workspace, algnN * 2, (double*)A, lda * 2);
  }
  else if (elem_bytes == 16) {
    double* mat_f64 = (double*)mat;
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, mat_f64, AHA, algnN);
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, &mat_f64[strideC], AHA_im, algnN);

    internal::int8::planar_to_interleave_f64(stream, N, mat_f64, algnN, gemm_expon, vexp, (std::complex<double>*)workspace, algnN);
    ret = device::Cholesky::zpotrfp(stream, N, (std::complex<double>*)workspace, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f64(stream, 2, N, (double*)workspace, algnN * 2, (double*)A, lda * 2);
  }
  else if (elem_bytes == 32 && params.use_fp64_over_32 == 1) {
    double2* mat_dd = (double2*)mat;
    internal::int8::decode_f128_dd_strided_i32(stream, order_lo, order_hi, N, mat_dd, AHA, algnN);
    internal::int8::decode_f128_dd_strided_i32(stream, order_lo, order_hi, N, &mat_dd[strideC], AHA_im, algnN);

    internal::int8::planar_to_interleave_f128_dd(stream, N, mat_dd, algnN, gemm_expon, vexp, (complex_double2*)workspace, algnN);
    ret = device::Cholesky::complex_double_double_potrfp(stream, N, (complex_double2*)workspace, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_dd_f64(stream, 2, N, (double2*)workspace, algnN * 2, (double*)A, lda * 2);
  }
  else {
    float4* mat_qf = (float4*)mat;
    internal::int8::decode_f128_qf_strided_i32(stream, order_lo, order_hi, N, mat_qf, AHA, algnN);
    internal::int8::decode_f128_qf_strided_i32(stream, order_lo, order_hi, N, &mat_qf[strideC], AHA_im, algnN);

    internal::int8::planar_to_interleave_f128_qf(stream, N, mat_qf, algnN, gemm_expon, vexp, (complex_float4*)workspace, algnN);
    ret = device::Cholesky::complex_quad_float_potrfp(stream, N, (complex_float4*)workspace, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_qf_f64(stream, 2, N, (float4*)workspace, algnN * 2, (double*)A, lda * 2);
  }

  return ret;
}

int32_t device::QR::cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<float>* A, int32_t lda, int32_t* ipiv, void* workspace) {
  int32_t M = params.M, N = params.N;
  int32_t algnM = params.algnM, algnN = params.algnN;
  int32_t orderA = params.orderA, orderC = params.orderC;
  int32_t order_hi = orderC / 2, order_lo = order_hi - orderC;
  int32_t gemm_expon = 2 * orderA - order_hi;

  int32_t elem_bytes = params.elem_bytes;
  int8_t* iA = (int8_t*)(workspace);
  int32_t* AHA = (int32_t*)(&iA[params.n_i8]);
  int32_t* vexp = (int32_t*)(&iA[params.work_bytes - 4 * uint64_t(algnN)]);
  int32_t ret = 0;

  uint64_t int_bytes = params.n_i8 + params.n_i32 * 4;
  cudaMemsetAsync(workspace, 0, params.work_bytes, stream);
  void* mat = &iA[int_bytes];
  internal::int8::vexp_cf32(stream, orderA, M, N, A, lda, vexp);
  internal::int8::encode_cf32(stream, orderA, M, N, A, lda, vexp, iA, algnM);

  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  const int8_t* A_im = &iA[uint64_t(orderA) * uint64_t(algnM) * uint64_t(N)];
  int32_t* AHA_im = &AHA[uint64_t(orderC) * strideC];

  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, iA, iA, orderA, AHA, orderC);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, A_im, A_im, orderA, AHA, orderC);
  internal::int8::r8i_TN_gemm_stridedA(stream, handle, N, 1 << params.iter_k, algnN, algnM, iA, A_im, orderA, AHA_im, orderC);

  if (elem_bytes == 8) {
    float* mat_f32 = (float*)mat;
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, mat_f32, AHA, algnN);
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, &mat_f32[strideC], AHA_im, algnN);

    internal::int8::planar_to_interleave_f32(stream, N, mat_f32, algnN, gemm_expon, vexp, (std::complex<float>*)workspace, algnN);
    ret = device::Cholesky::cpotrfp(stream, N, (std::complex<float>*)workspace, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f32_f32(stream, 2, N, (float*)workspace, algnN * 2, (float*)A, lda * 2);
  }
  else if (elem_bytes == 16) {
    double* mat_f64 = (double*)mat;
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, mat_f64, AHA, algnN);
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, &mat_f64[strideC], AHA_im, algnN);

    internal::int8::planar_to_interleave_f64(stream, N, mat_f64, algnN, gemm_expon, vexp, (std::complex<double>*)workspace, algnN);
    ret = device::Cholesky::zpotrfp(stream, N, (std::complex<double>*)workspace, algnN, ipiv);
    internal::Cholesky::copy_convert_upper_f64_f32(stream, 2, N, (double*)workspace, algnN * 2, (float*)A, lda * 2);
  }

  return ret;
}
