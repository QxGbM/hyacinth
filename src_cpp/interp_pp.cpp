
#include <hyacin.hpp>
#include <internal.hpp>
#include <cuComplex.h>
#include <double_double.hpp>
#include <quad_float.hpp>

template <device::Precision precR, class matrix_t>
inline void cublas_trsm_real(cublasHandle_t handle, int32_t M, int32_t N, matrix_t* R, int32_t ldr) {
  if (M < N) {
    if constexpr(precR == device::Precision::FP64) {
      double one = 1.;
      cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, R, ldr, &R[M * ldr], ldr);
    }
    else if constexpr(precR == device::Precision::FP32) {
      float one = 1.f;
      cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, R, ldr, &R[M * ldr], ldr);
    }
  }
}

template <device::Precision prec, class matrix_t>
inline void interp_pp_uni_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, matrix_t* R, int32_t ldr, const int32_t* ipiv, matrix_t* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&R[int64_t(ldr) * int64_t(N)];
    cublas_trsm_real<prec>(handle, M, N, R, ldr);
    device::Utils::strided_identity(stream, M, M, R, ldr, prec);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::Utils::copy_scatter(stream, M, N, work, R, ldr, X, ldx, prec);
  }
}

template <device::Precision precR, device::Precision precX, class typeR, class typeX>
inline void interp_pp_loR_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeR* R, int32_t ldr, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&R[int64_t(ldr) * int64_t(N)];
    cublas_trsm_real<precR>(handle, M, N, R, ldr);
    device::Utils::strided_identity(stream, M, M, R, ldr, precR);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::Utils::copy_scatter(stream, M, N, work, R, ldr, &work[ldr], ldr, precR);
    device::Utils::convert_and_copy(stream, M, N, &work[ldr], ldr, precR, X, ldx, precX);
  }
}

template <device::Precision precR, device::Precision precX, class typeR, class typeX>
inline void interp_pp_loX_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeR* R, int32_t ldr, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&R[int64_t(ldr) * int64_t(N)];
    device::Utils::convert_and_copy(stream, M, N, R, ldr, precR, &work[ldr], ldr, precX);
    cublas_trsm_real<precX>(handle, M, N, (typeX*)&work[ldr], ldr);
    device::Utils::strided_identity(stream, M, M, &work[ldr], ldr, precX);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::Utils::copy_scatter(stream, M, N, work, &work[ldr], ldr, X, ldx, precX);
  }
}

void internal::InterpolativeDecomposition::interp_pp_f64_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double* R, int32_t ldr, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_uni_real<device::Precision::FP64>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f32_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float* R, int32_t ldr, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_loR_real<device::Precision::FP32, device::Precision::FP64>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_dd_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double2* R, int32_t ldr, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_DD, device::Precision::FP64>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_qf_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float4* R, int32_t ldr, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_QF, device::Precision::FP64>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f64_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double* R, int32_t ldr, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP64, device::Precision::FP32>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f32_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float* R, int32_t ldr, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_uni_real<device::Precision::FP32>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_dd_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double2* R, int32_t ldr, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_DD, device::Precision::FP32>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_qf_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float4* R, int32_t ldr, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_QF, device::Precision::FP32>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

template <device::Precision precR, class matrix_t>
inline void cublas_trsm_complex(cublasHandle_t handle, int32_t M, int32_t N, matrix_t* R, int32_t ldr) {
  if (M < N) {
    if constexpr(precR == device::Precision::FP64_COMPLEX) {
      std::complex<double> one(1., 0.);
      cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuDoubleComplex*)&one, (cuDoubleComplex*)R, ldr, (cuDoubleComplex*)&R[M * ldr], ldr);
    }
    else if constexpr(precR == device::Precision::FP32_COMPLEX) {
      std::complex<float> one(1.f, 0.f);
      cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuComplex*)&one, (cuComplex*)R, ldr, (cuComplex*)&R[M * ldr], ldr);
    }
  }
}

template <device::Precision prec, class matrix_t>
inline void interp_pp_uni_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, matrix_t* R, int32_t ldr, const int32_t* ipiv, matrix_t* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&R[int64_t(ldr) * int64_t(N)];
    cublas_trsm_complex<prec>(handle, M, N, R, ldr);
    device::Utils::strided_identity(stream, M, M, R, ldr, prec);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::Utils::copy_scatter(stream, M, N, work, R, ldr, X, ldx, prec);
  }
}

template <device::Precision precR, device::Precision precX, class typeR, class typeX>
inline void interp_pp_loR_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeR* R, int32_t ldr, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&R[int64_t(ldr) * int64_t(N)];
    cublas_trsm_complex<precR>(handle, M, N, R, ldr);
    device::Utils::strided_identity(stream, M, M, R, ldr, precR);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::Utils::copy_scatter(stream, M, N, work, R, ldr, &work[ldr], ldr, precR);
    device::Utils::convert_and_copy(stream, M, N, &work[ldr], ldr, precR, X, ldx, precX);
  }
}

template <device::Precision precR, device::Precision precX, class typeR, class typeX>
inline void interp_pp_loX_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeR* R, int32_t ldr, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&R[int64_t(ldr) * int64_t(N)];
    device::Utils::convert_and_copy(stream, M, N, R, ldr, precR, &work[ldr], ldr, precX);
    cublas_trsm_complex<precX>(handle, M, N, (typeX*)&work[ldr], ldr);
    device::Utils::strided_identity(stream, M, M, &work[ldr], ldr, precX);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::Utils::copy_scatter(stream, M, N, work, &work[ldr], ldr, X, ldx, precX);
  }
}

void internal::InterpolativeDecomposition::interp_pp_cf64_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<double>* R, int32_t ldr, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_uni_complex<device::Precision::FP64_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf32_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<float>* R, int32_t ldr, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_loR_complex<device::Precision::FP32_COMPLEX, device::Precision::FP64_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_double2* R, int32_t ldr, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_DD_COMPLEX, device::Precision::FP64_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_float4* R, int32_t ldr, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_QF_COMPLEX, device::Precision::FP64_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf64_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<double>* R, int32_t ldr, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP64_COMPLEX, device::Precision::FP32_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf32_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<float>* R, int32_t ldr, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_uni_complex<device::Precision::FP32_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_double2* R, int32_t ldr, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_DD_COMPLEX, device::Precision::FP32_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_float4* R, int32_t ldr, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_QF_COMPLEX, device::Precision::FP32_COMPLEX>(stream, handle, M, N, R, ldr, ipiv, X, ldx);
}
