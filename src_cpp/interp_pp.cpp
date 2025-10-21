
#include <hyacin.hpp>
#include <internal.hpp>
#include <cuComplex.h>
#include <double_double.hpp>
#include <quad_float.hpp>

template <device::Precision precA, class matrix_t>
inline void cublas_trsm_real(cublasHandle_t handle, int32_t M, int32_t N, matrix_t* A, int32_t lda) {
  if (M < N) {
    if constexpr(precA == device::Precision::FP64) {
      double one = 1.;
      cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, A, lda, &A[M * lda], lda);
    }
    else if constexpr(precA == device::Precision::FP32) {
      float one = 1.f;
      cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, A, lda, &A[M * lda], lda);
    }
  }
}

template <device::Precision prec, class matrix_t>
inline void interp_pp_uni_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, matrix_t* A, int32_t lda, const int32_t* ipiv, matrix_t* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&A[int64_t(lda) * int64_t(N)];
    cublas_trsm_real<prec>(handle, M, N, A, lda);
    device::strided_identity(stream, M, M, M + 1, A, lda, prec);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::copy_scatter(stream, M, N, work, A, lda, X, ldx, prec);
  }
}

template <device::Precision precA, device::Precision precX, class typeA, class typeX>
inline void interp_pp_loA_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeA* A, int32_t lda, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&A[int64_t(lda) * int64_t(N)];
    cublas_trsm_real<precA>(handle, M, N, A, lda);
    device::strided_identity(stream, M, M, M + 1, A, lda, precA);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::copy_scatter(stream, M, N, work, A, lda, &work[lda], lda, precA);
    device::convert_and_copy(stream, M, N, &work[lda], lda, precA, 0, X, ldx, precX);
  }
}

template <device::Precision precA, device::Precision precX, class typeA, class typeX>
inline void interp_pp_loX_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeA* A, int32_t lda, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&A[int64_t(lda) * int64_t(N)];
    device::convert_and_copy(stream, M, N, A, lda, precA, 0, &work[lda], lda, precX);
    cublas_trsm_real<precX>(handle, M, N, (typeX*)&work[lda], lda);
    device::strided_identity(stream, M, M, M + 1, &work[lda], lda, precX);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::copy_scatter(stream, M, N, work, &work[lda], lda, X, ldx, precX);
  }
}

void internal::InterpolativeDecomposition::interp_pp_f64_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_uni_real<device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f32_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_loA_real<device::Precision::FP32, device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_dd_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double2* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_DD, device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_qf_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float4* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_QF, device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f64_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP64, device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f32_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_uni_real<device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_dd_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double2* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_DD, device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_f128_qf_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float4* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx) {
  interp_pp_loX_real<device::Precision::FP128_QF, device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

template <device::Precision precA, class matrix_t>
inline void cublas_trsm_complex(cublasHandle_t handle, int32_t M, int32_t N, matrix_t* A, int32_t lda) {
  if (M < N) {
    if constexpr(precA == device::Precision::FP64) {
      std::complex<double> one(1., 0.);
      cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuDoubleComplex*)&one, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)&A[M * lda], lda);
    }
    else if constexpr(precA == device::Precision::FP32) {
      std::complex<float> one(1.f, 0.f);
      cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuComplex*)&one, (cuComplex*)A, lda, (cuComplex*)&A[M * lda], lda);
    }
  }
}

template <device::Precision prec, class matrix_t>
inline void interp_pp_uni_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, matrix_t* A, int32_t lda, const int32_t* ipiv, matrix_t* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&A[int64_t(lda) * int64_t(N)];
    cublas_trsm_complex<prec>(handle, M, N, A, lda);
    device::strided_identity(stream, 2 * M, M, 2 * M + 2, A, 2 * lda, prec);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::copy_scatter(stream, 2 * M, N, work, A, 2 * lda, X, 2 * ldx, prec);
  }
}

template <device::Precision precA, device::Precision precX, class typeA, class typeX>
inline void interp_pp_loA_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeA* A, int32_t lda, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&A[int64_t(lda) * int64_t(N)];
    cublas_trsm_complex<precA>(handle, M, N, A, lda);
    device::strided_identity(stream, 2 * M, M, 2 * M + 2, A, 2 * lda, precA);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::copy_scatter(stream, 2 * M, N, work, A, 2 * lda, &work[lda], 2 * lda, precA);
    device::convert_and_copy(stream, 2 * M, N, &work[lda], 2 * lda, precA, 0, X, 2 * ldx, precX);
  }
}

template <device::Precision precA, device::Precision precX, class typeA, class typeX>
inline void interp_pp_loX_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, typeA* A, int32_t lda, const int32_t* ipiv, typeX* X, int32_t ldx) {
  if (0 < M) {
    int32_t* work = (int32_t*)&A[int64_t(lda) * int64_t(N)];
    device::convert_and_copy(stream, 2 * M, N, A, 2 * lda, precA, 0, &work[lda], 2 * lda, precX);
    cublas_trsm_complex<precX>(handle, M, N, (typeX*)&work[lda], lda);
    device::strided_identity(stream, 2 * M, M, 2 * M + 2, &work[lda], 2 * lda, precX);
    cudaMemcpyAsync(work, ipiv, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
    device::copy_scatter(stream, 2 * M, N, work, &work[lda], 2 * lda, X, 2 * ldx, precX);
  }
}

void internal::InterpolativeDecomposition::interp_pp_cf64_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_uni_complex<device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf32_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_loA_complex<device::Precision::FP32, device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_double2* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_DD, device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_float4* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_QF, device::Precision::FP64>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf64_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP64, device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf32_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_uni_complex<device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_double2* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_DD, device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}

void internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_float4* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx) {
  interp_pp_loX_complex<device::Precision::FP128_QF, device::Precision::FP32>(stream, handle, M, N, A, lda, ipiv, X, ldx);
}
