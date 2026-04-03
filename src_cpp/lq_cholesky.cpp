
#include <hyacin.h>
#include <cuComplex.h>
#include <stdexcept>

template <class T> inline cudaDataType_t cuda_type();
template <> inline cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }
template <> inline cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <> inline cudaDataType_t cuda_type<cuDoubleComplex>() { return CUDA_C_64F; }
template <> inline cudaDataType_t cuda_type<cuComplex>() { return CUDA_C_32F; }

extern "C" void hyacinXlqchol_bufferSize(cusolverDnHandle_t s_handle, cusolverDnParams_t params, int32_t K, hyacinPrecision_t AXtype, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  if (K <= 0) { *dev_work_bytes = 0; return; }
  int32_t x_bytes; cudaDataType_t type_c; hyacinXelem('A', AXtype, nullptr, &x_bytes, &type_c);
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXpotrf_bufferSize(s_handle, params, CUBLAS_FILL_MODE_LOWER, K, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  *dev_work_bytes = std::max(*dev_work_bytes, uint64_t(int64_t(K) * int64_t(K) * int64_t(x_bytes)) + uint64_t(workspaceInBytesOnDevice) + uint64_t(sizeof(int32_t)));
  *pinned_work_bytes = std::max(*pinned_work_bytes, uint64_t(workspaceInBytesOnHost));
}

inline void nh_herk(cublasHandle_t handle, int32_t K, int32_t N, const double* A, int32_t lda, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, N, &one, A, lda, &zero, C, ldc); }
inline void nh_herk(cublasHandle_t handle, int32_t K, int32_t N, const float* A, int32_t lda, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, N, &one, A, lda, &zero, C, ldc); }
inline void nh_herk(cublasHandle_t handle, int32_t K, int32_t N, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, N, &one, A, lda, &zero, C, ldc); }
inline void nh_herk(cublasHandle_t handle, int32_t K, int32_t N, const cuComplex* A, int32_t lda, cuComplex* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, K, N, &one, A, lda, &zero, C, ldc); }

inline void ln_trsm(cublasHandle_t handle, int32_t K, int32_t N, const double* A, int32_t lda, double* C, int32_t ldc)
{ double one = 1.; cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N, &one, A, lda, C, ldc); }
inline void ln_trsm(cublasHandle_t handle, int32_t K, int32_t N, const float* A, int32_t lda, float* C, int32_t ldc)
{ float one = 1.f; cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N, &one, A, lda, C, ldc); }
inline void ln_trsm(cublasHandle_t handle, int32_t K, int32_t N, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* C, int32_t ldc)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.); cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N, &one, A, lda, C, ldc); }
inline void ln_trsm(cublasHandle_t handle, int32_t K, int32_t N, const cuComplex* A, int32_t lda, cuComplex* C, int32_t ldc)
{ cuComplex one = make_cuComplex(1.f, 0.f); cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N, &one, A, lda, C, ldc); }

template <class complex_t>
inline void lq_cholesky(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, int32_t K, int32_t N, complex_t* X, int32_t ldx, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  int64_t gram_bytes = int64_t(K) * int64_t(K) * int64_t(sizeof(complex_t));
  cudaDataType_t type_c = cuda_type<complex_t>();
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXpotrf_bufferSize(s_handle, params, CUBLAS_FILL_MODE_LOWER, K, type_c, dev_work, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  if ((uint64_t(workspaceInBytesOnDevice + sizeof(int32_t)) + uint64_t(gram_bytes)) <= dev_work_bytes && uint64_t(workspaceInBytesOnHost) <= pinned_work_bytes) {
    complex_t* W = (complex_t*)dev_work, *G = (complex_t*)&((int8_t*)dev_work)[workspaceInBytesOnDevice];
    int32_t* info = (int32_t*)&((int8_t*)dev_work)[int64_t(workspaceInBytesOnDevice) + gram_bytes];
    nh_herk(handle, K, N, X, ldx, G, K);
    cusolverDnXpotrf(s_handle, params, CUBLAS_FILL_MODE_LOWER, K, type_c, G, K, type_c, W, workspaceInBytesOnDevice, pinned_work, workspaceInBytesOnHost, info);
    ln_trsm(handle, K, N, G, K, X, ldx);
  } else throw std::runtime_error("Insufficient workspace for Cholesky-LQ.");
}

extern "C" void hyacinXlqchol(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, int32_t K, int32_t N, hyacinPrecision_t AXtype, void* X, int32_t ldx, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  if (K <= 0 || N <= 0) { return; }
  switch(AXtype) {
    case HYACIN_F64:
      lq_cholesky(handle, s_handle, params, K, N, (double*)X, ldx, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes); return;
    case HYACIN_F32:
      lq_cholesky(handle, s_handle, params, K, N, (float*)X, ldx, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes); return;
    case HYACIN_F64_COMPLEX:
      lq_cholesky(handle, s_handle, params, K, N, (cuDoubleComplex*)X, ldx, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes); return;
    case HYACIN_F32_COMPLEX:
      lq_cholesky(handle, s_handle, params, K, N, (cuComplex*)X, ldx, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes); return;
    default: return;
  }
}
