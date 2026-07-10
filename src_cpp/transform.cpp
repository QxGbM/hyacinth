
#include <hyacin.h>
#include <internal.hpp>
#include <cuComplex.h>
#include <stdexcept>

inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }

template <class matrix_t>
inline void ax_transform(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, matrix_t* A, int32_t lda, const matrix_t* X, int32_t ldx) {
  const int32_t rows = 16384;
  matrix_t* dev_work = nullptr;
  uint64_t dev_work_bytes = uint64_t(K) * uint64_t(rows) * uint64_t(sizeof(matrix_t));
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes, stream))
    throw std::runtime_error("Workspace allocation fail at transforming A.");
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
    nn_gemm(handle, m, K, N, &A[i], lda, X, ldx, dev_work, ld);
    internal::Cholesky::scatter_matcopy(stream, 'A', m, K, nullptr, dev_work, ld, &A[i], lda);
  }
  cudaFreeAsync(dev_work, stream);
}

inline void ax_transform_f16(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, __half* A, int32_t lda, const float* X, int32_t ldx) {
  const int32_t rows = 16384;
  uint8_t* dev_work = nullptr;
  uint64_t dev_work_bytes = uint64_t(K) * uint64_t(rows) * uint64_t(sizeof(float));
  uint64_t dev_matrix_bytes = uint64_t(K) * uint64_t(N) * uint64_t(sizeof(__half));
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes + dev_matrix_bytes, stream))
    throw std::runtime_error("Workspace allocation fail at transforming A.");

  float one = 1.f, zero = 0.f;
  __half* f16x = (__half*)&dev_work[dev_work_bytes];
  internal::Cholesky::scatter_matcopy(stream, 'A', K, N, nullptr, X, ldx, f16x, K);
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, K, N, &one, &A[i], CUDA_R_16F, lda, f16x, CUDA_R_16F, K, &zero, dev_work, CUDA_R_32F, ld, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    internal::Cholesky::scatter_matcopy(stream, 'A', m, K, nullptr, (__half*)dev_work, ld, &A[i], lda);
  }
  cudaFreeAsync(dev_work, stream);
}

inline void ax_transform_cf16(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, __half2* A, int32_t lda, const cuComplex* X, int32_t ldx) {
  const int32_t rows = 16384;
  uint8_t* dev_work = nullptr;
  uint64_t dev_work_bytes = uint64_t(K) * uint64_t(rows) * uint64_t(sizeof(cuComplex));
  uint64_t dev_matrix_bytes = uint64_t(rows) * uint64_t(N) * uint64_t(sizeof(cuComplex));
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes + dev_matrix_bytes, stream))
    throw std::runtime_error("Workspace allocation fail at transforming A.");

  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f), *f32a = (cuComplex*)&dev_work[dev_work_bytes];
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
    internal::Cholesky::scatter_matcopy(stream, 'A', m, N, nullptr, &A[i], lda, f32a, ld);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, K, N, &one, f32a, CUDA_C_32F, ld, X, CUDA_C_32F, ldx, &zero, dev_work, CUDA_C_32F, ld, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    internal::Cholesky::scatter_matcopy(stream, 'A', m, K, nullptr, (cuComplex*)dev_work, ld, &A[i], lda);
  }
  cudaFreeAsync(dev_work, stream);
}

extern "C" void hyacinXtransform(hyacinHandle_t handle, int32_t M, int32_t N, int32_t K, hyacinPrecision_t Atype, void* A, int32_t lda, const void* X, int32_t ldx) {
  if ((M <= 0) || (K <= 0)) return;
  Timer::register_kernel(handle.cudaStream, handle.timer);
  if (N <= 0) switch(Atype) {
    case HYACIN_F64: internal::Cholesky::scatter_matcopy(handle.cudaStream, 'A', M, K, nullptr, (const double*)X, ldx, (double*)A, lda); return;
    case HYACIN_F32: internal::Cholesky::scatter_matcopy(handle.cudaStream, 'A', M, K, nullptr, (const float*)X, ldx, (float*)A, lda); return;
    case HYACIN_F16: internal::Cholesky::scatter_matcopy(handle.cudaStream, 'A', M, K, nullptr, (const __half*)X, ldx, (__half*)A, lda); return;
    case HYACIN_F64_COMPLEX: internal::Cholesky::scatter_matcopy(handle.cudaStream, 'A', M, K, nullptr, (const cuDoubleComplex*)X, ldx, (cuDoubleComplex*)A, lda); return;
    case HYACIN_F32_COMPLEX: internal::Cholesky::scatter_matcopy(handle.cudaStream, 'A', M, K, nullptr, (const cuComplex*)X, ldx, (cuComplex*)A, lda); return;
    case HYACIN_F16_COMPLEX: internal::Cholesky::scatter_matcopy(handle.cudaStream, 'A', M, K, nullptr, (const __half2*)X, ldx, (__half2*)A, lda); return;
    default: return;
  }
  else switch(Atype) {
    case HYACIN_F64: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (double*)A, lda, (const double*)X, ldx); return;
    case HYACIN_F32: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (float*)A, lda, (const float*)X, ldx); return;
    case HYACIN_F16: ax_transform_f16(handle.cudaStream, handle.cublasHandle, M, N, K, (__half*)A, lda, (const float*)X, ldx); return;
    case HYACIN_F64_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (cuDoubleComplex*)A, lda, (const cuDoubleComplex*)X, ldx); return;
    case HYACIN_F32_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (cuComplex*)A, lda, (const cuComplex*)X, ldx); return;
    case HYACIN_F16_COMPLEX: ax_transform_cf16(handle.cudaStream, handle.cublasHandle, M, N, K, (__half2*)A, lda, (const cuComplex*)X, ldx); return;
    default: return;
  }
}
