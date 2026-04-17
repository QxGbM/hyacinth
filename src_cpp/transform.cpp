
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

inline void matcopy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const double* A, int32_t lda, double* B, int32_t ldb)
{ double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matcopy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const float* A, int32_t lda, float* B, int32_t ldb)
{ float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matcopy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matcopy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }

template <class complex_t>
inline void ax_transform(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, complex_t* A, int32_t lda, const complex_t* X, int32_t ldx) {
  const int32_t rows = 16384;
  complex_t* dev_work = nullptr;
  uint64_t dev_work_bytes = uint64_t(K) * uint64_t(rows) * uint64_t(sizeof(complex_t));
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes, stream))
    throw std::runtime_error("Workspace allocation fail at transforming A.");
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
    nn_gemm(handle, m, K, N, &A[i], lda, X, ldx, dev_work, ld);
    matcopy(handle, m, K, dev_work, ld, &A[i], lda);
  }
  cudaFreeAsync(dev_work, stream);
}

extern "C" void hyacinXtransform(hyacinHandle_t handle, int32_t M, int32_t N, int32_t K, hyacinPrecision_t AXtype, void* A, int32_t lda, const void* X, int32_t ldx) {
  if ((M <= 0) || (K <= 0)) return;
  Timer::register_kernel(handle.cudaStream, handle.timer);
  if (N <= 0) switch(AXtype) {
    case HYACIN_F64: matcopy(handle.cublasHandle, M, K, (const double*)X, ldx, (double*)A, lda); return;
    case HYACIN_F32: matcopy(handle.cublasHandle, M, K, (const float*)X, ldx, (float*)A, lda); return;
    case HYACIN_F64_COMPLEX: matcopy(handle.cublasHandle, M, K, (const cuDoubleComplex*)X, ldx, (cuDoubleComplex*)A, lda); return;
    case HYACIN_F32_COMPLEX: matcopy(handle.cublasHandle, M, K, (const cuComplex*)X, ldx, (cuComplex*)A, lda); return;
    default: return;
  }
  else switch(AXtype) {
    case HYACIN_F64: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (double*)A, lda, (const double*)X, ldx); return;
    case HYACIN_F32: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (float*)A, lda, (const float*)X, ldx); return;
    case HYACIN_F64_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (cuDoubleComplex*)A, lda, (const cuDoubleComplex*)X, ldx); return;
    case HYACIN_F32_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (cuComplex*)A, lda, (const cuComplex*)X, ldx); return;
    default: return;
  }
}
