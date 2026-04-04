
#include <hyacin.h>
#include <internal.hpp>
#include <cuComplex.h>
#include <stdexcept>

extern "C" void hyacinXtransform_bufferSize(int32_t K, hyacinPrecision_t AXtype, uint64_t* dev_work_bytes) {
  if (K <= 0) { return; }
  int32_t x_bytes; hyacinXelem('A', AXtype, nullptr, &x_bytes, nullptr);
  *dev_work_bytes = std::max(*dev_work_bytes, uint64_t(int64_t(K) * int64_t(16384) * int64_t(x_bytes)));
}

inline int32_t char_adj(char trans) { return trans == 'T' || trans == 't' || trans == 'C' || trans == 'c'; }
template <class complex_t> inline cublasOperation_t adjoint_op(char trans);
template <> inline cublasOperation_t adjoint_op<double>(char trans) { return char_adj(trans) ? CUBLAS_OP_T : CUBLAS_OP_N; }
template <> inline cublasOperation_t adjoint_op<float>(char trans) { return char_adj(trans) ? CUBLAS_OP_T : CUBLAS_OP_N; }
template <> inline cublasOperation_t adjoint_op<cuDoubleComplex>(char trans) { return char_adj(trans) ? CUBLAS_OP_C : CUBLAS_OP_N; }
template <> inline cublasOperation_t adjoint_op<cuComplex>(char trans) { return char_adj(trans) ? CUBLAS_OP_C : CUBLAS_OP_N; }

inline void nb_gemm(cublasHandle_t handle, cublasOperation_t transb, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasDgemm(handle, CUBLAS_OP_N, transb, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nb_gemm(cublasHandle_t handle, cublasOperation_t transb, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasSgemm(handle, CUBLAS_OP_N, transb, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nb_gemm(cublasHandle_t handle, cublasOperation_t transb, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgemm(handle, CUBLAS_OP_N, transb, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nb_gemm(cublasHandle_t handle, cublasOperation_t transb, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgemm(handle, CUBLAS_OP_N, transb, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }

inline void matcopy(cublasHandle_t handle, cublasOperation_t transa, int32_t Mb, int32_t Nb, const double* A, int32_t lda, double* B, int32_t ldb)
{ double one = 1., zero = 0.; cublasDgeam(handle, transa, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matcopy(cublasHandle_t handle, cublasOperation_t transa, int32_t Mb, int32_t Nb, const float* A, int32_t lda, float* B, int32_t ldb)
{ float one = 1.f, zero = 0.f; cublasSgeam(handle, transa, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matcopy(cublasHandle_t handle, cublasOperation_t transa, int32_t Mb, int32_t Nb, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, transa, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matcopy(cublasHandle_t handle, cublasOperation_t transa, int32_t Mb, int32_t Nb, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, transa, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }

template <class complex_t>
inline void ax_transform(cublasHandle_t handle, cublasOperation_t transb, int32_t M, int32_t N, int32_t K, complex_t* A, int32_t lda, const complex_t* X, int32_t ldx, void* dev_work, uint64_t dev_work_bytes) {
  int32_t rows = int32_t(dev_work_bytes / (uint64_t(K) * sizeof(complex_t))) & (~63);
  if (rows < 256) { throw std::runtime_error("Insufficient workspace for transforming A."); }
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
    nb_gemm(handle, transb, m, K, N, &A[i], lda, X, ldx, (complex_t*)dev_work, ld);
    matcopy(handle, CUBLAS_OP_N, m, K, (const complex_t*)dev_work, ld, &A[i], lda);
  }
}

extern "C" void hyacinXtransform(cublasHandle_t handle, char transx, int32_t M, int32_t N, int32_t K, hyacinPrecision_t AXtype, void* A, int32_t lda, const void* X, int32_t ldx, void* dev_work, uint64_t dev_work_bytes) {
  if ((M <= 0) || (K <= 0)) return;
  cudaStream_t stream; cublasGetStream(handle, &stream); Timer::register_kernel(stream);
  if (N <= 0) switch(AXtype) {
    case HYACIN_F64: matcopy(handle, adjoint_op<double>(transx), M, K, (const double*)X, ldx, (double*)A, lda); return;
    case HYACIN_F32: matcopy(handle, adjoint_op<float>(transx), M, K, (const float*)X, ldx, (float*)A, lda); return;
    case HYACIN_F64_COMPLEX: matcopy(handle, adjoint_op<cuDoubleComplex>(transx), M, K, (const cuDoubleComplex*)X, ldx, (cuDoubleComplex*)A, lda); return;
    case HYACIN_F32_COMPLEX: matcopy(handle, adjoint_op<cuComplex>(transx), M, K, (const cuComplex*)X, ldx, (cuComplex*)A, lda); return;
    default: return;
  }
  else switch(AXtype) {
    case HYACIN_F64: ax_transform(handle, adjoint_op<double>(transx), M, N, K, (double*)A, lda, (const double*)X, ldx, dev_work, dev_work_bytes); return;
    case HYACIN_F32: ax_transform(handle, adjoint_op<float>(transx), M, N, K, (float*)A, lda, (const float*)X, ldx, dev_work, dev_work_bytes); return;
    case HYACIN_F64_COMPLEX: ax_transform(handle, adjoint_op<cuDoubleComplex>(transx), M, N, K, (cuDoubleComplex*)A, lda, (const cuDoubleComplex*)X, ldx, dev_work, dev_work_bytes); return;
    case HYACIN_F32_COMPLEX: ax_transform(handle, adjoint_op<cuComplex>(transx), M, N, K, (cuComplex*)A, lda, (const cuComplex*)X, ldx, dev_work, dev_work_bytes); return;
    default: return;
  }
}
