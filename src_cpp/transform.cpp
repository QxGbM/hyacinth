
#include <hyacin.h>
#include <internal.hpp>
#include <cuComplex.h>
#include <stdexcept>

inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, const __half* B, int32_t ldb, __half* C, int32_t ldc)
{ __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16; cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }

template <class matrix_t> inline void ax_transform(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const matrix_t* Ain, int32_t lda_in, matrix_t* Aout, int32_t lda_out, const matrix_t* X, int32_t ldx) {
  if (Ain == Aout && lda_in == lda_out) {
    const int32_t rows = 16384;
    matrix_t* dev_work = nullptr;
    uint64_t dev_work_bytes = uint64_t(rows) * uint64_t(K) * uint64_t(sizeof(matrix_t));
    if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes, stream))
      throw std::runtime_error("Workspace allocation failed at transforming A.");
    for (int32_t i = 0; i < M; i += rows) {
      int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
      nn_gemm(handle, m, K, N, &Ain[i], lda_in, X, ldx, dev_work, ld);
      internal::scatter_matcopy(stream, handle, 'A', m, K, nullptr, dev_work, ld, &Aout[i], lda_out);
    }
    cudaFreeAsync(dev_work, stream);
  } else { nn_gemm(handle, M, K, N, Ain, lda_in, X, ldx, Aout, lda_out); }
}

template <> inline void ax_transform<__half2>(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const __half2* Ain, int32_t lda_in, __half2* Aout, int32_t lda_out, const __half2* X, int32_t ldx) {
  const int32_t rows = 16384;
  uint8_t* dev_work = nullptr;
  uint64_t dev_work_bytes = uint64_t(rows) * uint64_t(K) * uint64_t(sizeof(cuComplex));
  uint64_t devA_bytes = uint64_t(rows) * uint64_t(N) * uint64_t(sizeof(cuComplex));
  uint64_t devX_bytes = uint64_t(N) * uint64_t(K) * uint64_t(sizeof(cuComplex));
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes + devA_bytes + devX_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at transforming A.");

  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
  cuComplex* f32a = (cuComplex*)&dev_work[dev_work_bytes], *f32x = (cuComplex*)&dev_work[dev_work_bytes + devA_bytes];
  internal::scatter_matcopy(stream, handle, 'A', N, K, nullptr, X, ldx, f32x, N);
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
    internal::scatter_matcopy(stream, handle, 'A', m, N, nullptr, &Ain[i], lda_in, f32a, ld);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, K, N, &one, f32a, CUDA_C_32F, ld, f32x, CUDA_C_32F, N, &zero, dev_work, CUDA_C_32F, ld, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    internal::scatter_matcopy(stream, handle, 'A', m, K, nullptr, (cuComplex*)dev_work, ld, &Aout[i], lda_out);
  }
  cudaFreeAsync(dev_work, stream);
}

extern "C" void hyacinXtransform(hyacinHandle_t handle, int32_t M, int32_t N, int32_t K, hyacinPrecision_t Atype, const void* Ain, int32_t lda_in, void* Aout, int32_t lda_out, const void* X, int32_t ldx) {
  if ((M <= 0) || (K <= 0)) return;
  Timer::register_kernel(handle.cudaStream, handle.timer);
  if (N <= 0) switch(Atype) {
    case HYACIN_F64: internal::scatter_matcopy(handle.cudaStream, handle.cublasHandle, 'A', M, K, nullptr, (const double*)X, ldx, (double*)Aout, lda_out); return;
    case HYACIN_F32: internal::scatter_matcopy(handle.cudaStream, handle.cublasHandle, 'A', M, K, nullptr, (const float*)X, ldx, (float*)Aout, lda_out); return;
    case HYACIN_F16: internal::scatter_matcopy(handle.cudaStream, handle.cublasHandle, 'A', M, K, nullptr, (const __half*)X, ldx, (__half*)Aout, lda_out); return;
    case HYACIN_F64_COMPLEX: internal::scatter_matcopy(handle.cudaStream, handle.cublasHandle, 'A', M, K, nullptr, (const cuDoubleComplex*)X, ldx, (cuDoubleComplex*)Aout, lda_out); return;
    case HYACIN_F32_COMPLEX: internal::scatter_matcopy(handle.cudaStream, handle.cublasHandle, 'A', M, K, nullptr, (const cuComplex*)X, ldx, (cuComplex*)Aout, lda_out); return;
    case HYACIN_F16_COMPLEX: internal::scatter_matcopy(handle.cudaStream, handle.cublasHandle, 'A', M, K, nullptr, (const __half2*)X, ldx, (__half2*)Aout, lda_out); return;
    default: return;
  } else switch(Atype) {
    case HYACIN_F64: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (const double*)Ain, lda_in, (double*)Aout, lda_out, (const double*)X, ldx); return;
    case HYACIN_F32: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (const float*)Ain, lda_in, (float*)Aout, lda_out, (const float*)X, ldx); return;
    case HYACIN_F16: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (const __half*)Ain, lda_in, (__half*)Aout, lda_out, (const __half*)X, ldx); return;
    case HYACIN_F64_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (const cuDoubleComplex*)Ain, lda_in, (cuDoubleComplex*)Aout, lda_out, (const cuDoubleComplex*)X, ldx); return;
    case HYACIN_F32_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (const cuComplex*)Ain, lda_in, (cuComplex*)Aout, lda_out, (const cuComplex*)X, ldx); return;
    case HYACIN_F16_COMPLEX: ax_transform(handle.cudaStream, handle.cublasHandle, M, N, K, (const __half2*)Ain, lda_in, (__half2*)Aout, lda_out, (const __half2*)X, ldx); return;
    default: return;
  }
}
