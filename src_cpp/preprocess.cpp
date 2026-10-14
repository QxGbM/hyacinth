
#include <hyacin.h>
#include <internal.hpp>
#include <stdexcept>
#define CUDART_NEG_ONE_FP16 __ushort_as_half((unsigned short)0xBC00U)

inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, &zero, C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &zero, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, const __half* B, int32_t ldb, __half* C, int32_t ldc)
{ __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16; cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &zero, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc) {
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_C_64F, lda, B, CUDA_C_64F, ldb, &zero, C, CUDA_C_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
}
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc) {
  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_C_32F, lda, B, CUDA_C_32F, ldb, &zero, C, CUDA_C_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
}

inline void nc_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., minus_one = -1.; cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &minus_one, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, &one, C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT); }
inline void nc_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, minus_one = -1.; cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &minus_one, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &one, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT); }
inline void nc_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, const __half* B, int32_t ldb, __half* C, int32_t ldc)
{ __half one = CUDART_ONE_FP16, minus_one = CUDART_NEG_ONE_FP16; cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &minus_one, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &one, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT); }
inline void nc_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc) {
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), minus_one = make_cuDoubleComplex(-1., 0.);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_C, M, N, K, &minus_one, A, CUDA_C_64F, lda, B, CUDA_C_64F, ldb, &one, C, CUDA_C_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
}
inline void nc_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc) {
  cuComplex one = make_cuComplex(1.f, 0.f), minus_one = make_cuComplex(-1.f, 0.f);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_C, M, N, K, &minus_one, A, CUDA_C_32F, lda, B, CUDA_C_32F, ldb, &one, C, CUDA_C_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
}

template <class matrix_t> inline void preprocess(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, char transpose, int32_t M, int32_t N, int32_t Vcols, const matrix_t* A, int32_t lda, const matrix_t* V, int32_t ldv, matrix_t* B, int32_t ldb) {
  internal::scatter_matcopy(stream, handle, transpose, M, N, A, lda, B, ldb);
  if (0 < Vcols) {
    const int32_t rows = 16384;
    uint64_t work_bytes = uint64_t(rows) * uint64_t(Vcols) * uint64_t(sizeof(matrix_t));
    matrix_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes, mempool, stream))
      throw std::runtime_error("Workspace allocation failed at preprocessing A.");
    for (int32_t i = 0; i < M; i += rows) {
      int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
      nn_gemm(handle, m, Vcols, N, &B[i], ldb, V, ldv, dev_work, ld);
      nc_gemm(handle, m, N, Vcols, dev_work, ld, V, ldv, &B[i], ldb);
    }
  }
}

template <> inline void preprocess<__half2>(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, char transpose, int32_t M, int32_t N, int32_t Vcols, const __half2* A, int32_t lda, const __half2* V, int32_t ldv, __half2* B, int32_t ldb) {
  if (0 < Vcols) {
    const int32_t rows = 16384;
    uint64_t work_bytes = uint64_t(rows) * uint64_t(Vcols) * uint64_t(sizeof(cuComplex));
    uint64_t devA_bytes = uint64_t(rows) * uint64_t(N) * uint64_t(sizeof(cuComplex));
    uint64_t devV_bytes = uint64_t(N) * uint64_t(Vcols) * uint64_t(sizeof(cuComplex));
    uint8_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes + devA_bytes + devV_bytes, mempool, stream))
      throw std::runtime_error("Workspace allocation failed at transforming A.");

    cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f), minus_one = make_cuComplex(-1.f, 0.f);
    cuComplex* f32a = (cuComplex*)&dev_work[work_bytes], *f32v = (cuComplex*)&dev_work[work_bytes + devA_bytes];
    internal::scatter_matcopy(stream, handle, 'A', N, Vcols, V, ldv, f32v, N);
    int64_t incA = (transpose == 'T' || transpose == 'C') ? (int64_t(rows) * int64_t(lda)) : int64_t(rows);
    for (int32_t i = 0; i < M; i += rows) {
      int32_t m = std::min(M - i, rows), ld = std::min((m + 63) & (~63), rows);
      internal::scatter_matcopy(stream, handle, transpose, m, N, A, lda, f32a, ld); A = &A[incA];
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, Vcols, N, &one, f32a, CUDA_C_32F, ld, f32v, CUDA_C_32F, N, &zero, dev_work, CUDA_C_32F, ld, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_C, m, N, Vcols, &minus_one, dev_work, CUDA_C_32F, ld, f32v, CUDA_C_32F, N, &one, f32a, CUDA_C_32F, ld, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
      internal::scatter_matcopy(stream, handle, 'A', m, N, f32a, ld, &B[i], ldb);
    }
  } else { internal::scatter_matcopy(stream, handle, transpose, M, N, A, lda, B, ldb); }
}

extern "C" void hyacinXpreprocess(const hyacinHandle_t* handle, char transpose, int32_t M, int32_t N, int32_t Vcols, hyacinPrecision_t Atype, const void* A, int32_t lda, const void* V, int32_t ldv, void* B, int32_t ldb) {
  if ((M <= 0) || (N <= 0)) { return; } transpose = (transpose == 't') ? 'T' : (transpose == 'c') ? 'C' : transpose;
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  switch (Atype) {
    case HYACIN_F64: preprocess(handle->cudaStream, handle->mempool, handle->cublasHandle, transpose, M, N, Vcols, (const double*)A, lda, (const double*)V, ldv, (double*)B, ldb); return;
    case HYACIN_F32: preprocess(handle->cudaStream, handle->mempool, handle->cublasHandle, transpose, M, N, Vcols, (const float*)A, lda, (const float*)V, ldv, (float*)B, ldb); return;
    case HYACIN_F16: preprocess(handle->cudaStream, handle->mempool, handle->cublasHandle, transpose, M, N, Vcols, (const __half*)A, lda, (const __half*)V, ldv, (__half*)B, ldb); return;
    case HYACIN_F64_COMPLEX: preprocess(handle->cudaStream, handle->mempool, handle->cublasHandle, transpose, M, N, Vcols, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)V, ldv, (cuDoubleComplex*)B, ldb); return;
    case HYACIN_F32_COMPLEX: preprocess(handle->cudaStream, handle->mempool, handle->cublasHandle, transpose, M, N, Vcols, (const cuComplex*)A, lda, (const cuComplex*)V, ldv, (cuComplex*)B, ldb); return;
    case HYACIN_F16_COMPLEX: preprocess(handle->cudaStream, handle->mempool, handle->cublasHandle, transpose, M, N, Vcols, (const __half2*)A, lda, (const __half2*)V, ldv, (__half2*)B, ldb); return;
    default: return;
  }
}
