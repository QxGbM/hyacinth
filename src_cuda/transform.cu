
#include <hyacin.h>
#include <internal.hpp>
#include <stdexcept>

template <class T> __global__ void gather_cpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) { B[y + x * ldb] = A[y + int64_t(jpiv[x] - 1) * lda]; }
};

template <class matrix_t> inline void gather_cols(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t K, const matrix_t* Ain, int32_t lda_in, matrix_t* Aout, int32_t lda_out, const int32_t* jpiv) {
  if (Ain == Aout && lda_in == lda_out) {
    const int32_t rows = 16384;
    uint64_t work_bytes = uint64_t(rows) * uint64_t(K) * uint64_t(sizeof(matrix_t));
    matrix_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes, mempool, stream))
      throw std::runtime_error("Workspace allocation failed at transforming A.");
    for (int32_t i = 0; i < M; i += rows) {
      int32_t m = std::min(M - i, rows);
      gather_cpy_kernel <<< dim3(uint32_t(M + 511) >> 9, uint32_t(K)), 512, 0, stream >>> (int64_t(m), jpiv, Ain, int64_t(lda_in), dev_work, int64_t(m));
      internal::scatter_matcopy(stream, handle, 'A', m, K, dev_work, m, &Aout[i], lda_out);
    }
    cudaFreeAsync(dev_work, stream);
  } else { gather_cpy_kernel <<< dim3(uint32_t(M + 511) >> 9, uint32_t(K)), 512, 0, stream >>> (int64_t(M), jpiv, Ain, int64_t(lda_in), Aout, int64_t(lda_out)); }
}

inline void an_gemm(cublasHandle_t handle, cublasOperation_t opA, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasGemmEx(handle, opA == CUBLAS_OP_C ? CUBLAS_OP_T : opA, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, &zero, C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT); }
inline void an_gemm(cublasHandle_t handle, cublasOperation_t opA, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasGemmEx(handle, opA == CUBLAS_OP_C ? CUBLAS_OP_T : opA, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &zero, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT); }
inline void an_gemm(cublasHandle_t handle, cublasOperation_t opA, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, const __half* B, int32_t ldb, __half* C, int32_t ldc)
{ __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16; cublasGemmEx(handle, opA == CUBLAS_OP_C ? CUBLAS_OP_T : opA, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &zero, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT); }
inline void an_gemm(cublasHandle_t handle, cublasOperation_t opA, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc) {
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
  cublasGemmEx(handle, opA, CUBLAS_OP_N, M, N, K, &one, A, CUDA_C_64F, lda, B, CUDA_C_64F, ldb, &zero, C, CUDA_C_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
}
inline void an_gemm(cublasHandle_t handle, cublasOperation_t opA, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc) {
  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
  cublasGemmEx(handle, opA, CUBLAS_OP_N, M, N, K, &one, A, CUDA_C_32F, lda, B, CUDA_C_32F, ldb, &zero, C, CUDA_C_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
}

template <class matrix_t> inline void ax_transform(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const matrix_t* Ain, int32_t lda_in, matrix_t* Aout, int32_t lda_out, const matrix_t* X, int32_t ldx) {
  if (Ain == Aout && lda_in == lda_out) {
    const int32_t rows = 16384;
    uint64_t work_bytes = uint64_t(rows) * uint64_t(K) * uint64_t(sizeof(matrix_t));
    matrix_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes, mempool, stream))
      throw std::runtime_error("Workspace allocation failed at transforming A.");
    for (int32_t i = 0; i < M; i += rows) {
      int32_t m = std::min(M - i, rows);
      an_gemm(handle, CUBLAS_OP_N, m, K, N, &Ain[i], lda_in, X, ldx, dev_work, m);
      internal::scatter_matcopy(stream, handle, 'A', m, K, dev_work, m, &Aout[i], lda_out);
    }
    cudaFreeAsync(dev_work, stream);
  } else { an_gemm(handle, CUBLAS_OP_N, M, K, N, Ain, lda_in, X, ldx, Aout, lda_out); }
}

template <> inline void ax_transform<__half2>(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const __half2* Ain, int32_t lda_in, __half2* Aout, int32_t lda_out, const __half2* X, int32_t ldx) {
  const int32_t rows = 16384;
  uint64_t work_bytes = uint64_t(rows) * uint64_t(K) * uint64_t(sizeof(cuComplex));
  uint64_t devA_bytes = uint64_t(rows) * uint64_t(N) * uint64_t(sizeof(cuComplex));
  uint64_t devX_bytes = uint64_t(N) * uint64_t(K) * uint64_t(sizeof(cuComplex));
  uint8_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes + devA_bytes + devX_bytes, mempool, stream))
    throw std::runtime_error("Workspace allocation failed at transforming A.");

  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
  cuComplex* f32a = (cuComplex*)&dev_work[work_bytes], *f32x = (cuComplex*)&dev_work[work_bytes + devA_bytes];
  internal::scatter_matcopy(stream, handle, 'A', N, K, X, ldx, f32x, N);
  for (int32_t i = 0; i < M; i += rows) {
    int32_t m = std::min(M - i, rows);
    internal::scatter_matcopy(stream, handle, 'A', m, N, &Ain[i], lda_in, f32a, m);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, K, N, &one, f32a, CUDA_C_32F, m, f32x, CUDA_C_32F, N, &zero, dev_work, CUDA_C_32F, m, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    internal::scatter_matcopy(stream, handle, 'A', m, K, (cuComplex*)dev_work, m, &Aout[i], lda_out);
  }
  cudaFreeAsync(dev_work, stream);
}

template <class matrix_t> inline void xa_transform(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const matrix_t* Ain, int32_t lda_in, matrix_t* Aout, int32_t lda_out, const matrix_t* X, int32_t ldx) {
  if (Ain == Aout && lda_in == lda_out) {
    const int32_t cols = 16384;
    uint64_t work_bytes = uint64_t(K) * uint64_t(cols) * uint64_t(sizeof(matrix_t));
    matrix_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes, mempool, stream))
      throw std::runtime_error("Workspace allocation failed at transforming A.");
    for (int32_t j = 0; j < N; j += cols) {
      int32_t n = std::min(N - j, cols);
      an_gemm(handle, CUBLAS_OP_C, K, n, M, X, ldx, &Ain[int64_t(j) * int64_t(lda_in)], lda_in, dev_work, K);
      internal::scatter_matcopy(stream, handle, 'A', K, n, dev_work, K, &Aout[int64_t(j) * int64_t(lda_out)], lda_out);
    }
    cudaFreeAsync(dev_work, stream);
  } else { an_gemm(handle, CUBLAS_OP_C, K, N, M, X, ldx, Ain, lda_in, Aout, lda_out); }
}

template <> inline void xa_transform<__half2>(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const __half2* Ain, int32_t lda_in, __half2* Aout, int32_t lda_out, const __half2* X, int32_t ldx) {
  const int32_t cols = 16384;
  uint64_t work_bytes = uint64_t(K) * uint64_t(cols) * uint64_t(sizeof(cuComplex));
  uint64_t devA_bytes = uint64_t(M) * uint64_t(cols) * uint64_t(sizeof(cuComplex));
  uint64_t devX_bytes = uint64_t(M) * uint64_t(K) * uint64_t(sizeof(cuComplex));
  uint8_t* dev_work = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dev_work, work_bytes + devA_bytes + devX_bytes, mempool, stream))
    throw std::runtime_error("Workspace allocation failed at transforming A.");

  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
  cuComplex* f32a = (cuComplex*)&dev_work[work_bytes], *f32x = (cuComplex*)&dev_work[work_bytes + devA_bytes];
  internal::scatter_matcopy(stream, handle, 'A', M, K, X, ldx, f32x, M);
  for (int32_t j = 0; j < N; j += cols) {
    int32_t n = std::min(N - j, cols);
    internal::scatter_matcopy(stream, handle, 'A', M, n, &Ain[int64_t(j) * int64_t(lda_in)], lda_in, f32a, M);
    cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_N, K, n, M, &one, f32a, CUDA_C_32F, M, f32x, CUDA_C_32F, M, &zero, dev_work, CUDA_C_32F, K, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    internal::scatter_matcopy(stream, handle, 'A', K, n, (cuComplex*)dev_work, K, &Aout[int64_t(j) * int64_t(lda_out)], lda_out);
  }
  cudaFreeAsync(dev_work, stream);
}

extern "C" void hyacinXtransform(const hyacinHandle_t* handle, int32_t M, int32_t N, int32_t K, hyacinPrecision_t Atype, const void* Ain, int32_t lda_in, void* Aout, int32_t lda_out, char Xtype, const void* X, int32_t ldx) {
  if ((M <= 0) || (K <= 0)) { return; }
  cudaStream_t stream = handle->cudaStream; Timer::register_distribute_kernel(stream, handle->timer);
  if (K <= N && (Xtype == 'J' || Xtype == 'j')) switch (Atype) {
    case HYACIN_F64: gather_cols(stream, handle->mempool, handle->cublasHandle, M, K, (const double*)Ain, lda_in, (double*)Aout, lda_out, (const int32_t*)X); return;
    case HYACIN_F32: gather_cols(stream, handle->mempool, handle->cublasHandle, M, K, (const float*)Ain, lda_in, (float*)Aout, lda_out, (const int32_t*)X); return;
    case HYACIN_F16: gather_cols(stream, handle->mempool, handle->cublasHandle, M, K, (const __half*)Ain, lda_in, (__half*)Aout, lda_out, (const int32_t*)X); return;
    case HYACIN_F64_COMPLEX: gather_cols(stream, handle->mempool, handle->cublasHandle, M, K, (const cuDoubleComplex*)Ain, lda_in, (cuDoubleComplex*)Aout, lda_out, (const int32_t*)X); return;
    case HYACIN_F32_COMPLEX: gather_cols(stream, handle->mempool, handle->cublasHandle, M, K, (const cuComplex*)Ain, lda_in, (cuComplex*)Aout, lda_out, (const int32_t*)X); return;
    case HYACIN_F16_COMPLEX: gather_cols(stream, handle->mempool, handle->cublasHandle, M, K, (const __half2*)Ain, lda_in, (__half2*)Aout, lda_out, (const int32_t*)X); return;
    default: return;
  } else if (Xtype == 'V' || Xtype == 'v') { if (N <= 0) switch (Atype) {
    case HYACIN_F64: internal::scatter_matcopy(stream, handle->cublasHandle, 'A', M, K, (const double*)X, ldx, (double*)Aout, lda_out); return;
    case HYACIN_F32: internal::scatter_matcopy(stream, handle->cublasHandle, 'A', M, K, (const float*)X, ldx, (float*)Aout, lda_out); return;
    case HYACIN_F16: internal::scatter_matcopy(stream, handle->cublasHandle, 'A', M, K, (const __half*)X, ldx, (__half*)Aout, lda_out); return;
    case HYACIN_F64_COMPLEX: internal::scatter_matcopy(stream, handle->cublasHandle, 'A', M, K, (const cuDoubleComplex*)X, ldx, (cuDoubleComplex*)Aout, lda_out); return;
    case HYACIN_F32_COMPLEX: internal::scatter_matcopy(stream, handle->cublasHandle, 'A', M, K, (const cuComplex*)X, ldx, (cuComplex*)Aout, lda_out); return;
    case HYACIN_F16_COMPLEX: internal::scatter_matcopy(stream, handle->cublasHandle, 'A', M, K, (const __half2*)X, ldx, (__half2*)Aout, lda_out); return;
    default: return;
  } else switch (Atype) {
    case HYACIN_F64: ax_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const double*)Ain, lda_in, (double*)Aout, lda_out, (const double*)X, ldx); return;
    case HYACIN_F32: ax_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const float*)Ain, lda_in, (float*)Aout, lda_out, (const float*)X, ldx); return;
    case HYACIN_F16: ax_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const __half*)Ain, lda_in, (__half*)Aout, lda_out, (const __half*)X, ldx); return;
    case HYACIN_F64_COMPLEX: ax_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const cuDoubleComplex*)Ain, lda_in, (cuDoubleComplex*)Aout, lda_out, (const cuDoubleComplex*)X, ldx); return;
    case HYACIN_F32_COMPLEX: ax_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const cuComplex*)Ain, lda_in, (cuComplex*)Aout, lda_out, (const cuComplex*)X, ldx); return;
    case HYACIN_F16_COMPLEX: ax_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const __half2*)Ain, lda_in, (__half2*)Aout, lda_out, (const __half2*)X, ldx); return;
    default: return;
  }} else if (0 < N && (Xtype == 'U' || Xtype == 'u')) switch(Atype) {
    case HYACIN_F64: xa_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const double*)Ain, lda_in, (double*)Aout, lda_out, (const double*)X, ldx); return;
    case HYACIN_F32: xa_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const float*)Ain, lda_in, (float*)Aout, lda_out, (const float*)X, ldx); return;
    case HYACIN_F16: xa_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const __half*)Ain, lda_in, (__half*)Aout, lda_out, (const __half*)X, ldx); return;
    case HYACIN_F64_COMPLEX: xa_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const cuDoubleComplex*)Ain, lda_in, (cuDoubleComplex*)Aout, lda_out, (const cuDoubleComplex*)X, ldx); return;
    case HYACIN_F32_COMPLEX: xa_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const cuComplex*)Ain, lda_in, (cuComplex*)Aout, lda_out, (const cuComplex*)X, ldx); return;
    case HYACIN_F16_COMPLEX: xa_transform(stream, handle->mempool, handle->cublasHandle, M, N, K, (const __half2*)Ain, lda_in, (__half2*)Aout, lda_out, (const __half2*)X, ldx); return;
    default: return;
  }
}
