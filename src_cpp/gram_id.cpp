
#include <hyacin.h>
#include <internal.hpp>
#include <stdexcept>

inline void ltrsm(cublasHandle_t handle, int32_t Mb, int32_t Nb, double* R, int32_t ldr)
{ double one = 1.; cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Mb, Nb, &one, R, ldr, &R[int64_t(Mb) * int64_t(ldr)], ldr); }
inline void ltrsm(cublasHandle_t handle, int32_t Mb, int32_t Nb, float* R, int32_t ldr)
{ float one = 1.f; cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Mb, Nb, &one, R, ldr, &R[int64_t(Mb) * int64_t(ldr)], ldr); }
inline void ltrsm(cublasHandle_t handle, int32_t Mb, int32_t Nb, cuDoubleComplex* R, int32_t ldr)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.); cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Mb, Nb, &one, R, ldr, &R[int64_t(Mb) * int64_t(ldr)], ldr); }
inline void ltrsm(cublasHandle_t handle, int32_t Mb, int32_t Nb, cuComplex* R, int32_t ldr)
{ cuComplex one = make_cuComplex(1.f, 0.f); cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Mb, Nb, &one, R, ldr, &R[int64_t(Mb) * int64_t(ldr)], ldr); }

template <class Btype, class Rtype, class Xtype, class Gtype>
inline int32_t interp(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, char fillmode, double epi, int32_t N, int32_t K, int32_t p, int32_t* jpiv, Xtype* X, int32_t ldx, Gtype* G, int32_t ldg, int32_t* r_ptr) {
  Rtype* potrf_work = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&potrf_work, uint64_t(65536) + (uint64_t(N) + uint64_t(1)) * uint64_t(sizeof(Rtype)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at Interpolative decomposition.");

  K = internal::Cholesky::potrfp(stream, fillmode, epi, K, p, N, G, ldg, jpiv, potrf_work, r_ptr);
  cudaFreeAsync(potrf_work, stream);
  if (K == N) { internal::scatter_matcopy(stream, handle, 'I', N, N, jpiv, (const Xtype*)nullptr, N, X, ldx); } else
  if (0 < K) {
    Btype* B = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, uint64_t(N) * uint64_t(K) * uint64_t(sizeof(Xtype)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at Interpolative decomposition.");

    internal::scatter_matcopy(stream, handle, 'U', K, N, nullptr, G, ldg, B, K);
    ltrsm(handle, K, N - K, B, K);
    internal::scatter_matcopy(stream, handle, 'I', K, N, jpiv, B, K, X, ldx);
    cudaFreeAsync(B, stream);
  }
  return K;
}

extern "C" int32_t hyacinXGinterp(const hyacinHandle_t* handle, char fillmode, double epi, int32_t N, int32_t K, hyacinPrecision_t Atype, void* X, int32_t ldx, int32_t* jpiv, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0) { return 0; } K = K <= 0 ? N : std::min(N, K);
  cudaStream_t stream = handle->cudaStream; Timer::register_replicate_kernel(stream, handle->timer);
  cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle;
  int32_t *r_ptr = (int32_t*)handle->pinnedWorkspace, oversampling = handle->RankOversampling; *r_ptr = handle->DeviceSMs;
  switch (Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return interp<double, double>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (double*)X, ldx, (double*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return interp<float, double>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (float*)X, ldx, (double*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return interp<float, double>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (__half*)X, ldx, (double*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F64)
    { return interp<float, float>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (double*)X, ldx, (float*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return interp<float, float>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (float*)X, ldx, (float*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return interp<float, float>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (__half*)X, ldx, (float*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return interp<double, double2>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (double*)X, ldx, (double2*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return interp<float, double2>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (float*)X, ldx, (double2*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_TF: if (Atype == HYACIN_F64)
    { return interp<double, float3>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (double*)X, ldx, (float3*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return interp<float, float3>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (float*)X, ldx, (float3*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return interp<double, float4>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (double*)X, ldx, (float4*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return interp<float, float4>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (float*)X, ldx, (float4*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<cuDoubleComplex, double>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuDoubleComplex*)X, ldx, (cuDoubleComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<cuComplex, double>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuComplex*)X, ldx, (cuDoubleComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<cuComplex, double>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (__half2*)X, ldx, (cuDoubleComplex*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<cuComplex, float>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuDoubleComplex*)X, ldx, (cuComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<cuComplex, float>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuComplex*)X, ldx, (cuComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<cuComplex, float>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (__half2*)X, ldx, (cuComplex*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<cuDoubleComplex, double2>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuDoubleComplex*)X, ldx, (complex_double2*)G, ldg, r_ptr); }
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<cuComplex, double2>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuComplex*)X, ldx, (complex_double2*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_TF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<cuDoubleComplex, float3>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuDoubleComplex*)X, ldx, (complex_float3*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<cuComplex, float3>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuComplex*)X, ldx, (complex_float3*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<cuDoubleComplex, float4>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuDoubleComplex*)X, ldx, (complex_float4*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<cuComplex, float4>(stream, mempool, cublasH, fillmode, epi, N, K, oversampling, jpiv, (cuComplex*)X, ldx, (complex_float4*)G, ldg, r_ptr); } else { return 0; }
    default: return 0;
  }
}
