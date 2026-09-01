
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
inline int32_t diag_piv_dispatcher(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t N, int32_t K, int32_t p, int32_t* jpiv, Xtype* X, int32_t ldx, Gtype* G, int32_t ldg, void* pinned_work) {
  uint64_t dev_work_bytes = uint64_t(std::max(int64_t(sizeof(Xtype)) * int64_t(N) * int64_t(std::min(N, K)), int64_t(8192) + int64_t(sizeof(Rtype)) * int64_t(N)));
  void* dev_work = nullptr; 
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at Interpolative decomposition.");

  K = internal::Cholesky::potrfp(stream, handle, fillmode, epi, K, p, N, G, ldg, jpiv, (Rtype*)dev_work, pinned_work);
  if (0 < K) {
    Btype* B = (Btype*)dev_work;
    internal::scatter_matcopy(stream, handle, 'A', K, N, nullptr, G, ldg, B, K);
    if (K < N) { ltrsm(handle, K, N - K, B, K); }
    internal::scatter_matcopy(stream, handle, 'I', K, N, jpiv, B, K, X, ldx);
  }
  cudaFreeAsync(dev_work, stream);
  return K;
}

extern "C" int32_t hyacinXGinterp(hyacinHandle_t handle, char fillmode, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t Atype, void* X, int32_t ldx, int32_t* jpiv, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0 || K <= 0) { return 0; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch (Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return diag_piv_dispatcher<double, double>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (double*)X, ldx, (double*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F32)
    { return diag_piv_dispatcher<float, double>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (float*)X, ldx, (double*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F32)
    { return diag_piv_dispatcher<float, float>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (float*)X, ldx, (float*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F16)
    { return diag_piv_dispatcher<float, float>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (__half*)X, ldx, (float*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return diag_piv_dispatcher<double, double2>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (double*)X, ldx, (double2*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return diag_piv_dispatcher<double, float4>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (double*)X, ldx, (float4*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return diag_piv_dispatcher<cuDoubleComplex, double>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (cuDoubleComplex*)X, ldx, (cuDoubleComplex*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return diag_piv_dispatcher<cuComplex, double>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (cuComplex*)X, ldx, (cuDoubleComplex*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F32_COMPLEX)
    { return diag_piv_dispatcher<cuComplex, float>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (cuComplex*)X, ldx, (cuComplex*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return diag_piv_dispatcher<cuComplex, float>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (__half2*)X, ldx, (cuComplex*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return diag_piv_dispatcher<cuDoubleComplex, double2>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (cuDoubleComplex*)X, ldx, (complex_double2*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return diag_piv_dispatcher<cuDoubleComplex, float4>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, jpiv, (cuDoubleComplex*)X, ldx, (complex_float4*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    default: return 0;
  }
}
