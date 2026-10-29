
#include <hyacin.h>
#include <internal.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

static inline void ltrsm(cublasHandle_t handle, int32_t M, int32_t N, double* R, int32_t ldr)
{ double one = 1.; cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &one, R, ldr, &R[int64_t(M) * int64_t(ldr)], ldr); }
static inline void ltrsm(cublasHandle_t handle, int32_t M, int32_t N, float* R, int32_t ldr)
{ float one = 1.f; cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &one, R, ldr, &R[int64_t(M) * int64_t(ldr)], ldr); }
static inline void ltrsm(cublasHandle_t handle, int32_t M, int32_t N, cuDoubleComplex* R, int32_t ldr)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.); cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &one, R, ldr, &R[int64_t(M) * int64_t(ldr)], ldr); }
static inline void ltrsm(cublasHandle_t handle, int32_t M, int32_t N, cuComplex* R, int32_t ldr)
{ cuComplex one = make_cuComplex(1.f, 0.f); cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &one, R, ldr, &R[int64_t(M) * int64_t(ldr)], ldr); }

template <class matrix_t> static inline void write_log(int32_t M, int32_t N, FILE* logger) { if (logger) {
  if constexpr(std::is_same_v<matrix_t, double>)
  { std::fprintf(logger, "TRSM-LEFT-FP64 [M=%d] [N=%d]\n", M, N); } else
  if constexpr(std::is_same_v<matrix_t, float>)
  { std::fprintf(logger, "TRSM-LEFT-FP32 [M=%d] [N=%d]\n", M, N); } else
  if constexpr(std::is_same_v<matrix_t, cuDoubleComplex>)
  { std::fprintf(logger, "TRSM-LEFT-FP64-COMPLEX [M=%d] [N=%d]\n", M, N); } else
  if constexpr(std::is_same_v<matrix_t, cuComplex>)
  { std::fprintf(logger, "TRSM-LEFT-FP32-COMPLEX [M=%d] [N=%d]\n", M, N); }
}}

template <class Rtype, class Xtype, class Gtype>
static inline int32_t interp(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t handle, char fillmode, double eps, int32_t N, int32_t K, int32_t* jpiv, Xtype* X, int32_t ldx, Gtype* G, int32_t ldg, int32_t* r_ptr, void* logger) {
  using matrix_t = typename std::conditional_t<std::is_same_v<Xtype, cuDoubleComplex> || std::is_same_v<Xtype, cuComplex> || std::is_same_v<Xtype, __half2>,
    std::conditional_t<std::is_same_v<Xtype, cuDoubleComplex>, cuDoubleComplex, cuComplex>, std::conditional_t<std::is_same_v<Xtype, double>, double, float>>;
  using real_t = typename std::conditional_t<std::is_same_v<matrix_t, cuDoubleComplex>, double, std::conditional_t<std::is_same_v<matrix_t, cuComplex>, float, matrix_t>>;

  Rtype* potrf_work = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&potrf_work, uint64_t(98304) + uint64_t(N) * uint64_t(sizeof(Rtype)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at Interpolative decomposition.");

  constexpr double machine_eps = double(std::numeric_limits<real_t>::epsilon()); eps = std::clamp(std::abs(eps), machine_eps, 1.);
  K = internal::Cholesky::potrfp(stream, fillmode, eps, K, N, G, ldg, jpiv, potrf_work, r_ptr, logger).first;
  cudaFreeAsync(potrf_work, stream);
  if (K == N) { internal::scatter_matcopy(stream, handle, 'I', N, N, (const Xtype*)nullptr, N, X, ldx, jpiv); } else
  if (0 < K) {
    matrix_t* B = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&B, uint64_t(N) * uint64_t(K) * uint64_t(sizeof(Xtype)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at Interpolative decomposition.");

    internal::scatter_matcopy(stream, handle, 'U', K, N, G, ldg, B, K);
    ltrsm(handle, K, N - K, B, K); write_log<matrix_t>(K, N - K, (FILE*)logger);
    internal::scatter_matcopy(stream, handle, 'I', K, N, B, K, X, ldx, jpiv);
    cudaFreeAsync(B, stream);
  }
  return K;
}

extern "C" int32_t hyacinXGinterp(const hyacinHandle_t* handle, char fillmode, double eps, int32_t N, int32_t K, hyacinPrecision_t Atype, void* X, int32_t ldx, int32_t* jpiv, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0) { return 0; } K = K <= 0 ? N : std::min(N, K);
  cudaStream_t stream = handle->cudaStream; Timer::register_replicate_kernel(stream, handle->timer);
  cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle;
  int32_t *r_ptr = (int32_t*)handle->pinnedWorkspace; *r_ptr = handle->DeviceSMs;
  switch (Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return interp<double>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (double*)X, ldx, (double*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32)
    { return interp<double>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (float*)X, ldx, (double*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16)
    { return interp<double>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half*)X, ldx, (double*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F64)
    { return interp<float>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (double*)X, ldx, (float*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32)
    { return interp<float>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (float*)X, ldx, (float*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16)
    { return interp<float>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half*)X, ldx, (float*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return interp<double2>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (double*)X, ldx, (double2*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32)
    { return interp<double2>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (float*)X, ldx, (double2*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16)
    { return interp<double2>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half*)X, ldx, (double2*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_TF: if (Atype == HYACIN_F64)
    { return interp<float3>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (double*)X, ldx, (float3*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32)
    { return interp<float3>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (float*)X, ldx, (float3*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16)
    { return interp<float3>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half*)X, ldx, (float3*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return interp<float4>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (double*)X, ldx, (float4*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32)
    { return interp<float4>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (float*)X, ldx, (float4*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16)
    { return interp<float4>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half*)X, ldx, (float4*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<double>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuDoubleComplex*)X, ldx, (cuDoubleComplex*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<double>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuComplex*)X, ldx, (cuDoubleComplex*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<double>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half2*)X, ldx, (cuDoubleComplex*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<float>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuDoubleComplex*)X, ldx, (cuComplex*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<float>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuComplex*)X, ldx, (cuComplex*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<float>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half2*)X, ldx, (cuComplex*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<double2>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuDoubleComplex*)X, ldx, (complex_double2*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<double2>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuComplex*)X, ldx, (complex_double2*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<double2>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half2*)X, ldx, (complex_double2*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_TF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<float3>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuDoubleComplex*)X, ldx, (complex_float3*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<float3>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuComplex*)X, ldx, (complex_float3*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<float3>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half2*)X, ldx, (complex_float3*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return interp<float4>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuDoubleComplex*)X, ldx, (complex_float4*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return interp<float4>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (cuComplex*)X, ldx, (complex_float4*)G, ldg, r_ptr, handle->logger); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return interp<float4>(stream, mempool, cublasH, fillmode, eps, N, K, jpiv, (__half2*)X, ldx, (complex_float4*)G, ldg, r_ptr, handle->logger); } else { return 0; }
    default: return 0;
  }
}
