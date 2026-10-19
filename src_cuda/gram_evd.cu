
#include <hyacin.h>
#include <internal.hpp>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <limits>
#include <algorithm>
#include <stdexcept>

inline cusolverStatus_t Xgesvj_bufferSize(cusolverDnHandle_t handle, int32_t m, int32_t n, const double* A, int32_t lda, const double* S, const double* U, int32_t ldu, const double* V, int32_t ldv, int32_t* lwork, gesvdjInfo_t params)
{ return cusolverDnDgesvdj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, lwork, params); }
inline cusolverStatus_t Xgesvj_bufferSize(cusolverDnHandle_t handle, int32_t m, int32_t n, const float* A, int32_t lda, const float* S, const float* U, int32_t ldu, const float* V, int32_t ldv, int32_t* lwork, gesvdjInfo_t params)
{ return cusolverDnSgesvdj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, lwork, params); }
inline cusolverStatus_t Xgesvj_bufferSize(cusolverDnHandle_t handle, int32_t m, int32_t n, const cuDoubleComplex* A, int32_t lda, const double* S, const cuDoubleComplex* U, int32_t ldu, const cuDoubleComplex* V, int32_t ldv, int32_t* lwork, gesvdjInfo_t params)
{ return cusolverDnZgesvdj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, lwork, params); }
inline cusolverStatus_t Xgesvj_bufferSize(cusolverDnHandle_t handle, int32_t m, int32_t n, const cuComplex* A, int32_t lda, const float* S, const cuComplex* U, int32_t ldu, const cuComplex* V, int32_t ldv, int32_t* lwork, gesvdjInfo_t params)
{ return cusolverDnCgesvdj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, lwork, params); }

inline cusolverStatus_t Xgesvj(cusolverDnHandle_t handle, int32_t m, int32_t n, double* A, int32_t lda, double* S, double* U, int32_t ldu, double* V, int32_t ldv, double* work, int32_t lwork, gesvdjInfo_t params)
{ return cusolverDnDgesvdj(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, nullptr, params); }
inline cusolverStatus_t Xgesvj(cusolverDnHandle_t handle, int32_t m, int32_t n, float* A, int32_t lda, float* S, float* U, int32_t ldu, float* V, int32_t ldv, float* work, int32_t lwork, gesvdjInfo_t params)
{ return cusolverDnSgesvdj(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, nullptr, params); }
inline cusolverStatus_t Xgesvj(cusolverDnHandle_t handle, int32_t m, int32_t n, cuDoubleComplex* A, int32_t lda, double* S, cuDoubleComplex* U, int32_t ldu, cuDoubleComplex* V, int32_t ldv, cuDoubleComplex* work, int32_t lwork, gesvdjInfo_t params)
{ return cusolverDnZgesvdj(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, nullptr, params); }
inline cusolverStatus_t Xgesvj(cusolverDnHandle_t handle, int32_t m, int32_t n, cuComplex* A, int32_t lda, float* S, cuComplex* U, int32_t ldu, cuComplex* V, int32_t ldv, cuComplex* work, int32_t lwork, gesvdjInfo_t params)
{ return cusolverDnCgesvdj(handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, nullptr, params); }

template <class matrix_t> inline cusolverStatus_t Xgesvd_bufferSize(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork);
template <> inline cusolverStatus_t Xgesvd_bufferSize<double>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork) { return cusolverDnDgesvd_bufferSize(handle, m, n, lwork); }
template <> inline cusolverStatus_t Xgesvd_bufferSize<float>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork)  { return cusolverDnSgesvd_bufferSize(handle, m, n, lwork); }
template <> inline cusolverStatus_t Xgesvd_bufferSize<cuDoubleComplex>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork)  { return cusolverDnZgesvd_bufferSize(handle, m, n, lwork); }
template <> inline cusolverStatus_t Xgesvd_bufferSize<cuComplex>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork)  { return cusolverDnCgesvd_bufferSize(handle, m, n, lwork); }

inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, double* A, int32_t lda, double* S, double* U, int32_t ldu, double* work, int32_t lwork, double* rwork)
{ return cusolverDnDgesvd(handle, 'S', 'N', m, n, A, lda, S, U, ldu, nullptr, n, work, lwork, rwork, nullptr); }
inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, float* A, int32_t lda, float* S, float* U, int32_t ldu, float* work, int32_t lwork, float* rwork)
{ return cusolverDnSgesvd(handle, 'S', 'N', m, n, A, lda, S, U, ldu, nullptr, n, work, lwork, rwork, nullptr); }
inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, cuDoubleComplex* A, int32_t lda, double* S, cuDoubleComplex* U, int32_t ldu, cuDoubleComplex* work, int32_t lwork, double* rwork)
{ return cusolverDnZgesvd(handle, 'S', 'N', m, n, A, lda, S, U, ldu, nullptr, n, work, lwork, rwork, nullptr); }
inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, cuComplex* A, int32_t lda, float* S, cuComplex* U, int32_t ldu, cuComplex* work, int32_t lwork, float* rwork)
{ return cusolverDnCgesvd(handle, 'S', 'N', m, n, A, lda, S, U, ldu, nullptr, n, work, lwork, rwork, nullptr); }

template <int32_t BLOCK_THREADS, class real_t, class Stype>
__global__ void find_srank_kernel(double epi_sq, int32_t K, const real_t* __restrict__ sigma, Stype* __restrict__ sigma_s, int32_t* __restrict__ rank, double* __restrict__ dnrm) {
  __shared__ typename cub::BlockScan<double, BLOCK_THREADS>::TempStorage temp_scan;
  constexpr int32_t BLOCK_THREADS_M1 = BLOCK_THREADS - 1;
  double sum = 0.; int32_t Kp = (K + BLOCK_THREADS_M1) & (~BLOCK_THREADS_M1);
  for (int32_t i = int32_t(threadIdx.x); i < Kp; i += BLOCK_THREADS) {
    int32_t rev_i = K - i - 1; double x = i < K ? double(sigma[rev_i]) : 0., thread_data = fma(x, x, threadIdx.x == 0 ? sum : 0.);
    cub::BlockScan<double, BLOCK_THREADS>(temp_scan).InclusiveSum(thread_data, thread_data, sum);
    if (i < K) { dnrm[rev_i] = thread_data; }
    cooperative_groups::this_thread_block().sync();
  }

  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  int32_t thread_cnt = 0; sum *= epi_sq; bool epi_zero = epi_sq == 0.;
  for (int32_t i = int32_t(threadIdx.x); i < K; i += BLOCK_THREADS) {
    bool pred = epi_zero || sum < dnrm[i]; thread_cnt += int32_t(pred);
    if (pred) {
      if constexpr(std::is_same_v<Stype, real_t>) { sigma_s[i] = sigma[i]; } else
      if constexpr(std::is_same_v<Stype, __half> && std::is_same_v<real_t, double>) { sigma_s[i] = __double2half(sigma[i]); } else
      if constexpr(std::is_same_v<Stype, __half> && std::is_same_v<real_t, float>) { sigma_s[i] = __float2half(sigma[i]); } else
      { sigma_s[i] = Stype(sigma[i]); }
    }
  }
  thread_cnt = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Sum(thread_cnt);
  if (threadIdx.x == 0) { *rank = thread_cnt; }
}

template <class Rtype, class Xtype, class Stype, class Gtype>
inline int32_t tsvd(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char fillmode, double epi, int32_t sweeps, int32_t N, int32_t K, Xtype* X, int32_t ldx, Stype* S, Gtype* G, int32_t ldg, int32_t* r_ptr) {
  using matrix_t = typename std::conditional_t<std::is_same_v<Xtype, cuDoubleComplex> || std::is_same_v<Xtype, cuComplex> || std::is_same_v<Xtype, __half2>,
    std::conditional_t<std::is_same_v<Xtype, cuDoubleComplex>, cuDoubleComplex, cuComplex>, std::conditional_t<std::is_same_v<Xtype, double>, double, float>>;
  using real_t = typename std::conditional_t<std::is_same_v<matrix_t, cuDoubleComplex>, double, std::conditional_t<std::is_same_v<matrix_t, cuComplex>, float, matrix_t>>;

  int32_t* jpiv = nullptr; Rtype* potrf_work = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&jpiv, uint64_t(N) * uint64_t(sizeof(int32_t)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&potrf_work, uint64_t(98304) + (uint64_t(N) + uint64_t(1)) * uint64_t(sizeof(Rtype)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");

  int32_t rank = internal::Cholesky::potrfp(stream, fillmode, epi, K, N, G, ldg, jpiv, potrf_work, r_ptr);
  cudaFreeAsync(potrf_work, stream);
  if (0 < rank) {
    matrix_t* A = nullptr, *U = nullptr; real_t* sigma = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&A, uint64_t(N) * uint64_t(rank) * uint64_t(sizeof(matrix_t)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");

    internal::scatter_matcopy(stream, cublasH, 'U', rank, N, G, ldg, A, N, jpiv); cudaFreeAsync(jpiv, stream); 
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&sigma, uint64_t(rank) * uint64_t(sizeof(real_t)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&U, uint64_t(N) * uint64_t(rank) * uint64_t(sizeof(matrix_t)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");

    if (0 < sweeps) {
      matrix_t* V = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&V, uint64_t(rank) * uint64_t(rank) * uint64_t(sizeof(matrix_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVDJ.");

      gesvdjInfo_t gesvdj_params = nullptr; cusolverDnCreateGesvdjInfo(&gesvdj_params);
      cusolverDnXgesvdjSetTolerance(gesvdj_params, std::max(double(std::numeric_limits<real_t>::epsilon()), epi));
      cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, sweeps);
      int32_t lwork = 0; Xgesvj_bufferSize(cusolverH, N, rank, A, N, sigma, U, N, V, rank, &lwork, gesvdj_params);
      matrix_t* workspaceOnDevice = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&workspaceOnDevice, uint64_t(lwork) * uint64_t(sizeof(matrix_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVDJ.");

      Xgesvj(cusolverH, N, rank, A, N, sigma, U, N, V, rank, workspaceOnDevice, lwork, gesvdj_params);
      cudaFreeAsync(V, stream); cudaFreeAsync(workspaceOnDevice, stream); cusolverDnDestroyGesvdjInfo(gesvdj_params);
    } else {
      int32_t lwork = 0; Xgesvd_bufferSize<matrix_t>(cusolverH, N, rank, &lwork);
      matrix_t* workspaceOnDevice = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&workspaceOnDevice, uint64_t(lwork) * uint64_t(sizeof(matrix_t)) + uint64_t(rank) * uint64_t(sizeof(real_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVD.");

      Xgesvd(cusolverH, N, rank, A, N, sigma, U, N, workspaceOnDevice, lwork, (real_t*)(&workspaceOnDevice[lwork]));
      cudaFreeAsync(workspaceOnDevice, stream);
    }

    double* dnrm = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&dnrm, uint64_t(rank) * uint64_t(sizeof(double)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");
    cudaFreeAsync(A, stream); find_srank_kernel<512> <<< 1, 512, 0, stream >>> (std::pow(epi, 2), rank, sigma, S, r_ptr, dnrm); cudaStreamSynchronize(stream);
    rank = std::min(rank, *r_ptr); cudaFreeAsync(sigma, stream); cudaFreeAsync(dnrm, stream);
    internal::scatter_matcopy(stream, cublasH, 'A', N, rank, U, N, X, ldx); cudaFreeAsync(U, stream); 
  } else { cudaFreeAsync(jpiv, stream); }
  return rank;
}

extern "C" int32_t hyacinXGevd(const hyacinHandle_t* handle, char fillmode, double epi, int32_t N, int32_t K, hyacinPrecision_t Atype, void* X, int32_t ldx, void* S, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0) { return 0; } K = K <= 0 ? N : std::min(N, K); epi = std::min(1., std::abs(epi));
  cudaStream_t stream = handle->cudaStream; Timer::register_replicate_kernel(stream, handle->timer);
  cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; cusolverDnHandle_t cusolverH = handle->cusolverHandle;
  int32_t *r_ptr = (int32_t*)handle->pinnedWorkspace, sweeps = handle->JacobiSVDSweeps; *r_ptr = handle->DeviceSMs;
  switch (Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return tsvd<double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (double*)X, ldx, (double*)S, (double*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (float*)X, ldx, (float*)S, (double*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return tsvd<double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half*)X, ldx, (__half*)S, (double*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F64)
    { return tsvd<float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (double*)X, ldx, (double*)S, (float*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (float*)X, ldx, (float*)S, (float*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return tsvd<float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half*)X, ldx, (__half*)S, (float*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return tsvd<double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (double*)X, ldx, (double*)S, (double2*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (float*)X, ldx, (float*)S, (double2*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return tsvd<double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half*)X, ldx, (__half*)S, (double2*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_TF: if (Atype == HYACIN_F64)
    { return tsvd<float3>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (double*)X, ldx, (double*)S, (float3*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<float3>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (float*)X, ldx, (float*)S, (float3*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return tsvd<float3>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half*)X, ldx, (__half*)S, (float3*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return tsvd<float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (double*)X, ldx, (double*)S, (float4*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (float*)X, ldx, (float*)S, (float4*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return tsvd<float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half*)X, ldx, (__half*)S, (float4*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuDoubleComplex*)X, ldx, (double*)S, (cuDoubleComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuComplex*)X, ldx, (float*)S, (cuDoubleComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half2*)X, ldx, (__half*)S, (cuDoubleComplex*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuDoubleComplex*)X, ldx, (double*)S, (cuComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuComplex*)X, ldx, (float*)S, (cuComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half2*)X, ldx, (__half*)S, (cuComplex*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuDoubleComplex*)X, ldx, (double*)S, (complex_double2*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuComplex*)X, ldx, (float*)S, (complex_double2*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half2*)X, ldx, (__half*)S, (complex_double2*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_TF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<float3>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuDoubleComplex*)X, ldx, (double*)S, (complex_float3*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float3>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuComplex*)X, ldx, (float*)S, (complex_float3*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<float3>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half2*)X, ldx, (__half*)S, (complex_float3*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuDoubleComplex*)X, ldx, (double*)S, (complex_float4*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (cuComplex*)X, ldx, (float*)S, (complex_float4*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, (__half2*)X, ldx, (__half*)S, (complex_float4*)G, ldg, r_ptr); } else { return 0; }
    default: return 0;
  }
}
