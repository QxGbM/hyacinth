
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <limits>
#include <algorithm>
#include <stdexcept>

template <class T, class S> __device__ __forceinline__ S conv(S a, T& b) {
  if constexpr(std::is_same_v<T, S>) { return b = a; }
  else if constexpr(std::is_same_v<T, __half> && std::is_same_v<S, float>) { b = __float2half(a); return a; }
  else if constexpr(std::is_same_v<T, __half2> && std::is_same_v<S, cuComplex>) { b.x = __float2half(a.x); b.y = __float2half(a.y); return a; }
  else if constexpr(std::is_same_v<T, cuComplex> && std::is_same_v<S, cuDoubleComplex>) { b.x = float(a.x); b.y = float(a.y); return a; }
  else if constexpr(std::is_same_v<T, cuDoubleComplex> && std::is_same_v<S, cuComplex>) { b.x = double(a.x); b.y = double(a.y); return a; }
  else { b = T(a); return a; }
}

template <int32_t BLOCK_THREADS, class real_t, class Stype>
__global__ void find_srank_kernel(double epi, int32_t N, const real_t* __restrict__ X, Stype* __restrict__ reX, int32_t* __restrict__ rank) {
  __shared__ double s0; __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;
  if (threadIdx.x == 0) { s0 = epi * double(X[0]); }
  cooperative_groups::this_thread_block().sync();

  int32_t thread_x = 0;
  for (int32_t i = int32_t(threadIdx.x); i < N; i += BLOCK_THREADS)
  { thread_x += int32_t(s0 <= double(conv(X[i], reX[i]))); }

  thread_x = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Sum(thread_x);
  if (threadIdx.x == 0) { *rank = thread_x; }
}

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

template <class complex_t> inline cusolverStatus_t Xgesvd_bufferSize(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork);
template <> inline cusolverStatus_t Xgesvd_bufferSize<double>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork) { return cusolverDnDgesvd_bufferSize(handle, m, n, lwork); }
template <> inline cusolverStatus_t Xgesvd_bufferSize<float>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork)  { return cusolverDnSgesvd_bufferSize(handle, m, n, lwork); }
template <> inline cusolverStatus_t Xgesvd_bufferSize<cuDoubleComplex>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork)  { return cusolverDnZgesvd_bufferSize(handle, m, n, lwork); }
template <> inline cusolverStatus_t Xgesvd_bufferSize<cuComplex>(cusolverDnHandle_t handle, int32_t m, int32_t n, int32_t* lwork)  { return cusolverDnCgesvd_bufferSize(handle, m, n, lwork); }

inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, double* A, int32_t lda, double* S, double* work, int32_t lwork, double* rwork)
{ return cusolverDnDgesvd(handle, 'O', 'N', m, n, A, lda, S, A, lda, nullptr, n, work, lwork, rwork, nullptr); }
inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, float* A, int32_t lda, float* S, float* work, int32_t lwork, float* rwork)
{ return cusolverDnSgesvd(handle, 'O', 'N', m, n, A, lda, S, A, lda, nullptr, n, work, lwork, rwork, nullptr); }
inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, cuDoubleComplex* A, int32_t lda, double* S, cuDoubleComplex* work, int32_t lwork, double* rwork)
{ return cusolverDnZgesvd(handle, 'O', 'N', m, n, A, lda, S, A, lda, nullptr, n, work, lwork, rwork, nullptr); }
inline cusolverStatus_t Xgesvd(cusolverDnHandle_t handle, int32_t m, int32_t n, cuComplex* A, int32_t lda, float* S, cuComplex* work, int32_t lwork, float* rwork)
{ return cusolverDnCgesvd(handle, 'O', 'N', m, n, A, lda, S, A, lda, nullptr, n, work, lwork, rwork, nullptr); }

template <class real_t, class complex_t, class Rtype, class Xtype, class Stype, class Gtype>
inline int32_t tsvd(cudaStream_t stream, cudaMemPool_t mempool, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char fillmode, double epi, int32_t sweeps, int32_t N, int32_t K, int32_t p, Xtype* X, int32_t ldx, Stype* S, Gtype* G, int32_t ldg, int32_t* r_ptr) {
  int32_t* piv = nullptr; Rtype* potrf_work = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&piv, uint64_t(N) * uint64_t(sizeof(int32_t)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&potrf_work, uint64_t(65536) + (uint64_t(N) + uint64_t(1)) * uint64_t(sizeof(Rtype)), mempool, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");

  K = internal::Cholesky::potrfp(stream, fillmode, epi, K, p, N, G, ldg, piv, potrf_work, r_ptr);
  cudaFreeAsync(potrf_work, stream);
  if (0 < K) {
    complex_t* A = nullptr, *U = nullptr; real_t* sigma = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&A, uint64_t(N) * uint64_t(K) * uint64_t(sizeof(complex_t)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");

    internal::scatter_matcopy(stream, cublasH, 'U', K, N, piv, G, ldg, A, N); cudaFreeAsync(piv, stream); 
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&sigma, uint64_t(K) * uint64_t(sizeof(real_t)), mempool, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");

    if (0 < sweeps) {
      complex_t* V = nullptr;
      if (cudaSuccess != cudaMallocFromPoolAsync((void**)&U, uint64_t(N) * uint64_t(K) * uint64_t(sizeof(complex_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVDJ.");
      if (cudaSuccess != cudaMallocFromPoolAsync((void**)&V, uint64_t(K) * uint64_t(K) * uint64_t(sizeof(complex_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVDJ.");

      gesvdjInfo_t gesvdj_params = nullptr; cusolverDnCreateGesvdjInfo(&gesvdj_params);
      cusolverDnXgesvdjSetTolerance(gesvdj_params, std::min(1., std::max(double(std::numeric_limits<real_t>::epsilon()), std::abs(epi))));
      cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, sweeps);
      int32_t lwork = 0; Xgesvj_bufferSize(cusolverH, N, K, A, N, sigma, U, N, V, K, &lwork, gesvdj_params);
      complex_t* workspaceOnDevice = nullptr;
      if (cudaSuccess != cudaMallocFromPoolAsync((void**)&workspaceOnDevice, uint64_t(lwork) * uint64_t(sizeof(complex_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVDJ.");

      Xgesvj(cusolverH, N, K, A, N, sigma, U, N, V, K, workspaceOnDevice, lwork, gesvdj_params);
      cudaFreeAsync(A, stream); cudaFreeAsync(V, stream); cudaFreeAsync(workspaceOnDevice, stream); cusolverDnDestroyGesvdjInfo(gesvdj_params);
    } else {
      int32_t lwork = 0; Xgesvd_bufferSize<complex_t>(cusolverH, N, K, &lwork);
      complex_t* workspaceOnDevice = nullptr;
      if (cudaSuccess != cudaMallocFromPoolAsync((void**)&workspaceOnDevice, uint64_t(lwork) * uint64_t(sizeof(complex_t)) + uint64_t(K) * uint64_t(sizeof(real_t)), mempool, stream))
        throw std::runtime_error("Workspace allocation failed at GESVD.");

      Xgesvd(cusolverH, N, K, A, N, sigma, workspaceOnDevice, lwork, (real_t*)(&workspaceOnDevice[lwork]));
      cudaFreeAsync(workspaceOnDevice, stream); U = A;
    }

    find_srank_kernel<512> <<< 1, 512, 0, stream >>> (epi, K, sigma, S, r_ptr); cudaStreamSynchronize(stream);
    K = std::min(K, *r_ptr); cudaFreeAsync(sigma, stream);
    internal::scatter_matcopy(stream, cublasH, 'A', N, K, nullptr, U, N, X, ldx); cudaFreeAsync(U, stream); 
  } else { cudaFreeAsync(piv, stream); }
  return K;
}

extern "C" int32_t hyacinXGevd(const hyacinHandle_t* handle, char fillmode, double epi, int32_t N, int32_t K, hyacinPrecision_t Atype, void* X, int32_t ldx, void* S, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0) { return 0; } K = K <= 0 ? N : std::min(N, K);
  cudaStream_t stream = handle->cudaStream; Timer::register_replicate_kernel(stream, handle->timer);
  cudaMemPool_t mempool = handle->mempool; cublasHandle_t cublasH = handle->cublasHandle; cusolverDnHandle_t cusolverH = handle->cusolverHandle;
  int32_t *r_ptr = (int32_t*)handle->pinnedWorkspace, sweeps = handle->JacobiSVDSweeps, oversampling = handle->RankOversampling; *r_ptr = handle->DeviceSMs;
  switch (Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return tsvd<double, double, double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (double*)X, ldx, (double*)S, (double*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<float, float, double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (float*)X, ldx, (float*)S, (double*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F64)
    { return tsvd<float, float, float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (double*)X, ldx, (double*)S, (float*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32)
    { return tsvd<float, float, float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (float*)X, ldx, (float*)S, (float*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16)
    { return tsvd<float, float, float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (__half*)X, ldx, (__half*)S, (float*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return tsvd<double, double, double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (double*)X, ldx, (double*)S, (double2*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return tsvd<double, double, float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (double*)X, ldx, (double*)S, (float4*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double, cuDoubleComplex, double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (cuDoubleComplex*)X, ldx, (double*)S, (cuDoubleComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float, cuComplex, double>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (cuComplex*)X, ldx, (float*)S, (cuDoubleComplex*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<float, cuComplex, float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (cuDoubleComplex*)X, ldx, (double*)S, (cuComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float, cuComplex, float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (cuComplex*)X, ldx, (float*)S, (cuComplex*)G, ldg, r_ptr); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<float, cuComplex, float>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (__half2*)X, ldx, (__half*)S, (cuComplex*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double, cuDoubleComplex, double2>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (cuDoubleComplex*)X, ldx, (double*)S, (complex_double2*)G, ldg, r_ptr); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double, cuDoubleComplex, float4>(stream, mempool, cublasH, cusolverH, fillmode, epi, sweeps, N, K, oversampling, (cuDoubleComplex*)X, ldx, (double*)S, (complex_float4*)G, ldg, r_ptr); } else { return 0; }
    default: return 0;
  }
}
