
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cub/cub.cuh>
#include <cuComplex.h>
#include <algorithm>
#include <vector>
#include <stdexcept>

__device__ __forceinline__ double sqrt_relu(double a) { return sqrt(fmax(a, 0.)); };
__device__ __forceinline__ float sqrt_relu(float a) { return sqrtf(fmaxf(a, 0.f)); };
template <class T, class S> __device__ __forceinline__ S conv(S a, T& b) {
  if constexpr(std::is_same_v<T, S>) { return b = a; }
  else if constexpr(std::is_same_v<T, __half> && std::is_same_v<S, float>) { b = __float2half(a); return a; }
  else { b = T(a); return a; }
}

template <int32_t EVD, int32_t BLOCK_THREADS, class real_t, class Stype>
__global__ void find_srank_kernel(double epi, int32_t N, const real_t* __restrict__ X, Stype* __restrict__ reX, int32_t* __restrict__ rank) {
  int32_t thread_x = 0, N_minus_one = N - 1; double s0;
  if (0 < N) { if constexpr(EVD) s0 = epi * double(sqrt_relu(X[N_minus_one])); else s0 = epi * double(X[0]); }
    else { s0 = 0.; }
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;

  for (int32_t i = int32_t(threadIdx.x); i < N; i += BLOCK_THREADS)
    if constexpr(EVD) { thread_x += int32_t(s0 <= double(reX[N_minus_one - i] = sqrt_relu(X[i]))); }
      else { thread_x += int32_t(s0 <= double(conv(X[i], reX[i]))); }

  thread_x = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Sum(thread_x);
  if (threadIdx.x == 0)
    *rank = thread_x;
}

template <class complex_t>
__global__ void evd_reorder_kernel(int64_t M, int64_t N_minus_one, complex_t* __restrict__ A, int64_t lda) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (y < M) {
    int64_t lx = int64_t(blockIdx.y) * lda, e1 = y + lx, e2 = y + N_minus_one - lx;
    complex_t e = A[e1]; A[e1] = A[e2]; A[e2] = e;
  }
};

template <class T> inline cudaDataType_t cuda_type();
template <> inline cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }
template <> inline cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <> inline cudaDataType_t cuda_type<cuDoubleComplex>() { return CUDA_C_64F; }
template <> inline cudaDataType_t cuda_type<cuComplex>() { return CUDA_C_32F; }

template <class real_t, class complex_t>
inline int32_t tevd(cudaStream_t stream, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char fillmode, double epi, int32_t N, int32_t K, int32_t p, complex_t* G, int32_t ldg, real_t* S) {
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  K = std::min(K, N); uint64_t s_bytes = std::max(uint64_t(sizeof(int32_t)), uint64_t(sizeof(real_t)) * N);
  cublasFillMode_t fill = (fillmode == 'U' || fillmode == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXsyevd_bufferSize(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, fill, N, type_c, G, ldg, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  std::vector<uint8_t> workspaceOnHost(workspaceInBytesOnHost);
  uint8_t* workspaceOnDevice = nullptr, *workspaceOnHostPtr = workspaceOnHost.empty() ? nullptr : workspaceOnHost.data();
  if (cudaSuccess != cudaMallocAsync((void**)&workspaceOnDevice, workspaceInBytesOnDevice + s_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at SYEVD.");

  real_t* dev_S = (real_t*)(&workspaceOnDevice[workspaceInBytesOnDevice]);
  cusolverDnXsyevd(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, fill, N, type_c, G, ldg, type_r, dev_S, type_c, workspaceOnDevice, workspaceInBytesOnDevice, workspaceOnHostPtr, workspaceInBytesOnHost, nullptr);
  find_srank_kernel<1, 512> <<< 1, 512, 0, stream >>> (epi, K, &dev_S[N - K], S, (int32_t*)workspaceOnDevice);
  int32_t rank = 0; cudaMemcpyAsync(&rank, workspaceOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(workspaceOnDevice, stream);
  K = std::min(K, p + rank);

  evd_reorder_kernel<complex_t> <<< dim3(uint32_t(N + 511) >> 9, std::min(uint32_t(K), uint32_t(N) >> 1)), 512, 0, stream >>> (int64_t(N), int64_t(N - 1) * int64_t(ldg), G, int64_t(ldg));
  return K;
}

template <class real_t, class complex_t, class Stype>
inline int32_t ge_tsvd(cudaStream_t stream, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p, complex_t* X, int32_t ldx, Stype* S) {
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvd_bufferSize(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, nullptr, type_c, X, ldx, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  std::vector<uint8_t> workspaceOnHost(workspaceInBytesOnHost);
  uint64_t s_bytes = std::max(uint64_t(sizeof(int32_t)), uint64_t(sizeof(real_t)) * K);
  uint8_t* workspaceOnDevice = nullptr, *workspaceOnHostPtr = workspaceOnHost.empty() ? nullptr : workspaceOnHost.data();
  if (cudaSuccess != cudaMallocAsync((void**)&workspaceOnDevice, workspaceInBytesOnDevice + s_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD.");

  real_t* dev_S = (real_t*)(&workspaceOnDevice[workspaceInBytesOnDevice]);
  cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, dev_S, type_c, X, ldx, type_c, nullptr, K, type_c, workspaceOnDevice, workspaceInBytesOnDevice, workspaceOnHostPtr, workspaceInBytesOnHost, nullptr);
  find_srank_kernel<0, 512> <<< 1, 512, 0, stream >>> (epi, K, dev_S, S, (int32_t*)workspaceOnDevice);
  int32_t rank = 0; cudaMemcpyAsync(&rank, workspaceOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(workspaceOnDevice, stream);
  return std::min(K, p + rank);
}

inline void transpose_copy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const double* A, int32_t lda, double* B, int32_t ldb)
{ double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void transpose_copy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const float* A, int32_t lda, float* B, int32_t ldb)
{ float one = 1.f, zero = 0.f;  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void transpose_copy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void transpose_copy(cublasHandle_t handle, int32_t Mb, int32_t Nb, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }

template <class Xtype, class XRtype, class GRtype, class Stype, class Gtype>
inline int32_t gsvd(cudaStream_t stream, cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char fillmode, double epi, int32_t N, int32_t K, int32_t p, Stype* S, Gtype* G, int32_t ldg, void* pinned_work) {
  uint64_t piv_bytes = uint64_t(N) * uint64_t(sizeof(int32_t));
  uint64_t matrix_bytes = uint64_t(std::max(int64_t(sizeof(Xtype)) * int64_t(N) * int64_t(std::min(N, K)), int64_t(8192) + int64_t(sizeof(GRtype)) * int64_t(N)));
  uint8_t* dev_work = nullptr; 
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, matrix_bytes + piv_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");

  int32_t* piv = (int32_t*)(&dev_work[matrix_bytes]);
  K = internal::Cholesky::potrfp(stream, handle, fillmode, epi, K, p, N, G, ldg, piv, (GRtype*)dev_work, pinned_work);
  if (0 < K) {
    int32_t ldx = int64_t(ldg) * int64_t(sizeof(Gtype)) / int64_t(sizeof(Xtype));
    Xtype* Xptr = (Xtype*)G, *Wptr = (Xtype*)dev_work;
    internal::Cholesky::scatter_matcopy(stream, 'U', K, N, piv, G, ldg, Wptr, K);
    transpose_copy(handle, N, K, Wptr, K, Xptr, ldx);
    cudaFreeAsync(dev_work, stream);
    return ge_tsvd<XRtype>(stream, s_handle, params, epi, N, K, p, Xptr, ldx, S);
  }
  else { cudaFreeAsync(dev_work, stream); return K; }
}

extern "C" int32_t hyacinXGevPcsvd(hyacinHandle_t handle, char use_evd, char fillmode, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t Atype, void* S, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0 || K <= 0) { return 0; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  if (use_evd == 'Y' || use_evd == 'y') switch (Gtype) {
    case HYACIN_F64:
      return tevd(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)G, ldg, (double*)S);
    case HYACIN_F32:
      return tevd(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)G, ldg, (float*)S);
    case HYACIN_F64_COMPLEX:
      return tevd(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuDoubleComplex*)G, ldg, (double*)S);
    case HYACIN_F32_COMPLEX:
      return tevd(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuComplex*)G, ldg, (float*)S);
    default: return 0;
  }
  else switch(Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return gsvd<double, double, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)S, (double*)G, ldg, handle.pinnedWorkspace); }
    else if (Atype == HYACIN_F32)
    { return gsvd<float, float, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)S, (double*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F32)
    { return gsvd<float, float, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)S, (float*)G, ldg, handle.pinnedWorkspace); }
    else if (Atype == HYACIN_F16) 
    { return gsvd<float, float, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (__half*)S, (float*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return gsvd<double, double, double2>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)S, (double2*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return gsvd<double, double, float4>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)S, (float4*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return gsvd<cuDoubleComplex, double, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)S, (cuDoubleComplex*)G, ldg, handle.pinnedWorkspace); }
    else if (Atype == HYACIN_F32_COMPLEX)
    { return gsvd<cuComplex, float, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)S, (cuDoubleComplex*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F32_COMPLEX)
    { return gsvd<cuComplex, float, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)S, (cuComplex*)G, ldg, handle.pinnedWorkspace); }
    else if (Atype == HYACIN_F16_COMPLEX)
    { return gsvd<cuComplex, float, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (__half*)S, (cuComplex*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return gsvd<cuDoubleComplex, double, double2>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)S, (complex_double2*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return gsvd<cuDoubleComplex, double, float4>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)S, (complex_float4*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    default: return 0;
  }
}
