
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
  else if constexpr(std::is_same_v<T, __half2> && std::is_same_v<S, cuComplex>) { b.x = __float2half(a.x); b.y = __float2half(a.y); return a; }
  else if constexpr(std::is_same_v<T, cuComplex> && std::is_same_v<S, cuDoubleComplex>) { b.x = float(a.x); b.y = float(a.y); return a; }
  else { b = T(a); return a; }
}

template <int32_t EVD, int32_t BLOCK_THREADS, class real_t, class Stype>
__global__ void find_srank_kernel(double epi, int32_t N_minus_one, const real_t* __restrict__ X, Stype* __restrict__ reX, int32_t* __restrict__ rank) {
  int32_t thread_x = 0; double s0;
  if (0 <= N_minus_one) { if constexpr(EVD) s0 = epi * double(sqrt_relu(X[N_minus_one])); else s0 = epi * double(X[0]); }
    else { s0 = 0.; }
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;

  for (int32_t i = int32_t(threadIdx.x); i <= N_minus_one; i += BLOCK_THREADS)
    if constexpr(EVD) { thread_x += int32_t(s0 <= double(reX[N_minus_one - i] = sqrt_relu(X[i]))); }
      else { thread_x += int32_t(s0 <= double(conv(X[i], reX[i]))); }

  thread_x = cub::BlockReduce<int32_t, BLOCK_THREADS>(temp_reduce).Sum(thread_x);
  if (threadIdx.x == 0)
    *rank = thread_x;
}

template <class complex_t, class Xtype>
__global__ void evd_reorder_kernel(int64_t M, int64_t N_minus_one, const complex_t* __restrict__ A, int64_t lda, Xtype* __restrict__ X, int64_t ldx) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (y < M) {
    int64_t x = int64_t(blockIdx.y), nx = N_minus_one - x;
    conv(A[y + nx * lda], X[y + x * ldx]);
  }
};

template <class T> inline cudaDataType_t cuda_type();
template <> inline cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }
template <> inline cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <> inline cudaDataType_t cuda_type<cuDoubleComplex>() { return CUDA_C_64F; }
template <> inline cudaDataType_t cuda_type<cuComplex>() { return CUDA_C_32F; }

template <class real_t, class complex_t, class Stype, class Xtype>
inline int32_t tevd(cudaStream_t stream, cusolverDnHandle_t handle, cusolverDnParams_t params, char fillmode, double epi, int32_t N, int32_t K, int32_t p, complex_t* G, int32_t ldg, Xtype* X, int32_t ldx, Stype* S) {
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  K = std::min(K, N); uint64_t s_bytes = std::max(uint64_t(sizeof(int32_t)), uint64_t(sizeof(real_t)) * uint64_t(N));
  cublasFillMode_t fill = (fillmode == 'U' || fillmode == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, fill, N, type_c, G, ldg, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  std::vector<uint8_t> workspaceOnHost(workspaceInBytesOnHost);
  uint8_t* workspaceOnDevice = nullptr, *workspaceOnHostPtr = workspaceOnHost.empty() ? nullptr : workspaceOnHost.data();
  if (cudaSuccess != cudaMallocAsync((void**)&workspaceOnDevice, workspaceInBytesOnDevice + s_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at SYEVD.");

  real_t* dev_S = (real_t*)(&workspaceOnDevice[workspaceInBytesOnDevice]);
  cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, fill, N, type_c, G, ldg, type_r, dev_S, type_c, workspaceOnDevice, workspaceInBytesOnDevice, workspaceOnHostPtr, workspaceInBytesOnHost, nullptr);
  find_srank_kernel<1, 512> <<< 1, 512, 0, stream >>> (epi, K - 1, &dev_S[N - K], S, (int32_t*)workspaceOnDevice);
  int32_t rank = 0; cudaMemcpyAsync(&rank, workspaceOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(workspaceOnDevice, stream);
  K = std::min(K, p + rank);

  evd_reorder_kernel<complex_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(K)), 512, 0, stream >>> (int64_t(N), int64_t(N - 1), G, int64_t(ldg), X, int64_t(ldx));
  return K;
}

template <class real_t, class complex_t, class GRtype, class Xtype, class Stype, class Gtype>
inline int32_t tsvd(cudaStream_t stream, cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char fillmode, double epi, int32_t N, int32_t K, int32_t p, Xtype* X, int32_t ldx, Stype* S, Gtype* G, int32_t ldg, void* pinned_work) {
  uint64_t piv_bytes = uint64_t(N) * uint64_t(sizeof(int32_t));
  uint64_t matrix_bytes = uint64_t(std::max(int64_t(sizeof(complex_t)) * int64_t(N) * int64_t(std::min(N, K)), int64_t(8192) + int64_t(sizeof(GRtype)) * int64_t(N)));
  uint8_t* dev_work = nullptr; 
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, matrix_bytes + piv_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");

  int32_t* piv = (int32_t*)(&dev_work[matrix_bytes]);
  K = internal::Cholesky::potrfp(stream, handle, fillmode, epi, K, p, N, G, ldg, piv, (GRtype*)dev_work, pinned_work);
  if (0 < K) {
    complex_t* W = (complex_t*)dev_work;
    internal::Cholesky::scatter_matcopy(stream, handle, 'U', K, N, piv, G, ldg, W, N);

    cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    cusolverDnXgesvd_bufferSize(s_handle, params, 'O', 'N', N, K, type_c, W, N, type_r, nullptr, type_c, W, N, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    std::vector<uint8_t> workspaceOnHost(workspaceInBytesOnHost);
    uint64_t s_bytes = std::max(uint64_t(sizeof(int32_t)), uint64_t(sizeof(real_t)) * uint64_t(K));
    uint8_t* workspaceOnDevice = nullptr, *workspaceOnHostPtr = workspaceOnHost.empty() ? nullptr : workspaceOnHost.data();
    if (cudaSuccess != cudaMallocAsync((void**)&workspaceOnDevice, workspaceInBytesOnDevice + s_bytes, stream))
      throw std::runtime_error("Workspace allocation failed at GESVD.");

    real_t* dev_S = (real_t*)(&workspaceOnDevice[workspaceInBytesOnDevice]);
    cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K, type_c, W, N, type_r, dev_S, type_c, W, N, type_c, nullptr, K, type_c, workspaceOnDevice, workspaceInBytesOnDevice, workspaceOnHostPtr, workspaceInBytesOnHost, nullptr);
    find_srank_kernel<0, 512> <<< 1, 512, 0, stream >>> (epi, K - 1, dev_S, S, (int32_t*)workspaceOnDevice);
    int32_t rank = 0; cudaMemcpyAsync(&rank, workspaceOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(workspaceOnDevice, stream);
    K = std::min(K, p + rank);

    internal::Cholesky::scatter_matcopy(stream, handle, 'A', N, K, nullptr, W, N, X, ldx);
  }
  cudaFreeAsync(dev_work, stream); return K;
}

extern "C" int32_t hyacinXGevPcsvd(hyacinHandle_t handle, char use_evd, char fillmode, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t Atype, void* X, int32_t ldx, void* S, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0 || K <= 0) { return 0; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  if ((use_evd == 'Y' || use_evd == 'y') && (Gtype == HYACIN_F64 || Gtype == HYACIN_F32 || Gtype == HYACIN_F64_COMPLEX || Gtype == HYACIN_F32_COMPLEX)) switch (Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return tevd<double>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)G, ldg, (double*)X, ldx, (double*)S); } else
    if (Atype == HYACIN_F32)
    { return tevd<double>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)G, ldg, (float*)X, ldx, (float*)S); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F32)
    { return tevd<float>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)G, ldg, (float*)X, ldx, (float*)S); } else
    if (Atype == HYACIN_F16)
    { return tevd<float>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)G, ldg, (__half*)X, ldx, (__half*)S); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tevd<double>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuDoubleComplex*)G, ldg, (cuDoubleComplex*)X, ldx, (double*)S); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tevd<double>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuDoubleComplex*)G, ldg, (cuComplex*)X, ldx, (float*)S); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F32_COMPLEX)
    { return tevd<float>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuComplex*)G, ldg, (cuComplex*)X, ldx, (float*)S); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tevd<float>(handle.cudaStream, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuComplex*)G, ldg, (__half2*)X, ldx, (__half*)S); } else { return 0; }
    default: return 0;
  }
  else switch(Gtype) {
    case HYACIN_F64: if (Atype == HYACIN_F64)
    { return tsvd<double, double, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)X, ldx, (double*)S, (double*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F32)
    { return tsvd<float, float, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)X, ldx, (float*)S, (double*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F32: if (Atype == HYACIN_F32)
    { return tsvd<float, float, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (float*)X, ldx, (float*)S, (float*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F16) 
    { return tsvd<float, float, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (__half*)X, ldx, (__half*)S, (float*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_DD: if (Atype == HYACIN_F64)
    { return tsvd<double, double, double2>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)X, ldx, (double*)S, (double2*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_QF: if (Atype == HYACIN_F64)
    { return tsvd<double, double, float4>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (double*)X, ldx, (double*)S, (float4*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F64_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double, cuDoubleComplex, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuDoubleComplex*)X, ldx, (double*)S, (cuDoubleComplex*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float, cuComplex, double>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuComplex*)X, ldx, (float*)S, (cuDoubleComplex*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_F32_COMPLEX: if (Atype == HYACIN_F32_COMPLEX)
    { return tsvd<float, cuComplex, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuComplex*)X, ldx, (float*)S, (cuComplex*)G, ldg, handle.pinnedWorkspace); } else
    if (Atype == HYACIN_F16_COMPLEX)
    { return tsvd<float, cuComplex, float>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (__half2*)X, ldx, (__half*)S, (cuComplex*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_DD_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double, cuDoubleComplex, double2>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuDoubleComplex*)X, ldx, (double*)S, (complex_double2*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    case HYACIN_QF_COMPLEX: if (Atype == HYACIN_F64_COMPLEX)
    { return tsvd<double, cuDoubleComplex, float4>(handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, (cuDoubleComplex*)X, ldx, (double*)S, (complex_float4*)G, ldg, handle.pinnedWorkspace); } else { return 0; }
    default: return 0;
  }
}
