
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cub/cub.cuh>
#include <cuComplex.h>
#include <algorithm>
#include <vector>
#include <stdexcept>

template<class TGT, class SRC> __device__ __forceinline__ TGT conv(SRC a) { return TGT(a); }
template <> __device__ __forceinline__ double conv<double, double2>(double2 a) { return device::dd::dd2double(a); }
template <> __device__ __forceinline__ double conv<double, float4>(float4 a) { return device::qf::qf2double(a); }
template <> __device__ __forceinline__ float conv<float, double2>(double2 a) { return float(a.x); }
template <> __device__ __forceinline__ float conv<float, float4>(float4 a) { return a.x; }

template <int32_t complex, class Atype, class Btype> 
__global__ void scatter_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred; if constexpr(complex) pred = (((x << 1) | int64_t(1)) < y); else pred = (x < y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (pred) *B = Btype(); else *B = conv<Btype, Atype>(*A);
  }
};

__device__ __forceinline__ double float_relu_sqrt(double a) { return sqrt(fmax(a, 0.)); };
__device__ __forceinline__ float float_relu_sqrt(float a) { return sqrtf(fmaxf(a, 0.f)); };

template <int32_t EVD, int32_t BLOCK_THREADS, class real_t>
__global__ void find_srank_kernel(double epi, int32_t N, const real_t* __restrict__ X, real_t* __restrict__ reX, int32_t* __restrict__ rank) {
  int32_t thread_x = 0, N_minus_one = N - 1; double s0;
  if (0 < N) { if constexpr(EVD) s0 = epi * double(float_relu_sqrt(X[N_minus_one])); else s0 = epi * double(X[0]); }
    else s0 = 0.;
  __shared__ typename cub::BlockReduce<int32_t, BLOCK_THREADS>::TempStorage temp_reduce;

  for (int32_t i = int32_t(threadIdx.x); i < N; i += BLOCK_THREADS)
    if constexpr(EVD) { thread_x += int32_t(s0 <= double(reX[N_minus_one - i] = float_relu_sqrt(X[i]))); }
      else { thread_x += int32_t(s0 <= double(X[i])); }

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
  K = std::min(K, N); uint64_t s_bytes = std::max(uint64_t(sizeof(int32_t)), uint64_t(sizeof(real_t)) * K);
  cublasFillMode_t fill = (fillmode == 'U' || fillmode == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXsyevd_bufferSize(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, fill, N, type_c, G, ldg, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  std::vector<uint8_t> workspaceOnHost(workspaceInBytesOnHost);
  uint8_t* workspaceOnDevice = nullptr, *workspaceOnHostPtr = workspaceOnHost.empty() ? nullptr : workspaceOnHost.data();
  if (cudaSuccess != cudaMallocAsync((void**)&workspaceOnDevice, workspaceInBytesOnDevice + s_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at SYEVD.");

  real_t* dev_S = (real_t*)(&((int8_t*)workspaceOnDevice)[workspaceInBytesOnDevice]);
  cusolverDnXsyevd(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, fill, N, type_c, G, ldg, type_r, dev_S, type_c, workspaceOnDevice, workspaceInBytesOnDevice, workspaceOnHostPtr, workspaceInBytesOnHost, nullptr);

  find_srank_kernel<1, 512> <<< 1, 512, 0, stream >>> (epi, K, &dev_S[N - K], S, (int32_t*)workspaceOnDevice);
  int32_t rank = 0; cudaMemcpyAsync(&rank, workspaceOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(workspaceOnDevice, stream);
  K = std::min(K, p + rank);

  evd_reorder_kernel<complex_t> <<< dim3(uint32_t(N + 511) >> 9, std::min(uint32_t(K), uint32_t(N) >> 1)), 512, 0, stream >>> (int64_t(N), int64_t(N - 1) * int64_t(ldg), G, int64_t(ldg));
  return K;
}

template <class real_t, class complex_t>
inline int32_t ge_tsvd(cudaStream_t stream, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p, complex_t* X, int32_t ldx, real_t* S) {
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvd_bufferSize(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, S, type_c, X, ldx, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  std::vector<uint8_t> workspaceOnHost(workspaceInBytesOnHost);
  uint8_t* workspaceOnDevice = nullptr, *workspaceOnHostPtr = workspaceOnHost.empty() ? nullptr : workspaceOnHost.data();
  if (cudaSuccess != cudaMallocAsync((void**)&workspaceOnDevice, std::max(sizeof(int32_t), workspaceInBytesOnDevice), stream))
    throw std::runtime_error("Workspace allocation failed at GESVD.");

  cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, S, type_c, X, ldx, type_c, nullptr, K, type_c, workspaceOnDevice, workspaceInBytesOnDevice, workspaceOnHostPtr, workspaceInBytesOnHost, nullptr);
  find_srank_kernel<0, 512> <<< 1, 512, 0, stream >>> (epi, K, S, (real_t*)nullptr, (int32_t*)workspaceOnDevice);
  int32_t rank = 0; cudaMemcpyAsync(&rank, workspaceOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(workspaceOnDevice, stream);
  return std::min(K, p + rank);
}

inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f64(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f32(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f128_dd(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f128_qf(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf64(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf32(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf128_dd(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }
inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf128_qf(stream, handle, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

template <class Rtype, class Gtype>
inline int32_t gsvd_dispatcher(cudaStream_t stream, cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char fillmode,
  double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t Atype, void* S, Gtype* G, int32_t ldg, void* pinned_work) {
  int32_t x_bytes, g_real_bytes = int32_t(sizeof(Rtype)); hyacinXelem('A', Atype, nullptr, &x_bytes, nullptr);
  uint64_t piv_bytes = uint64_t(N) * uint64_t(sizeof(int32_t));
  uint64_t matrix_bytes = uint64_t(std::max(int64_t(x_bytes) * int64_t(N) * int64_t(std::min(N, K)), int64_t(8192) + int64_t(g_real_bytes) * int64_t(N)));
  uint8_t* dev_work = nullptr; 
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, matrix_bytes + piv_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at GESVD Preconditioning.");

  int32_t* piv = (int32_t*)(&dev_work[matrix_bytes]);
  K = potrfp(stream, handle, fillmode, epi, K, p, N, G, ldg, piv, dev_work, pinned_work);
  if (0 < K) {
    constexpr int32_t block_threads = 512;
    int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
    int32_t ldx = int64_t(ldg) * int64_t(sizeof(Gtype)) / int64_t(x_bytes);
    uint32_t grid_x = uint32_t(K + 511) >> 9, grid_cx = uint32_t((uint64_t(CK) + uint64_t(511)) >> 9);
    if (Atype == HYACIN_F64) {
      double* Xptr = (double*)G, *Wptr = (double*)dev_work;
      scatter_cvcpy_kernel<0, Rtype, double> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), piv, (const Rtype*)G, int64_t(ldg), Wptr, int64_t(K));
      double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      cudaFreeAsync(dev_work, stream);
      return ge_tsvd(stream, s_handle, params, epi, N, K, p, Xptr, ldx, (double*)S);
    }
    else if (Atype == HYACIN_F32 || Atype == HYACIN_F16) {
      float* Xptr = (float*)G, *Wptr = (float*)dev_work;
      scatter_cvcpy_kernel<0, Rtype, float> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), piv, (const Rtype*)G, int64_t(ldg), Wptr, int64_t(K));
      float one = 1., zero = 0.; cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      cudaFreeAsync(dev_work, stream);
      return ge_tsvd(stream, s_handle, params, epi, N, K, p, Xptr, ldx, (float*)S);
    }
    else if (Atype == HYACIN_F64_COMPLEX) {
      cuDoubleComplex* Xptr = (cuDoubleComplex*)G, *Wptr = (cuDoubleComplex*)dev_work;
      scatter_cvcpy_kernel<1, Rtype, double> <<< dim3(grid_cx, N), block_threads, 0, stream >>> (CK, piv, (const Rtype*)G, cldg, (double*)Wptr, CK);
      cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
      cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      cudaFreeAsync(dev_work, stream);
      return ge_tsvd(stream, s_handle, params, epi, N, K, p, Xptr, ldx, (double*)S);
    }
    else if (Atype == HYACIN_F32_COMPLEX || Atype == HYACIN_F16_COMPLEX) {
      cuComplex* Xptr = (cuComplex*)G, *Wptr = (cuComplex*)dev_work;
      scatter_cvcpy_kernel<1, Rtype, float> <<< dim3(grid_cx, N), block_threads, 0, stream >>> (CK, piv, (const Rtype*)G, cldg, (float*)Wptr, CK);
      cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
      cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      cudaFreeAsync(dev_work, stream);
      return ge_tsvd(stream, s_handle, params, epi, N, K, p, Xptr, ldx, (float*)S);
    }
  }
  cudaFreeAsync(dev_work, stream);
  return K;
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
    case HYACIN_F64: return gsvd_dispatcher<double>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (double*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_F32: return gsvd_dispatcher<float>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (float*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_DD: return gsvd_dispatcher<double2>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (double2*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_QF: return gsvd_dispatcher<float4>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (float4*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_F64_COMPLEX: return gsvd_dispatcher<double>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (std::complex<double>*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_F32_COMPLEX: return gsvd_dispatcher<float>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (std::complex<float>*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_DD_COMPLEX: return gsvd_dispatcher<double2>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (complex_double2*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_QF_COMPLEX: return gsvd_dispatcher<float4>
      (handle.cudaStream, handle.cublasHandle, handle.cusolverHandle, handle.cusolverParams, fillmode, epi, N, K, p, Atype, S, (complex_float4*)G, ldg, handle.pinnedWorkspace);
    default: return 0;
  }
}
