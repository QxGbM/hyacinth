
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <tuple>
#include <algorithm>
#include <numeric>

__device__ __forceinline__ void conv(float a, double& b) { b = double(a); }
__device__ __forceinline__ void conv(double2 a, double& b) { b = device::dd::dd2double(a); }
__device__ __forceinline__ void conv(float4 a, double& b) { b = device::qf::qf2double(a); }

__device__ __forceinline__ void conv(double a, float& b) { b = float(a); }
__device__ __forceinline__ void conv(double2 a, float& b) { b = float(a.x); }
__device__ __forceinline__ void conv(float4 a, float& b) { b = a.x; }

template <int32_t complex, int32_t no_conv, class constAptr, class Btype, class Bptr> 
__global__ void scatter_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred; if constexpr(complex) pred = ((x << 1) < (y - int64_t(1))); else pred = (x < y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (pred) *B = Btype();
      else { if constexpr(no_conv) *B = *A; else conv(*A, *B); }
  }
};

template <class T> inline cudaDataType_t cuda_type();
template <> inline cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }
template <> inline cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <> inline cudaDataType_t cuda_type<cuDoubleComplex>() { return CUDA_C_64F; }
template <> inline cudaDataType_t cuda_type<cuComplex>() { return CUDA_C_32F; }

template <class real_t, class complex_t>
inline int32_t ge_tsvd(cudaStream_t stream, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p,
  void* X, int32_t ldx, void* S, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvd_bufferSize(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, S, type_c, X, ldx, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  if (uint64_t(workspaceInBytesOnDevice) <= dev_work_bytes && uint64_t(workspaceInBytesOnHost) <= pinned_work_bytes) {
    cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, S, type_c, X, ldx, type_c, nullptr, K, type_c, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes, nullptr);
    cudaMemcpyAsync(pinned_work, S, int64_t(N) * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    real_t *Svec = (real_t*)pinned_work;
    double s0 = epi * double(Svec[0]);
    return std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return s < s0; }))));
  }
  else throw std::runtime_error("Insufficient workspace provided for GESVD.\n");
}

template <hyacinPrecision_t precG> inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);
template<> inline int32_t potrfp<HYACIN_F64>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f64(stream, handle, epi, k, p, N, (double*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_F32>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f32(stream, handle, epi, k, p, N, (float*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_DD>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f128_dd(stream, handle, epi, k, p, N, (double2*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_QF>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f128_qf(stream, handle, epi, k, p, N, (float4*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_F64_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf64(stream, handle, epi, k, p, N, (std::complex<double>*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_F32_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf32(stream, handle, epi, k, p, N, (std::complex<float>*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_DD_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf128_dd(stream, handle, epi, k, p, N, (complex_double2*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_QF_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf128_qf(stream, handle, epi, k, p, N, (complex_float4*)A, lda, jpiv, dev_work, pinned_work); }

template <hyacinPrecision_t precG, class constGptr, class Gtype>
inline int32_t gsvd_dispatcher(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params,
  double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, void* S, void* X, int32_t ldx, Gtype* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t* hpiv = (int32_t*)(&((Gtype*)pinned_work)[512]);
  std::iota(hpiv, &hpiv[N], 1);
  K = potrfp<precG>(stream, handle, epi, K, p, N, G, ldg, hpiv, dev_work, pinned_work);
  cudaMemcpyAsync(S, hpiv, int64_t(N) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  if (0 < K) {
    constexpr int32_t block_threads = 512;
    if (AXtype == HYACIN_F64) {
      double* Xptr = (double*)X, *Wptr = (double*)dev_work;
      dim3 grid(uint32_t(K + block_threads - 1) >> 9, uint32_t(N), 1);
      if constexpr (precG == HYACIN_F64) scatter_cvcpy_kernel <0, 1, constGptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(K), (const int32_t*)S, G, int64_t(ldg), Wptr, int64_t(K));
        else scatter_cvcpy_kernel <0, 0, constGptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(K), (const int32_t*)S, G, int64_t(ldg), Wptr, int64_t(K));
      double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<double, double>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
    else if (AXtype == HYACIN_F32) {
      float* Xptr = (float*)X, *Wptr = (float*)dev_work;
      dim3 grid(uint32_t(K + block_threads - 1) >> 9, uint32_t(N), 1);
      if constexpr (precG == HYACIN_F32) scatter_cvcpy_kernel <0, 1, constGptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(K), (const int32_t*)S, G, int64_t(ldg), Wptr, int64_t(K));
        else scatter_cvcpy_kernel <0, 0, constGptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(K), (const int32_t*)S, G, int64_t(ldg), Wptr, int64_t(K));
      float one = 1., zero = 0.; cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<float, float>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
    else if (AXtype == HYACIN_F64_COMPLEX) {
      cuDoubleComplex* Xptr = (cuDoubleComplex*)X, *Wptr = (cuDoubleComplex*)dev_work;
      int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
      dim3 grid(uint32_t(CK + int64_t(block_threads - 1)) >> 9, uint32_t(N), 1);
      if constexpr (precG == HYACIN_F64_COMPLEX) scatter_cvcpy_kernel <1, 1, constGptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (CK, (const int32_t*)S, G, cldg, (double*)Wptr, CK);
        else scatter_cvcpy_kernel <1, 0, constGptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (CK, (const int32_t*)S, G, cldg, (double*)Wptr, CK);
      cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
      cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<double, cuDoubleComplex>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
    else if (AXtype == HYACIN_F32_COMPLEX) {
      cuComplex* Xptr = (cuComplex*)X, *Wptr = (cuComplex*)dev_work;
      int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
      dim3 grid(uint32_t(CK + int64_t(block_threads - 1)) >> 9, uint32_t(N), 1);
      if constexpr (precG == HYACIN_F32_COMPLEX) scatter_cvcpy_kernel <1, 1, constGptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (CK, (const int32_t*)S, G, cldg, (float*)Wptr, CK);
        else scatter_cvcpy_kernel <1, 0, constGptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (CK, (const int32_t*)S, G, cldg, (float*)Wptr, CK);
      cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
      cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<float, cuComplex>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
  }
  return K;
}

void hyacinXGsvd_bufferSize(int32_t N, int32_t K, hyacinPrecision_t AXtype, hyacinPrecision_t Gtype, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes);

int32_t hyacinXGsvd(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p,
  hyacinPrecision_t AXtype, void* S, void* X, int32_t ldx, hyacinPrecision_t Gtype, void* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  
  if (0 < N && 0 < K) switch(Gtype) {
    case HYACIN_F64:
      return gsvd_dispatcher<HYACIN_F64, const double* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (double*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32:
      return gsvd_dispatcher<HYACIN_F32, const float* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (float*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_DD:
      return gsvd_dispatcher<HYACIN_DD, const double2* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (double2*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_QF:
      return gsvd_dispatcher<HYACIN_QF, const float4* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (float4*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F64_COMPLEX:
      return gsvd_dispatcher<HYACIN_F64_COMPLEX, const double* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (double*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32_COMPLEX:
      return gsvd_dispatcher<HYACIN_F32_COMPLEX, const float* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (float*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_DD_COMPLEX:
      return gsvd_dispatcher<HYACIN_DD_COMPLEX, const double2* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (double2*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_QF_COMPLEX:
      return gsvd_dispatcher<HYACIN_QF_COMPLEX, const float4* __restrict__>(handle, s_handle, params, epi, N, K, p, AXtype, S, X, ldx, (float4*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    default: return 0;
  }
  return 0;
}

extern "C" void hyacinXsvdk_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, hyacinPrecision_t ComputeType, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  if (N <= 0 || K <= 0) { *dev_work_bytes = *pinned_work_bytes = uint64_t(0); return; }

  cudaDataType_t type_c = cudaDataType_t(), type_r = cudaDataType_t();
  uint64_t size_c = uint64_t(0), size_r = uint64_t(0);
  switch(ComputeType) {
    case HYACIN_F64: { type_c = type_r = cuda_type<double>(); size_c = size_r = sizeof(double); break; }
    case HYACIN_F32: { type_c = type_r = cuda_type<float>(); size_c = size_r = sizeof(float); break; }
    case HYACIN_F64_COMPLEX: { type_c = cuda_type<cuDoubleComplex>(); type_r = cuda_type<double>(); size_c = sizeof(cuDoubleComplex); size_r = sizeof(double); break; }
    case HYACIN_F32_COMPLEX: { type_c = cuda_type<cuComplex>(); type_r = cuda_type<float>(); size_c = sizeof(cuComplex); size_r = sizeof(float); break; }
    default: break;
  }

  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost; uint64_t basis_bytes;
  cusolverDnXgesvd_bufferSize(handle, params, 'O', 'N', N, K,
    type_c, nullptr, algnN, type_r, nullptr, type_c, nullptr, algnN, type_c, nullptr, algnK, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  basis_bytes = size_c * uint64_t(algnN) * uint64_t(K);

  uint64_t s_bytes = size_r * uint64_t(algnK);
  uint64_t trans_bytes = size_c * uint64_t(algnN + 16384) * uint64_t(K);
  *dev_work_bytes = std::max(uint64_t(workspaceInBytesOnDevice) + basis_bytes + s_bytes, trans_bytes);
  *pinned_work_bytes = uint64_t(workspaceInBytesOnHost) + s_bytes;
}

inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const double* A, int32_t lda, double* B, int32_t ldb)
{ double one = 1., zero = 0.; cublasDgeam(handle, trans == 'C' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const float* A, int32_t lda, float* B, int32_t ldb)
{ float one = 1.f, zero = 0.f; cublasSgeam(handle, trans == 'C' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, trans == 'C' ? CUBLAS_OP_C : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, trans == 'C' ? CUBLAS_OP_C : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }

inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }

template <class real_t, class complex_t>
inline int32_t svdk_dispatcher(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char transform, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  real_t* sigma, complex_t* UA, int32_t ldu, complex_t* RJ, int32_t ldr, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(real_t) * int64_t(algnK), r_bytes = sizeof(complex_t) * int64_t(algnN) * int64_t(K);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];

  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes));

  tranpose_copy(handle, 'C', N, K, RJ, ldr, (complex_t*)R, algnN);
  cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K,
    type_c, R, algnN, type_r, S, type_c, nullptr, algnN, type_c, nullptr, algnK, type_c, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, nullptr);
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  if (sigma) cudaMemcpyAsync(sigma, S, sizeof(real_t) * int64_t(K), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
  real_t *Svec = (real_t*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return s < s0; }))));

  if (0 < K) {
    tranpose_copy(handle, 'C', K, N, (complex_t*)R, algnN, RJ, ldr);
    if ((transform == 'Y' || transform == 'y')) {
      int32_t rows = int32_t((dev_work_bytes - r_bytes) / (uint64_t(K) * sizeof(complex_t))) & (~63);
      for (int32_t i = 0; i < M; i += rows) {
        int32_t m = std::min(M - i, rows), lds = std::min((m + 63) & (~63), rows);
        nn_gemm(handle, m, K, N, &UA[i], ldu, (complex_t*)R, algnN, (complex_t*)S, lds);
        tranpose_copy(handle, 'N', m, K, (complex_t*)S, lds, &UA[i], ldu);
      }
    }
  }
  return K;
}

extern "C" int32_t hyacinXsvdk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char transform, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  void* sigma, void* UA, int32_t ldu, void* RJ, int32_t ldr, hyacinPrecision_t ComputeType, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  if (0 < N && 0 < K) switch(ComputeType) {
    case HYACIN_F64:
      return svdk_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (double*)sigma, (double*)UA, ldu, (double*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32:
      return svdk_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (float*)sigma, (float*)UA, ldu, (float*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F64_COMPLEX:
      return svdk_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (double*)sigma, (cuDoubleComplex*)UA, ldu, (cuDoubleComplex*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32_COMPLEX:
      return svdk_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (float*)sigma, (cuComplex*)UA, ldu, (cuComplex*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    default: return 0;
  }
  return 0;
}
