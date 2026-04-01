
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

template<class TGT, class SRC> __device__ __forceinline__ TGT conv(SRC a);
template <> __device__ __forceinline__ double conv<double, double>(double a) { return a; }
template <> __device__ __forceinline__ double conv<double, float>(float a) { return double(a); }
template <> __device__ __forceinline__ double conv<double, double2>(double2 a) { return device::dd::dd2double(a); }
template <> __device__ __forceinline__ double conv<double, float4>(float4 a) { return device::qf::qf2double(a); }

template <> __device__ __forceinline__ float conv<float, double>(double a) { return float(a); }
template <> __device__ __forceinline__ float conv<float, float>(float a) { return a; }
template <> __device__ __forceinline__ float conv<float, double2>(double2 a) { return float(a.x); }
template <> __device__ __forceinline__ float conv<float, float4>(float4 a) { return a.x; }

template <int32_t complex, class Atype, class constAptr, class Btype, class Bptr> 
__global__ void scatter_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred; if constexpr(complex) pred = (((x << 1) | int64_t(1)) < y); else pred = (x < y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (pred) *B = Btype(); else *B = conv<Btype, Atype>(*A);
  }
};

template <class T> __device__ __forceinline__ T float_relu_sqrt(T a);
template <> __device__ __forceinline__ double float_relu_sqrt<double>(double a) { return sqrt(fmax(a, 0.)); };
template <> __device__ __forceinline__ float float_relu_sqrt<float>(float a) { return sqrtf(fmaxf(a, 0.f)); };

template <class real_t, class complex_t, class constCptr, class Cptr, class constRptr, class Rptr>
__global__ void evd_reorder_kernel(int64_t M, int64_t N_minus_one, constCptr A, int64_t lda, Cptr B, int64_t ldb, constRptr s_in, Rptr s_out) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) {
    if (x <= N_minus_one) B[y + int64_t(N_minus_one - x) * ldb] = A[y + x * lda];
      else if (y <= N_minus_one) s_out[N_minus_one - y] = float_relu_sqrt(s_in[y]);
  }
};

template <class T> inline cudaDataType_t cuda_type();
template <> inline cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }
template <> inline cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <> inline cudaDataType_t cuda_type<cuDoubleComplex>() { return CUDA_C_64F; }
template <> inline cudaDataType_t cuda_type<cuComplex>() { return CUDA_C_32F; }

template <class real_t, class complex_t, class constCptr, class Cptr, class constRptr, class Rptr>
inline int32_t tevd(cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p,
  void* X, int32_t ldx, void* G, int32_t ldg, void* S, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  cudaStream_t stream; cusolverDnGetStream(s_handle, &stream);
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  K = std::min(K, N); uint64_t s_bytes = uint64_t(sizeof(real_t)) * K;
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXsyevd_bufferSize(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, type_c, nullptr, ldx, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  if ((uint64_t(workspaceInBytesOnDevice) + s_bytes) <= dev_work_bytes && uint64_t(workspaceInBytesOnHost) <= pinned_work_bytes) {
    real_t* dev_S = (real_t*)(&((int8_t*)dev_work)[workspaceInBytesOnDevice]);
    int64_t x_start = int64_t(N - K);
    cusolverDnXsyevd(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, type_c, G, ldg, type_r, dev_S, type_c, dev_work, workspaceInBytesOnDevice, pinned_work, workspaceInBytesOnHost, nullptr);
    evd_reorder_kernel <real_t, complex_t, constCptr, Cptr, constRptr, Rptr> <<< dim3(uint32_t((N + 511) >> 9), uint32_t(K + 1)), 512, 0, stream >>>
      (int64_t(N), int64_t(K - 1), &((const complex_t*)G)[x_start * int64_t(ldg)], int64_t(ldg), (complex_t*)X, int64_t(ldx), &((const real_t*)dev_S)[x_start], (real_t*)S); 
    cudaMemcpyAsync(pinned_work, S, int64_t(K) * sizeof(real_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    real_t *Svec = (real_t*)pinned_work;
    double s0 = epi * double(Svec[0]);
    return std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return double(s) < s0; }))));
  }
  else throw std::runtime_error("Insufficient workspace provided for SYEVD.");
}

template <hyacinPrecision_t precG> inline int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);
template<> inline int32_t potrfp<HYACIN_F64>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f64(stream, handle, fillmode, epi, k, p, N, (double*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_F32>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f32(stream, handle, fillmode, epi, k, p, N, (float*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_DD>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f128_dd(stream, handle, fillmode, epi, k, p, N, (double2*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_QF>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_f128_qf(stream, handle, fillmode, epi, k, p, N, (float4*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_F64_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf64(stream, handle, fillmode, epi, k, p, N, (std::complex<double>*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_F32_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf32(stream, handle, fillmode, epi, k, p, N, (std::complex<float>*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_DD_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf128_dd(stream, handle, fillmode, epi, k, p, N, (complex_double2*)A, lda, jpiv, dev_work, pinned_work); }
template<> inline int32_t potrfp<HYACIN_QF_COMPLEX>(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, void* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work)
{ return internal::Cholesky::potrfp_cf128_qf(stream, handle, fillmode, epi, k, p, N, (complex_float4*)A, lda, jpiv, dev_work, pinned_work); }

template <class real_t, class complex_t>
inline int32_t ge_tsvd(cudaStream_t stream, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p,
  void* X, int32_t ldx, void* S, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgesvd_bufferSize(s_handle, params, 'O', 'N', N, K, type_c, nullptr, ldx, type_r, nullptr, type_c, nullptr, ldx, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  if (uint64_t(workspaceInBytesOnDevice) <= dev_work_bytes && uint64_t(workspaceInBytesOnHost) <= pinned_work_bytes) {
    cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K, type_c, X, ldx, type_r, S, type_c, X, ldx, type_c, nullptr, K, type_c, dev_work, workspaceInBytesOnDevice, pinned_work, workspaceInBytesOnHost, nullptr);
    cudaMemcpyAsync(pinned_work, S, int64_t(K) * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    real_t *Svec = (real_t*)pinned_work;
    double s0 = epi * double(Svec[0]);
    return std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return double(s) < s0; }))));
  }
  else throw std::runtime_error("Insufficient workspace provided for GESVD.");
}

template <hyacinPrecision_t precG, class constGptr, class Gtype>
inline int32_t gsvd_dispatcher(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char fillmode,
  double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, void* S, void* X, int32_t ldx, Gtype* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t* hpiv = (int32_t*)(&((Gtype*)pinned_work)[512]);
  std::iota(hpiv, &hpiv[N], 1);
  K = potrfp<precG>(stream, handle, fillmode, epi, K, p, N, G, ldg, hpiv, dev_work, pinned_work);
  cudaMemcpyAsync(X, hpiv, int64_t(N) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  if (0 < K) {
    constexpr int32_t block_threads = 512;
    int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
    uint32_t grid_x = uint32_t(K + 511) >> 9, grid_cx = uint32_t(uint64_t(CK) + uint64_t(511) >> 9);
    if (AXtype == HYACIN_F64) {
      double* Xptr = (double*)X, *Wptr = (double*)dev_work;
      scatter_cvcpy_kernel <0, Gtype, constGptr, double, double* __restrict__>
        <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), (const int32_t*)X, G, int64_t(ldg), Wptr, int64_t(K));
      double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<double, double>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
    else if (AXtype == HYACIN_F32) {
      float* Xptr = (float*)X, *Wptr = (float*)dev_work;
      scatter_cvcpy_kernel <0, Gtype, constGptr, float, float* __restrict__>
        <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), (const int32_t*)X, G, int64_t(ldg), Wptr, int64_t(K));
      float one = 1., zero = 0.; cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<float, float>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
    else if (AXtype == HYACIN_F64_COMPLEX) {
      cuDoubleComplex* Xptr = (cuDoubleComplex*)X, *Wptr = (cuDoubleComplex*)dev_work;
      scatter_cvcpy_kernel <1, Gtype, constGptr, double, double* __restrict__>
        <<< dim3(grid_cx, N), block_threads, 0, stream >>> (CK, (const int32_t*)X, G, cldg, (double*)Wptr, CK);
      cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
      cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<double, cuDoubleComplex>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
    else if (AXtype == HYACIN_F32_COMPLEX) {
      cuComplex* Xptr = (cuComplex*)X, *Wptr = (cuComplex*)dev_work;
      scatter_cvcpy_kernel <1, Gtype, constGptr, float, float* __restrict__>
        <<< dim3(grid_cx, N), block_threads, 0, stream >>> (CK, (const int32_t*)X, G, cldg, (float*)Wptr, CK);
      cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
      cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      return ge_tsvd<float, cuComplex>(stream, s_handle, params, epi, N, K, p, X, ldx, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    }
  }
  return K;
}

extern "C" void hyacinXGevPcsvd_bufferSize(cusolverDnHandle_t s_handle, cusolverDnParams_t params, char use_evd, int32_t N, int32_t K, hyacinPrecision_t AXtype, int32_t ldx, hyacinPrecision_t Gtype, int32_t ldg, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  if (N <= 0 || K <= 0) { *dev_work_bytes = *pinned_work_bytes = uint64_t(0); return; }
  int64_t x_bytes = int64_t(0), x_real_bytes = int64_t(0), g_real_bytes = int64_t(0);
  cudaDataType_t type_c = cudaDataType_t(), type_r = cudaDataType_t();
  switch (Gtype) {
    case HYACIN_F64: { g_real_bytes = sizeof(double); break; } case HYACIN_F32: { g_real_bytes = sizeof(float); break; } 
    case HYACIN_DD: { g_real_bytes = sizeof(double2); break; } case HYACIN_QF: { g_real_bytes = sizeof(float4); break; }
    case HYACIN_F64_COMPLEX: { g_real_bytes = sizeof(double); break; } case HYACIN_F32_COMPLEX: { g_real_bytes = sizeof(float); break; } 
    case HYACIN_DD_COMPLEX: { g_real_bytes = sizeof(double2); break; } case HYACIN_QF_COMPLEX: { g_real_bytes = sizeof(float4); break; }
    default: break;
  }

  switch (AXtype) {
    case HYACIN_F64: { type_c = type_r = cuda_type<double>(); x_bytes = x_real_bytes = sizeof(double); break; }
    case HYACIN_F32: { type_c = type_r = cuda_type<float>(); x_bytes = x_real_bytes = sizeof(float); break; } 
    case HYACIN_F64_COMPLEX: { type_c = cuda_type<cuDoubleComplex>(); type_r = cuda_type<double>(); x_bytes = sizeof(cuDoubleComplex); x_real_bytes = sizeof(double); break; }
    case HYACIN_F32_COMPLEX: { type_c = cuda_type<cuComplex>(); type_r = cuda_type<float>(); x_bytes = sizeof(cuComplex); x_real_bytes = sizeof(float); break; }
    default: break;
  }

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  if ((use_evd == 'Y' || use_evd == 'y') && (AXtype == Gtype)) {
    cusolverDnXsyevd_bufferSize(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, type_c, nullptr, ldg, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    *dev_work_bytes = uint64_t(workspaceInBytesOnDevice) + uint64_t(x_real_bytes * int64_t(K));
    *pinned_work_bytes = std::max(uint64_t(workspaceInBytesOnHost), uint64_t(x_real_bytes * int64_t(K)));
  }
  else {
    cusolverDnXgesvd_bufferSize(s_handle, params, 'O', 'N', N, K, type_c, nullptr, ldx, type_r, nullptr, type_c, nullptr, ldx, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    *dev_work_bytes = std::max(uint64_t(workspaceInBytesOnDevice), uint64_t(std::max(x_bytes * int64_t(N) * int64_t(std::min(N, K)), g_real_bytes * int64_t(N))));
    *pinned_work_bytes = std::max(uint64_t(workspaceInBytesOnHost), uint64_t(std::max(g_real_bytes * int64_t(512) + int64_t(sizeof(int32_t)) * int64_t(N), x_real_bytes * int64_t(K))));
  }
}

extern "C" int32_t hyacinXGevPcsvd(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char use_evd, char fillmode, double epi, int32_t N, int32_t K, int32_t p,
  hyacinPrecision_t AXtype, void* S, void* X, int32_t ldx, hyacinPrecision_t Gtype, void* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  if (N <= 0 || K <= 0) { return 0; }
  else if ((use_evd == 'Y' || use_evd == 'y') && (AXtype == Gtype)) switch (AXtype) {
    case HYACIN_F64: return tevd<double, double, const double* __restrict__, double* __restrict__, const double* __restrict__, double* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32: return tevd<float, float, const float* __restrict__, float* __restrict__, const float* __restrict__, float* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F64_COMPLEX: return tevd<double, cuDoubleComplex, const cuDoubleComplex* __restrict__, cuDoubleComplex* __restrict__, const double* __restrict__, double* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32_COMPLEX: return tevd<float, cuComplex, const cuComplex* __restrict__, cuComplex* __restrict__, const float* __restrict__, float* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    default: return 0;
  }
  else switch(Gtype) {
    case HYACIN_F64: return gsvd_dispatcher<HYACIN_F64, const double* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (double*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32: return gsvd_dispatcher<HYACIN_F32, const float* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (float*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_DD: return gsvd_dispatcher<HYACIN_DD, const double2* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (double2*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_QF: return gsvd_dispatcher<HYACIN_QF, const float4* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (float4*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F64_COMPLEX: return gsvd_dispatcher<HYACIN_F64_COMPLEX, const double* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (double*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32_COMPLEX: return gsvd_dispatcher<HYACIN_F32_COMPLEX, const float* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (float*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_DD_COMPLEX: return gsvd_dispatcher<HYACIN_DD_COMPLEX, const double2* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (double2*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_QF_COMPLEX: return gsvd_dispatcher<HYACIN_QF_COMPLEX, const float4* __restrict__>
      (handle, s_handle, params, fillmode, epi, N, K, p, AXtype, S, X, ldx, (float4*)G, ldg, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    default: return 0;
  }
}
