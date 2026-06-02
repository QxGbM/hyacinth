
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
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

template <class Atype, class Btype>
__global__ void cvcpy_kernel(int64_t M, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) B[y + x * ldb] = conv<Btype, Atype>(A[y + x * lda]);
};

template <class T> __device__ __forceinline__ T float_one();
template <> __device__ __forceinline__ double float_one<double>() { return 1.; };
template <> __device__ __forceinline__ float float_one<float>() { return 1.f; };
template <> __device__ __forceinline__ cuDoubleComplex float_one<cuDoubleComplex>() { return make_cuDoubleComplex(1., 0.); };
template <> __device__ __forceinline__ cuComplex float_one<cuComplex>() { return make_cuComplex(1.f, 0.f); };

template <class real_t>
__global__ void scatter_cpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, const real_t* __restrict__ A, int64_t lda, real_t* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (x < M) { *B = (y == x) ? float_one<real_t>() : real_t(); }
      else *B = *A;
  }
};

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

template <hyacinPrecision_t precG, class Gtype>
inline int32_t diag_piv_dispatcher(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, int32_t* jpiv, void* X, int32_t ldx, Gtype* G, int32_t ldg, void* pinned_work) {
  int32_t x_bytes, g_real_bytes = int32_t(sizeof(Gtype)); hyacinXelem('A', AXtype, nullptr, &x_bytes, nullptr);
  uint64_t dev_work_bytes = uint64_t(std::max(int64_t(x_bytes) * int64_t(N) * int64_t(std::min(N, K)), int64_t(g_real_bytes) * int64_t(N)));
  void* dev_work = nullptr; 
  if (cudaSuccess != cudaMallocAsync((void**)&dev_work, dev_work_bytes, stream))
    throw std::runtime_error("Workspace allocation failed at Interpolative decomposition.");

  K = potrfp(stream, handle, fillmode, epi, K, p, N, G, ldg, jpiv, dev_work, pinned_work);
  if (0 < K) {
    constexpr int32_t block_threads = 512;
    int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
    uint32_t grid_x = uint32_t(K + 511) >> 9, grid_cx = uint32_t((uint64_t(CK) + uint64_t(511)) >> 9);
    if (AXtype == HYACIN_F64) {
      double* Bptr = (double*)dev_work;
      cvcpy_kernel<Gtype, double> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), G, int64_t(ldg), Bptr, int64_t(K));
      if (K < N)
      { double one = 1.; cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N - K, &one, Bptr, K, &Bptr[int64_t(K) * int64_t(K)], K); };
      scatter_cpy_kernel<double> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), jpiv, Bptr, int64_t(K), (double*)X, int64_t(ldx));
    }
    else if (AXtype == HYACIN_F32) {
      float* Bptr = (float*)dev_work;
      cvcpy_kernel<Gtype, float> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), G, int64_t(ldg), Bptr, int64_t(K));
      if (K < N)
      { float one = 1.f; cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N - K, &one, Bptr, K, &Bptr[int64_t(K) * int64_t(K)], K); };
      scatter_cpy_kernel<float> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), jpiv, Bptr, int64_t(K), (float*)X, int64_t(ldx));
    }
    else if (AXtype == HYACIN_F64_COMPLEX) {
      cuDoubleComplex* Bptr = (cuDoubleComplex*)dev_work;
      cvcpy_kernel<Gtype, double> <<< dim3(grid_cx, N), block_threads, 0, stream >>> (CK, G, cldg, (double*)Bptr, CK);
      if (K < N)
      { cuDoubleComplex one = make_cuDoubleComplex(1., 0.); cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N - K, &one, Bptr, K, &Bptr[int64_t(K) * int64_t(K)], K); };
      scatter_cpy_kernel<cuDoubleComplex> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), jpiv, Bptr, int64_t(K), (cuDoubleComplex*)X, int64_t(ldx));
    }
    else if (AXtype == HYACIN_F32_COMPLEX) {
      cuComplex* Bptr = (cuComplex*)dev_work;
      cvcpy_kernel<Gtype, float> <<< dim3(grid_cx, N), block_threads, 0, stream >>> (CK, G, cldg, (float*)Bptr, CK);
      if (K < N)
      { cuComplex one = make_cuComplex(1.f, 0.f); cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, K, N - K, &one, Bptr, K, &Bptr[int64_t(K) * int64_t(K)], K); };
      scatter_cpy_kernel<cuComplex> <<< dim3(grid_x, N), block_threads, 0, stream >>> (int64_t(K), jpiv, Bptr, int64_t(K), (cuComplex*)X, int64_t(ldx));
    }
  }
  cudaFreeAsync(dev_work, stream);
  return K;
}

extern "C" int32_t hyacinXGinterp(hyacinHandle_t handle, char fillmode, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, void* X, int32_t ldx, int32_t* jpiv, hyacinPrecision_t Gtype, void* G, int32_t ldg) {
  if (N <= 0 || K <= 0) { return 0; }
  Timer::register_kernel(handle.cudaStream, handle.timer);
  switch (Gtype) {
    case HYACIN_F64:
      return diag_piv_dispatcher<HYACIN_F64>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (double*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_F32:
      return diag_piv_dispatcher<HYACIN_F32>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (float*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_DD:
      return diag_piv_dispatcher<HYACIN_DD>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (double2*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_QF:
      return diag_piv_dispatcher<HYACIN_QF>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (float4*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_F64_COMPLEX:
      return diag_piv_dispatcher<HYACIN_F64_COMPLEX>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (double*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_F32_COMPLEX:
      return diag_piv_dispatcher<HYACIN_F32_COMPLEX>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (float*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_DD_COMPLEX:
      return diag_piv_dispatcher<HYACIN_DD_COMPLEX>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (double2*)G, ldg, handle.pinnedWorkspace);
    case HYACIN_QF_COMPLEX:
      return diag_piv_dispatcher<HYACIN_QF_COMPLEX>(handle.cudaStream, handle.cublasHandle, fillmode, epi, N, K, p, AXtype, jpiv, X, ldx, (float4*)G, ldg, handle.pinnedWorkspace);
    default: return 0;
  }
}
