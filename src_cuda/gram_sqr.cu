
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

template<class T> __device__ __forceinline__ T rcp(T a);
template <> __device__ __forceinline__ double rcp<double>(double a) { return 1. / a; }
template <> __device__ __forceinline__ float rcp<float>(float a) { return 1.f / a; }

template <class Atype, class constAptr, class Btype, class Bptr> 
__global__ void inverse_diag(int64_t N, constAptr A, int64_t inca, Bptr B) {
  int64_t x = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (x < N) B[x] = rcp<Btype>(conv<Btype, Atype>(A[x * inca]));
};

template <int32_t complex, class Atype, class constAptr, class Btype, class constBptr, class Bptr> 
__global__ void scale_scatter_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, constAptr A, int64_t lda, constBptr S, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred; if constexpr(complex) pred = (((x << 1) | int64_t(1)) < y); else pred = (x < y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (pred) *B = Btype(); else *B = conv<Btype, Atype>(*A) * S[y];
  }
};

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

template <class complex_t> inline uint64_t qr_work_bytes(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau);
template <> inline uint64_t qr_work_bytes<double>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau) {
  int32_t L1 = 0, L2 = 0;
  cusolverDnDgeqrf_bufferSize(s_handle, N, K, (double*)A, lda, &L1);
  cusolverDnDorgqr_bufferSize(s_handle, N, K, K, (const double*)A, lda, (const double*)tau, &L2);
  return uint64_t(int64_t(std::max(L1, L2)) * int64_t(sizeof(double)));
}
template <> inline uint64_t qr_work_bytes<float>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau) {
  int32_t L1 = 0, L2 = 0;
  cusolverDnSgeqrf_bufferSize(s_handle, N, K, (float*)A, lda, &L1);
  cusolverDnSorgqr_bufferSize(s_handle, N, K, K, (const float*)A, lda, (const float*)tau, &L2);
  return uint64_t(int64_t(std::max(L1, L2)) * int64_t(sizeof(float)));
}
template <> inline uint64_t qr_work_bytes<cuDoubleComplex>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau) {
  int32_t L1 = 0, L2 = 0;
  cusolverDnZgeqrf_bufferSize(s_handle, N, K, (cuDoubleComplex*)A, lda, &L1);
  cusolverDnZungqr_bufferSize(s_handle, N, K, K, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)tau, &L2);
  return uint64_t(int64_t(std::max(L1, L2)) * int64_t(sizeof(cuDoubleComplex)));
}
template <> inline uint64_t qr_work_bytes<cuComplex>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau) {
  int32_t L1 = 0, L2 = 0;
  cusolverDnCgeqrf_bufferSize(s_handle, N, K, (cuComplex*)A, lda, &L1);
  cusolverDnCungqr_bufferSize(s_handle, N, K, K, (const cuComplex*)A, lda, (const cuComplex*)tau, &L2);
  return uint64_t(int64_t(std::max(L1, L2)) * int64_t(sizeof(cuComplex)));
}

template <class complex_t> inline void qr_compute(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau, void* dev_work, uint64_t dev_work_bytes);
template <> inline void qr_compute<double>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau, void* dev_work, uint64_t dev_work_bytes) {
  int32_t Lwork = int32_t(dev_work_bytes / sizeof(double)), L1 = 0, L2 = 0;
  cusolverDnDgeqrf_bufferSize(s_handle, N, K, (double*)A, lda, &L1);
  cusolverDnDorgqr_bufferSize(s_handle, N, K, K, (const double*)A, lda, (const double*)tau, &L2);
  if (L1 <= Lwork && L2 <= Lwork) {
    cusolverDnDgeqrf(s_handle, N, K, (double*)A, lda, (double*)tau, (double*)dev_work, L1, nullptr);
    cusolverDnDorgqr(s_handle, N, K, K, (double*)A, lda, (const double*)tau, (double*)dev_work, L2, nullptr);
  } else throw std::runtime_error("Insufficient workspace provided for QR.");
}
template <> inline void qr_compute<float>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau, void* dev_work, uint64_t dev_work_bytes) {
  int32_t Lwork = int32_t(dev_work_bytes / sizeof(float)), L1 = 0, L2 = 0;
  cusolverDnSgeqrf_bufferSize(s_handle, N, K, (float*)A, lda, &L1);
  cusolverDnSorgqr_bufferSize(s_handle, N, K, K, (const float*)A, lda, (const float*)tau, &L2);
  if (L1 <= Lwork && L2 <= Lwork) {
    cusolverDnSgeqrf(s_handle, N, K, (float*)A, lda, (float*)tau, (float*)dev_work, L1, nullptr);
    cusolverDnSorgqr(s_handle, N, K, K, (float*)A, lda, (const float*)tau, (float*)dev_work, L2, nullptr);
  } else throw std::runtime_error("Insufficient workspace provided for QR.");
}
template <> inline void qr_compute<cuDoubleComplex>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau, void* dev_work, uint64_t dev_work_bytes) {
  int32_t Lwork = int32_t(dev_work_bytes / sizeof(cuDoubleComplex)), L1 = 0, L2 = 0;
  cusolverDnZgeqrf_bufferSize(s_handle, N, K, (cuDoubleComplex*)A, lda, &L1);
  cusolverDnZungqr_bufferSize(s_handle, N, K, K, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)tau, &L2);
  if (L1 <= Lwork && L2 <= Lwork) {
    cusolverDnZgeqrf(s_handle, N, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, (cuDoubleComplex*)dev_work, L1, nullptr);
    cusolverDnZungqr(s_handle, N, K, K, (cuDoubleComplex*)A, lda, (const cuDoubleComplex*)tau, (cuDoubleComplex*)dev_work, L2, nullptr);
  } else throw std::runtime_error("Insufficient workspace provided for QR.");
}
template <> inline void qr_compute<cuComplex>(cusolverDnHandle_t s_handle, int32_t N, int32_t K, void* A, int32_t lda, void* tau, void* dev_work, uint64_t dev_work_bytes) {
  int32_t Lwork = int32_t(dev_work_bytes / sizeof(cuComplex)), L1 = 0, L2 = 0;
  cusolverDnCgeqrf_bufferSize(s_handle, N, K, (cuComplex*)A, lda, &L1);
  cusolverDnCungqr_bufferSize(s_handle, N, K, K, (const cuComplex*)A, lda, (const cuComplex*)tau, &L2);
  if (L1 <= Lwork && L2 <= Lwork) {
    cusolverDnCgeqrf(s_handle, N, K, (cuComplex*)A, lda, (cuComplex*)tau, (cuComplex*)dev_work, L1, nullptr);
    cusolverDnCungqr(s_handle, N, K, K, (cuComplex*)A, lda, (const cuComplex*)tau, (cuComplex*)dev_work, L2, nullptr);
  } else throw std::runtime_error("Insufficient workspace provided for QR.");
}

template <hyacinPrecision_t precG, class constGptr, class Gtype>
inline int32_t gsqr_dispatcher(cublasHandle_t handle, cusolverDnHandle_t s_handle,
  double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, void* X, int32_t ldx, Gtype* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t* hpiv = (int32_t*)(&((Gtype*)pinned_work)[512]);
  std::iota(hpiv, &hpiv[N], 1);
  K = potrfp<precG>(stream, handle, epi, K, p, N, G, ldg, hpiv, dev_work, pinned_work);
  cudaMemcpyAsync(X, hpiv, int64_t(N) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  if (0 < K) {
    constexpr int32_t block_threads = 512;
    if (AXtype == HYACIN_F64) {
      double* Xptr = (double*)X, *Sptr = (double*)dev_work, *Wptr = &Sptr[N];
      uint32_t grid_x = uint32_t(K + block_threads - 1) >> 9;
      inverse_diag <Gtype, constGptr, double, double* __restrict__>
        <<< grid_x, block_threads, 0, stream >>> (int64_t(K), G, int64_t(ldg + 1), Sptr);
      scale_scatter_cvcpy_kernel <0, Gtype, constGptr, double, const double* __restrict__, double* __restrict__>
        <<< dim3(grid_x, N), block_threads, 0, stream >>>(int64_t(K), (const int32_t*)X, G, int64_t(ldg), Sptr, Wptr, int64_t(K));
      double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      qr_compute<double>(s_handle, N, K, X, ldx, G, dev_work, dev_work_bytes);
    }
    else if (AXtype == HYACIN_F32) {
      float* Xptr = (float*)X, *Sptr = (float*)dev_work, *Wptr = &Sptr[N];
      uint32_t grid_x = uint32_t(K + block_threads - 1) >> 9;
      inverse_diag <Gtype, constGptr, float, float* __restrict__>
        <<< grid_x, block_threads, 0, stream >>> (int64_t(K), G, int64_t(ldg + 1), Sptr);
      scale_scatter_cvcpy_kernel <0, Gtype, constGptr, float, const float* __restrict__, float* __restrict__>
        <<< dim3(grid_x, N), block_threads, 0, stream >>>(int64_t(K), (const int32_t*)X, G, int64_t(ldg), Sptr, Wptr, int64_t(K));
      float one = 1., zero = 0.; cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      qr_compute<float>(s_handle, N, K, X, ldx, G, dev_work, dev_work_bytes);
    }
    else if (AXtype == HYACIN_F64_COMPLEX) {
      cuDoubleComplex* Xptr = (cuDoubleComplex*)X, *Sptr = (cuDoubleComplex*)dev_work, *Wptr = &Sptr[N];
      int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
      uint32_t grid_x = uint32_t(K + block_threads - 1) >> 9;
      uint32_t grid_cx = uint32_t(uint64_t(CK) + uint64_t(block_threads - 1) >> 9);
      inverse_diag <Gtype, constGptr, double, double* __restrict__>
        <<< grid_x, block_threads, 0, stream >>> (int64_t(K), G, cldg + int64_t(2), (double*)Sptr);
      scale_scatter_cvcpy_kernel <1, Gtype, constGptr, double, const double* __restrict__, double* __restrict__>
        <<< dim3(grid_cx, N), block_threads, 0, stream >>>(CK, (const int32_t*)X, G, cldg, (const double*)Sptr, (double*)Wptr, CK);
      cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
      cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      qr_compute<cuDoubleComplex>(s_handle, N, K, X, ldx, G, dev_work, dev_work_bytes);
    }
    else if (AXtype == HYACIN_F32_COMPLEX) {
      cuComplex* Xptr = (cuComplex*)X, *Sptr = (cuComplex*)dev_work, *Wptr = &Sptr[N];
      int64_t CK = int64_t(K) << 1, cldg = int64_t(ldg) << 1;
      uint32_t grid_x = uint32_t(K + block_threads - 1) >> 9;
      uint32_t grid_cx = uint32_t(uint64_t(CK) + uint64_t(block_threads - 1) >> 9);
      inverse_diag <Gtype, constGptr, float, float* __restrict__>
        <<< grid_x, block_threads, 0, stream >>> (int64_t(K), G, cldg + int64_t(2), (float*)Sptr);
      scale_scatter_cvcpy_kernel <1, Gtype, constGptr, float, const float* __restrict__, float* __restrict__>
        <<< dim3(grid_cx, N), block_threads, 0, stream >>>(CK, (const int32_t*)X, G, cldg, (const float*)Sptr, (float*)Wptr, CK);
      cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
      cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, Wptr, K, &zero, Xptr, ldx, Xptr, ldx);
      qr_compute<cuComplex>(s_handle, N, K, X, ldx, G, dev_work, dev_work_bytes);
    }
  }
  return K;
}

extern "C" void hyacinXGsqr_bufferSize(cusolverDnHandle_t s_handle, int32_t N, int32_t K, hyacinPrecision_t AXtype, int32_t ldx, hyacinPrecision_t Gtype, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  if (N <= 0 || K <= 0) { *dev_work_bytes = *pinned_work_bytes = uint64_t(0); return; }
  int64_t x_bytes = int64_t(0), g_real_bytes = int64_t(0);
  uint64_t qr_bytes = uint64_t(0);
  switch (Gtype) {
    case HYACIN_F64: { g_real_bytes = sizeof(double); break; } case HYACIN_F32: { g_real_bytes = sizeof(float); break; } 
    case HYACIN_DD: { g_real_bytes = sizeof(double2); break; } case HYACIN_QF: { g_real_bytes = sizeof(float4); break; }
    case HYACIN_F64_COMPLEX: { g_real_bytes = sizeof(double); break; } case HYACIN_F32_COMPLEX: { g_real_bytes = sizeof(float); break; } 
    case HYACIN_DD_COMPLEX: { g_real_bytes = sizeof(double2); break; } case HYACIN_QF_COMPLEX: { g_real_bytes = sizeof(float4); break; }
    default: break;
  }

  switch (AXtype) {
    case HYACIN_F64: { qr_bytes = qr_work_bytes<double>(s_handle, N, K, nullptr, ldx, nullptr); x_bytes = sizeof(double); break; }
    case HYACIN_F32: { qr_bytes = qr_work_bytes<float>(s_handle, N, K, nullptr, ldx, nullptr); x_bytes = sizeof(float); break; } 
    case HYACIN_F64_COMPLEX: { qr_bytes = qr_work_bytes<cuDoubleComplex>(s_handle, N, K, nullptr, ldx, nullptr); x_bytes = sizeof(cuDoubleComplex); break; }
    case HYACIN_F32_COMPLEX: { qr_bytes = qr_work_bytes<cuComplex>(s_handle, N, K, nullptr, ldx, nullptr);x_bytes = sizeof(cuComplex); break; }
    default: break;
  }

  *dev_work_bytes = std::max(qr_bytes, uint64_t(std::max(x_bytes * int64_t(N) * int64_t(1 + std::min(N, K)), g_real_bytes * int64_t(N))));
  *pinned_work_bytes = uint64_t(g_real_bytes * int64_t(512) + int64_t(sizeof(int32_t)) * int64_t(N));
}

extern "C" int32_t hyacinXGsqr(cublasHandle_t handle, cusolverDnHandle_t s_handle, double epi, int32_t N, int32_t K, int32_t p,
  hyacinPrecision_t AXtype, void* X, int32_t ldx, hyacinPrecision_t Gtype, void* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work) {
  if (N <= 0 || K <= 0) { return 0; }

  switch(Gtype) {
    case HYACIN_F64:
      return gsqr_dispatcher<HYACIN_F64, const double* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (double*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_F32:
      return gsqr_dispatcher<HYACIN_F32, const float* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (float*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_DD:
      return gsqr_dispatcher<HYACIN_DD, const double2* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (double2*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_QF:
      return gsqr_dispatcher<HYACIN_QF, const float4* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (float4*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_F64_COMPLEX:
      return gsqr_dispatcher<HYACIN_F64_COMPLEX, const double* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (double*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_F32_COMPLEX:
      return gsqr_dispatcher<HYACIN_F32_COMPLEX, const float* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (float*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_DD_COMPLEX:
      return gsqr_dispatcher<HYACIN_DD_COMPLEX, const double2* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (double2*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    case HYACIN_QF_COMPLEX:
      return gsqr_dispatcher<HYACIN_QF_COMPLEX, const float4* __restrict__>(handle, s_handle, epi, N, K, p, AXtype, X, ldx, (float4*)G, ldg, dev_work, dev_work_bytes, pinned_work);
    default: return 0;
  }
}
