
#include <hyacin.h>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <numeric>

__device__ __forceinline__ void conv(float a, double& b) { b = double(a); }
__device__ __forceinline__ void conv(double2 a, double& b) { b = device::dd::dd2double(a); }
__device__ __forceinline__ void conv(float4 a, double& b) { b = device::qf::qf2double(a); }

__device__ __forceinline__ void conv(double a, float& b) { b = float(a); }
__device__ __forceinline__ void conv(double2 a, float& b) { b = float(a.x); }
__device__ __forceinline__ void conv(float4 a, float& b) { b = a.x; }

template <int32_t no_conv, class constAptr, class BType, class Bptr>
__global__ void cvcpy_kernel(int64_t M, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + x * ldb];
    if constexpr(no_conv) *B = *A; else conv(*A, *B);
  }
};

template <class T> __device__ __forceinline__ T float_one();
template <> __device__ __forceinline__ double float_one<double>() { return 1.; };
template <> __device__ __forceinline__ float float_one<float>() { return 1.f; };
template <> __device__ __forceinline__ cuDoubleComplex float_one<cuDoubleComplex>() { return make_cuDoubleComplex(1., 0.); };
template <> __device__ __forceinline__ cuComplex float_one<cuComplex>() { return make_cuComplex(1.f, 0.f); };

template <class real_t, class constAptr, class Bptr>
__global__ void scatter_cpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, constAptr A, int64_t lda, Bptr B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (x < M) { *B = (y == x) ? float_one<real_t>() : real_t(); }
      else *B = *A;
  }
};

template <hyacinPrecision_t precA, class constAptr, class Atype>
inline void conv_copy_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const Atype* A, int32_t lda, void* B, int32_t ldb, hyacinPrecision_t precB) {
  constexpr int32_t block_threads = 512;
  if (precB == HYACIN_F64) {
    double* Bptr = (double*)B;
    dim3 grid(uint32_t(M + block_threads - 1) >> 9, uint32_t(N), 1);
    if constexpr (precA == HYACIN_F64) cvcpy_kernel <1, constAptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), Bptr, int64_t(ldb));
      else cvcpy_kernel <0, constAptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), Bptr, int64_t(ldb));
    if (M < N)
    { double one = 1.; cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, Bptr, ldb, &Bptr[int64_t(M) * int64_t(ldb)], ldb); };
  }
  else if (precB == HYACIN_F32) {
    float* Bptr = (float*)B;
    dim3 grid(uint32_t(M + block_threads - 1) >> 9, uint32_t(N), 1);
    if constexpr (precA == HYACIN_F32) cvcpy_kernel <1, constAptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), Bptr, int64_t(ldb));
      else cvcpy_kernel <0, constAptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(M), A, int64_t(lda), Bptr, int64_t(ldb));
    if (M < N)
    { float one = 1.f; cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, Bptr, ldb, &Bptr[int64_t(M) * int64_t(ldb)], ldb); };
  }
  else if (precB == HYACIN_F64_COMPLEX) {
    cuDoubleComplex* Bptr = (cuDoubleComplex*)B;
    int64_t CM = int64_t(M) << 1, clda = int64_t(lda) << 1, cldb = int64_t(ldb) << 1;
    dim3 grid(uint32_t(CM + int64_t(block_threads - 1)) >> 9, uint32_t(N), 1);
    if constexpr (precA == HYACIN_F64_COMPLEX) cvcpy_kernel <1, constAptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (CM, A, clda, (double*)Bptr, cldb);
      else cvcpy_kernel <0, constAptr, double, double* __restrict__> <<< grid, block_threads, 0, stream >>> (CM, A, clda, (double*)Bptr, cldb);
    if (M < N)
    { cuDoubleComplex one = make_cuDoubleComplex(1., 0.); cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, Bptr, ldb, &Bptr[int64_t(M) * int64_t(ldb)], ldb); };
  }
  else if (precB == HYACIN_F32_COMPLEX) {
    cuComplex* Bptr = (cuComplex*)B;
    int64_t CM = int64_t(M) << 1, clda = int64_t(lda) << 1, cldb = int64_t(ldb) << 1;
    dim3 grid(uint32_t(CM + int64_t(block_threads - 1)) >> 9, uint32_t(N), 1);
    if constexpr (precA == HYACIN_F32_COMPLEX) cvcpy_kernel <1, constAptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (CM, A, clda, (float*)Bptr, cldb);
      else cvcpy_kernel <0, constAptr, float, float* __restrict__> <<< grid, block_threads, 0, stream >>> (CM, A, clda, (float*)Bptr, cldb);
    if (M < N)
    { cuComplex one = make_cuComplex(1.f, 0.f); cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, Bptr, ldb, &Bptr[int64_t(M) * int64_t(ldb)], ldb); };
  }
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
inline int32_t diag_piv_dispatcher(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, int32_t* jpiv, Gtype* G, int32_t ldg, void* dev_work, void* pinned_work) {
  int32_t* hpiv = (int32_t*)(&((Gtype*)pinned_work)[512]);
  std::iota(hpiv, &hpiv[N], 1);
  int32_t rank = potrfp<precG>(stream, handle, epi, K, p, N, G, ldg, hpiv, dev_work, pinned_work);
  cudaMemcpyAsync(jpiv, hpiv, int64_t(N) * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  if (0 < rank)
    conv_copy_dispatcher<precG, constGptr>(stream, handle, rank, N, G, ldg, dev_work, rank, AXtype);
  return rank;
}

extern "C" void hyacinXGinterp_bufferSize(int32_t N, int32_t K, hyacinPrecision_t AXtype, hyacinPrecision_t Gtype, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  if (N <= 0 || K <= 0) { *dev_work_bytes = *pinned_work_bytes = uint64_t(0); return; }
  int64_t x_bytes = 0, g_real_bytes = 0;
  switch (Gtype) {
    case HYACIN_F64: { g_real_bytes = sizeof(double); break; } case HYACIN_F32: { g_real_bytes = sizeof(float); break; } 
    case HYACIN_DD: { g_real_bytes = sizeof(double2); break; } case HYACIN_QF: { g_real_bytes = sizeof(float4); break; } 
    case HYACIN_F64_COMPLEX: { g_real_bytes = sizeof(double); break; } case HYACIN_F32_COMPLEX: { g_real_bytes = sizeof(float); break; } 
    case HYACIN_DD_COMPLEX: { g_real_bytes = sizeof(double2); break; } case HYACIN_QF_COMPLEX: { g_real_bytes = sizeof(float4); break; } 
    default: break;
  }

  switch (AXtype) {
    case HYACIN_F64: { x_bytes = sizeof(double); break; } case HYACIN_F32: { x_bytes = sizeof(float); break; } 
    case HYACIN_F64_COMPLEX: { x_bytes = sizeof(cuDoubleComplex); break; } case HYACIN_F32_COMPLEX: { x_bytes = sizeof(cuComplex); break; } 
    default: break;
  }

  *dev_work_bytes = uint64_t(std::max(x_bytes * int64_t(N) * int64_t(std::min(N, K)), g_real_bytes * int64_t(N)));
  *pinned_work_bytes = uint64_t(g_real_bytes * int64_t(512)) + uint64_t(sizeof(int32_t) * int64_t(N));
}

extern "C" int32_t hyacinXGinterp(cublasHandle_t handle, double epi, int32_t N, int32_t K, int32_t p, hyacinPrecision_t AXtype, void* X, int32_t ldx, int32_t* jpiv, hyacinPrecision_t Gtype, void* G, int32_t ldg, void* dev_work, void* pinned_work) {
  if (N <= 0 || K <= 0) { return 0; }
  int32_t rank = 0;
  cudaStream_t stream; cublasGetStream(handle, &stream);
  switch (Gtype) {
    case HYACIN_F64:
      rank = diag_piv_dispatcher<HYACIN_F64, const double* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (double*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_F32:
      rank = diag_piv_dispatcher<HYACIN_F32, const float* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (float*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_DD:
      rank = diag_piv_dispatcher<HYACIN_DD, const double2* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (double2*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_QF:
      rank = diag_piv_dispatcher<HYACIN_QF, const float4* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (float4*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_F64_COMPLEX:
      rank = diag_piv_dispatcher<HYACIN_F64_COMPLEX, const double* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (double*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_F32_COMPLEX:
      rank = diag_piv_dispatcher<HYACIN_F32_COMPLEX, const float* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (float*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_DD_COMPLEX:
      rank = diag_piv_dispatcher<HYACIN_DD_COMPLEX, const double2* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (double2*)G, ldg, dev_work, pinned_work); break;
    case HYACIN_QF_COMPLEX:
      rank = diag_piv_dispatcher<HYACIN_QF_COMPLEX, const float4* __restrict__>(stream, handle, epi, N, K, p, AXtype, jpiv, (float4*)G, ldg, dev_work, pinned_work); break;
    default: break;
  }

  if (0 < rank) {
    constexpr int32_t block_threads = 512;
    dim3 grid(uint32_t(rank + block_threads - 1) >> 9, uint32_t(N), 1);
    switch (AXtype) {
      case HYACIN_F64:
        scatter_cpy_kernel <double, const double* __restrict__, double* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(rank), jpiv, (double*)dev_work, int64_t(rank), (double*)X, int64_t(ldx)); break;
      case HYACIN_F32:
        scatter_cpy_kernel <float, const float* __restrict__, float* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(rank), jpiv, (float*)dev_work, int64_t(rank), (float*)X, int64_t(ldx)); break;
      case HYACIN_F64_COMPLEX:
        scatter_cpy_kernel <cuDoubleComplex, const cuDoubleComplex* __restrict__, cuDoubleComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(rank), jpiv, (cuDoubleComplex*)dev_work, int64_t(rank), (cuDoubleComplex*)X, int64_t(ldx)); break;
      case HYACIN_F32_COMPLEX:
        scatter_cpy_kernel <cuComplex, const cuComplex* __restrict__, cuComplex* __restrict__> <<< grid, block_threads, 0, stream >>> (int64_t(rank), jpiv, (cuComplex*)dev_work, int64_t(rank), (cuComplex*)X, int64_t(ldx)); break;
      default: break;
    }
  }
  return rank;
}
