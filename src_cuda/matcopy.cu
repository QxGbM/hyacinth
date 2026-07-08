
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

template<class T, class S> __device__ __forceinline__ T conv(S a) { return T(a); }
template <> __device__ __forceinline__ double conv<double, double2>(double2 a) { return device::dd::dd2double(a); }
template <> __device__ __forceinline__ double conv<double, float4>(float4 a) { return device::qf::qf2double(a); }
template <> __device__ __forceinline__ float conv<float, __half>(__half a) { return __half2float(a); }
template <> __device__ __forceinline__ __half conv<__half, float>(float a) { return __float2half(a); }

template <class Atype, class Btype>
__global__ void cvcpy_kernel(int64_t M, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) B[y + x * ldb] = conv<Btype, Atype>(A[y + x * lda]);
};

template <class T> __device__ __forceinline__ T float_one();
template <> __device__ __forceinline__ double float_one<double>() { return 1.; };
template <> __device__ __forceinline__ float float_one<float>() { return 1.f; };
template <> __device__ __forceinline__ __half float_one<__half>() { return CUDART_ONE_FP16; };

template <char mode, int32_t COMPLEX, class Atype, class Btype> 
__global__ void scatter_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred;
  if constexpr(mode == 'I') { if constexpr(COMPLEX) { pred = int32_t((x << 1) < M) + int32_t((x << 1) == y); } else { pred = int32_t(x < M) + int32_t(x == y); }}
    else { if constexpr(COMPLEX) { pred = int32_t(((x << 1) | int64_t(1)) < y); } else { pred = int32_t(x < y); }}
  if (y < M) {
    A = &A[y + x * lda]; B = &B[y + int64_t(jpiv[x] - 1) * ldb];
    if (pred) { if constexpr(mode == 'I') { *B = (pred == 2) ? float_one<Btype>() : Btype(); } else { *B = Btype(); }}
      else { *B = conv<Btype, Atype>(*A); }
  }
};

template <int32_t COMPLEX, class Atype, class Btype>
inline void matcopy_dispatcher(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const Atype* A, int32_t lda, Btype* B, int32_t ldb) {
  int64_t M64 = int64_t(M), lda64 = int64_t(lda), ldb64 = int64_t(ldb);
  if constexpr(COMPLEX) { M64 <<= 1; lda64 <<= 1; ldb64 <<= 1; }
  dim3 grid_x(uint32_t((uint64_t(M64) + uint64_t(511)) >> 9), uint32_t(N));
  if (jpiv) {
    if (mode == 'I' || mode == 'i') { scatter_cvcpy_kernel<'I', COMPLEX, Atype, Btype> <<< grid_x, 512, 0, stream >>> (M64, jpiv, A, lda64, B, ldb64); }
      else { scatter_cvcpy_kernel<'U', COMPLEX, Atype, Btype> <<< grid_x, 512, 0, stream >>> (M64, jpiv, A, lda64, B, ldb64); }
  }
    else { cvcpy_kernel<Atype, Btype> <<< grid_x, 512, 0, stream >>> (M64, A, lda64, B, ldb64); }
}

void internal::Cholesky::scatter_matcopy_f64_f64(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, double* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_f128_dd_f64(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, double* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_f128_qf_f64(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, double* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_f64_f32(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, float* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_f32_f32(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, float* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_f16_f32(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, float* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_f32_f16(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, __half* B, int32_t ldb)
{ matcopy_dispatcher<0>(stream, mode, M, N, jpiv, A, lda, B, ldb); }

void internal::Cholesky::scatter_matcopy_cf64_cf64(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, double2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const double*)A, lda, (double*)B, ldb); }

void internal::Cholesky::scatter_matcopy_cf128_dd_cf64(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, double2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const double2*)A, lda, (double*)B, ldb); }

void internal::Cholesky::scatter_matcopy_cf128_qf_cf64(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, double2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const float4*)A, lda, (double*)B, ldb); }

void internal::Cholesky::scatter_matcopy_cf64_cf32(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, float2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const double*)A, lda, (float*)B, ldb); }

void internal::Cholesky::scatter_matcopy_cf32_cf32(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float2* A, int32_t lda, float2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const float*)A, lda, (float*)B, ldb); }

void internal::Cholesky::scatter_matcopy_cf16_cf32(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, float2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const __half*)A, lda, (float*)B, ldb); }

void internal::Cholesky::scatter_matcopy_cf32_cf16(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float2* A, int32_t lda, __half2* B, int32_t ldb)
{ matcopy_dispatcher<1>(stream, mode, M, N, jpiv, (const float*)A, lda, (__half*)B, ldb); }
