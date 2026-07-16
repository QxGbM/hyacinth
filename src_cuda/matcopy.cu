
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

template<class T, class S> __device__ __forceinline__ T conv(S a) {
  constexpr bool cf128 = std::is_same_v<S, complex_double2> || std::is_same_v<S, complex_float4>;
  if constexpr(std::is_same_v<T, S>) { return a; } else
  if constexpr(std::is_same_v<T, cuDoubleComplex>)
  { if constexpr(cf128) return make_cuDoubleComplex(conv<double>(a.real), conv<double>(a.imag)); else return make_cuDoubleComplex(conv<double>(a.x), conv<double>(a.y)); } else
  if constexpr(std::is_same_v<T, cuComplex>)
  { if constexpr(cf128) return make_cuComplex(conv<float>(a.real), conv<float>(a.imag)); else return make_cuComplex(conv<float>(a.x), conv<float>(a.y)); } else
  if constexpr(std::is_same_v<T, __half2>)
  { if constexpr(cf128) return make_half2(conv<__half>(a.real), conv<__half>(a.imag)); else return make_half2(conv<__half>(a.x), conv<__half>(a.y)); } else
  if constexpr(std::is_same_v<T, __half>) { return __float2half(conv<float>(a)); } else
  if constexpr(std::is_same_v<S, __half>) { return conv<T>(__half2float(a)); } else
  if constexpr(std::is_same_v<S, double2>) { return conv<T>(device::dd::dd2double(a)); } else
  if constexpr(std::is_same_v<S, float4>) { return conv<T>(device::qf::qf2double(a)); } else
  { return T(a); }
}

template <class Atype, class Btype>
__global__ void cvcpy_kernel(int64_t M, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) B[y + x * ldb] = conv<Btype>(A[y + x * lda]);
};

template <class T> __device__ __forceinline__ T float_one();
template <> __device__ __forceinline__ double float_one<double>() { return 1.; };
template <> __device__ __forceinline__ float float_one<float>() { return 1.f; };
template <> __device__ __forceinline__ __half float_one<__half>() { return CUDART_ONE_FP16; };
template <> __device__ __forceinline__ cuDoubleComplex float_one<cuDoubleComplex>() { return make_cuDoubleComplex(1., 0.); };
template <> __device__ __forceinline__ cuComplex float_one<cuComplex>() { return make_cuComplex(1.f, 0.f); };
template <> __device__ __forceinline__ __half2 float_one<__half2>() { return make_half2(CUDART_ONE_FP16, CUDART_ZERO_FP16); };

template <class T> __device__ __forceinline__ T conj(T a) { return a; }
template <> __device__ __forceinline__ cuDoubleComplex conj<cuDoubleComplex>(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); };
template <> __device__ __forceinline__ cuComplex conj<cuComplex>(cuComplex a) { return make_cuComplex(a.x, -a.y); };
template <> __device__ __forceinline__ __half2 conj<__half2>(__half2 a) { return make_half2(a.x, -a.y); };

template <char mode, class Atype, class Btype>
__global__ void scatter_conj_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t pred; if constexpr(mode == 'I') { pred = int32_t(x < M) + int32_t(x == y); } else { pred = int32_t(x < y); }
  if (y < M) {
    A = &A[y + x * lda]; B = &B[int64_t(jpiv[x] - 1) + (y * ldb)];
    if (pred) { if constexpr(mode == 'I') { *B = (pred == 2) ? float_one<Btype>() : Btype(); } else { *B = Btype(); }}
      else { *B = conj(conv<Btype>(*A)); }
  }
};

template <class Atype, class Btype>
inline void matcopy_dispatcher(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const Atype* A, int32_t lda, Btype* B, int32_t ldb) {
  int64_t M64 = int64_t(M), lda64 = int64_t(lda), ldb64 = int64_t(ldb);
  dim3 grid_x(uint32_t((uint64_t(M64) + uint64_t(511)) >> 9), uint32_t(N));
  if (jpiv) {
    if (mode == 'I') { scatter_conj_cvcpy_kernel<'I'> <<< grid_x, 512, 0, stream >>> (M64, jpiv, A, lda64, B, ldb64); }
      else { scatter_conj_cvcpy_kernel<'U'> <<< grid_x, 512, 0, stream >>> (M64, jpiv, A, lda64, B, ldb64); }
  }
    else { cvcpy_kernel <<< grid_x, 512, 0, stream >>> (M64, A, lda64, B, ldb64); }
}

namespace internal::Cholesky {

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, double* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, double* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, double* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, double* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, double* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, float* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, float* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, float* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, float* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, float* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, __half* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, __half* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, __half* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, __half* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, __half* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, cuDoubleComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, cuComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, cuComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, cuComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, cuComplex* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, __half2* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, __half2* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, __half2* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, __half2* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, __half2* B, int32_t ldb) { matcopy_dispatcher(stream, mode, M, N, jpiv, A, lda, B, ldb); }

};
