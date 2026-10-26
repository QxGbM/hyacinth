
#include <internal.hpp>
#include <ext_arith.hpp>
#include <cooperative_groups.h>

template<class T, class S> static __device__ __forceinline__ T conv(S a) {
  constexpr bool ext_cf = std::is_same_v<S, complex_double2> || std::is_same_v<S, complex_float3> || std::is_same_v<S, complex_float4>;
  if constexpr(std::is_same_v<T, S>) { return a; } else
  if constexpr(std::is_same_v<T, cuDoubleComplex>)
  { if constexpr(ext_cf) return make_cuDoubleComplex(conv<double>(a.real), conv<double>(a.imag)); else return make_cuDoubleComplex(conv<double>(a.x), conv<double>(a.y)); } else
  if constexpr(std::is_same_v<T, cuComplex>)
  { if constexpr(ext_cf) return make_cuComplex(conv<float>(a.real), conv<float>(a.imag)); else return make_cuComplex(conv<float>(a.x), conv<float>(a.y)); } else
  if constexpr(std::is_same_v<T, __half2>)
  { if constexpr(ext_cf) return make_half2(conv<__half>(a.real), conv<__half>(a.imag)); else return make_half2(conv<__half>(a.x), conv<__half>(a.y)); } else
  if constexpr(std::is_same_v<T, __half> && std::is_same_v<S, double>) { return __double2half(a); } else
  if constexpr(std::is_same_v<T, __half> && std::is_same_v<S, float>) { return __float2half(a); } else
  if constexpr(std::is_same_v<S, __half>) { return conv<T>(__half2float(a)); } else
  if constexpr(std::is_same_v<S, double2>) { return conv<T>(a.x); } else
  if constexpr(std::is_same_v<T, double> && (std::is_same_v<S, float3> || std::is_same_v<S, float4>)) { return double(a.x) + double(a.y) + double(a.z); } else
  if constexpr(std::is_same_v<S, float3> || std::is_same_v<S, float4>) { return conv<T>(a.x); } else
  { return T(a); }
}

template <char mode, class Atype, class Btype>
__global__ void cvcpy_kernel(int64_t M, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) { if constexpr(mode == 'U') { B[y + x * ldb] = x < y ? Btype() : conv<Btype>(A[y + x * lda]); } else { B[y + x * ldb] = conv<Btype>(A[y + x * lda]); }}
};

template <class T> static __device__ __forceinline__ T conj(T a) {
  if constexpr(std::is_same_v<T, cuDoubleComplex>) { return make_cuDoubleComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<T, cuComplex>) { return make_cuComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<T, __half2>) { return make_half2(a.x, -a.y); } else
  { return a; }
}

template <char mode, class Atype, class Btype>
__global__ void cvcpy_transpose_kernel(int64_t M, int64_t N, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  __shared__ Btype matrix[16][17];
  int64_t yb = int64_t(blockIdx.x) << 4, y = int64_t(threadIdx.x), xb = int64_t(blockIdx.y) << 4, x = int64_t(threadIdx.y);
  if (((y + xb) < N) && ((x + yb) < M)) { if constexpr(mode == 'C') { matrix[y][x] = conj(conv<Btype>(A[(y + xb) + (x + yb) * lda])); } else { matrix[y][x] = conv<Btype>(A[(y + xb) + (x + yb) * lda]); }}
  cooperative_groups::this_thread_block().sync();
  if (((y + yb) < M) && ((x + xb) < N)) { B[(y + yb) + (x + xb) * ldb] = matrix[x][y]; }
};

template <class T> static __device__ __forceinline__ T float_one() {
  if constexpr(std::is_same_v<double, T>) { return 1.; } else if constexpr(std::is_same_v<float, T>) { return 1.f; } else if constexpr(std::is_same_v<__half, T>) { return CUDART_ONE_FP16; } else
  if constexpr(std::is_same_v<cuDoubleComplex, T>) { return make_cuDoubleComplex(1., 0.); } else if constexpr(std::is_same_v<cuComplex, T>) { return make_cuComplex(1.f, 0.f); } else
  if constexpr(std::is_same_v<__half2, T>) { return make_half2(CUDART_ONE_FP16, CUDART_ZERO_FP16); } else { return T(); }
}

template <char mode, class Atype, class Btype>
__global__ void scatter_conj_cvcpy_kernel(int64_t M, const int32_t* __restrict__ jpiv, const Atype* __restrict__ A, int64_t lda, Btype* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y); B = &B[int64_t(jpiv[x] - 1) + (y * ldb)];
  bool pred; if constexpr(mode == 'I') { pred = x < M; } else { pred = x < y; }
  if (y < M) { if (pred) { if constexpr(mode == 'I') { *B = (x == y) ? float_one<Btype>() : Btype(); } else { *B = Btype(); }} else { *B = conj(conv<Btype>(A[y + x * lda])); }}
};

template <class Atype, class Btype>
static inline void matcopy_dispatcher(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const Atype* A, int32_t lda, Btype* B, int32_t ldb, const int32_t* jpiv) {
  int64_t M64 = int64_t(M), N64 = int64_t(N), lda64 = int64_t(lda), ldb64 = int64_t(ldb);
  if (jpiv) {
    if (mode == 'I') { scatter_conj_cvcpy_kernel<'I'> <<< dim3(uint32_t((uint64_t(M64) + uint64_t(511)) >> 9), uint32_t(N)), 512, 0, stream >>> (M64, jpiv, A, lda64, B, ldb64); } else
    { scatter_conj_cvcpy_kernel<'U'> <<< dim3(uint32_t((uint64_t(M64) + uint64_t(511)) >> 9), uint32_t(N)), 512, 0, stream >>> (M64, jpiv, A, lda64, B, ldb64); }
  } else if (mode == 'U') { cvcpy_kernel<'U'> <<< dim3(uint32_t((uint64_t(M64) + uint64_t(511)) >> 9), uint32_t(N)), 512, 0, stream >>> (M64, A, lda64, B, ldb64); } else
  if (mode == 'C') {
    if constexpr(std::is_same_v<Atype, double> && std::is_same_v<Btype, double>)
    { double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, float> && std::is_same_v<Btype, float>)
    { float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, cuDoubleComplex> && std::is_same_v<Btype, cuDoubleComplex>)
    { cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, cuComplex> && std::is_same_v<Btype, cuComplex>)
    { cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    { cvcpy_transpose_kernel<'C'> <<< dim3(uint32_t((uint64_t(M64) + uint64_t(15)) >> 4), uint32_t((uint64_t(N64) + uint64_t(15)) >> 4)), dim3(16, 16), 0, stream >>> (M64, N64, A, lda64, B, ldb64); }
  } else if (mode == 'T') {
    if constexpr(std::is_same_v<Atype, double> && std::is_same_v<Btype, double>)
    { double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, float> && std::is_same_v<Btype, float>)
    { float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, cuDoubleComplex> && std::is_same_v<Btype, cuDoubleComplex>)
    { cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, cuComplex> && std::is_same_v<Btype, cuComplex>)
    { cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    { cvcpy_transpose_kernel<'T'> <<< dim3(uint32_t((uint64_t(M64) + uint64_t(15)) >> 4), uint32_t((uint64_t(N64) + uint64_t(15)) >> 4)), dim3(16, 16), 0, stream >>> (M64, N64, A, lda64, B, ldb64); }
  } else {
    if constexpr(std::is_same_v<Atype, double> && std::is_same_v<Btype, double>)
    { double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, float> && std::is_same_v<Btype, float>)
    { float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, cuDoubleComplex> && std::is_same_v<Btype, cuDoubleComplex>)
    { cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    if constexpr(std::is_same_v<Atype, cuComplex> && std::is_same_v<Btype, cuComplex>)
    { cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); } else
    { cvcpy_kernel<'A'> <<< dim3(uint32_t((uint64_t(M64) + uint64_t(511)) >> 9), uint32_t(N)), 512, 0, stream >>> (M64, A, lda64, B, ldb64); }
  }
}

namespace internal {

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv) { matcopy_dispatcher(stream, handle, mode, M, N, A, lda, B, ldb, jpiv); }

};
