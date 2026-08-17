
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <stdexcept>

struct u64_add {
  __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b) { a.x += b.x; a.y += b.y; return a; }
  __device__ __forceinline__ ulonglong3 operator()(ulonglong3 a, ulonglong3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
};

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, uint64_t lo, uint32_t, ulonglong2& acc) {
  int64_t q = device::int8::round_i64(x, expon, expon);
  lo += uint64_t(q) << expon;
  acc.x += uint32_t(lo); acc.y += uint32_t(lo >> 32);
}

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, uint64_t lo, uint32_t hi, ulonglong3& acc) {
  constexpr uint64_t i63 = 0x7fffffffffffffffllu;
  constexpr uint32_t i31 = 0x7fffffff;

  int64_t q = device::int8::round_i64(x, expon, expon);
  lo += (uint64_t(q) << expon) & i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
  acc.x += uint32_t(lo); acc.y += uint32_t(lo >> 32) & i31; acc.z += hi;
}

template <int32_t ORDER, int32_t sign, class reduc_t>
__device__ __forceinline__ uint64_t* conv_acc(reduc_t acc, uint64_t* out, int64_t stride) {
  uint64_t a[ORDER]{ acc.x };
  if constexpr(1 < ORDER) { if constexpr(std::is_same_v<reduc_t, ulonglong3>) a[1] = acc.z; else a[1] = uint64_t(0); }
  if constexpr(2 < ORDER) { a[2] = uint64_t(0); }

  device::int8::add_shifted(a, acc.y, uint32_t(32)); 
  if constexpr(sign) { *out = -a[0]; } else { *out = a[0]; }
  if constexpr(1 < ORDER && sign) { *(out += stride) = -a[1]; } else if constexpr(1 < ORDER) { *(out += stride) = a[1]; }
  if constexpr(2 < ORDER && sign) { *(out += stride) = -a[2]; } else if constexpr(2 < ORDER) { *(out += stride) = a[2]; }
  return &out[stride];
}

template <int32_t ORDER, int32_t Complex, int32_t BLOCK_THREADS, class reduc_t, class real_t>
__global__ void vector_sum_kernel(int64_t M, const real_t* __restrict__ A, int64_t lda, uint64_t lo, uint32_t hi, int32_t umax, const int32_t* __restrict__ vexp, uint64_t* __restrict__ vec_sum) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  int64_t iter = int64_t(blockIdx.x) * lda, iter_end = iter + M;
  int32_t expon = umax - vexp[blockIdx.x];
  reduc_t threadA = reduc_t();

  for (iter += int64_t(threadIdx.x); iter < iter_end; iter += inci)
    accumulate(A[iter], expon, lo, hi, threadA);
  iter = int64_t(blockIdx.x);

  if constexpr(Complex) {
    __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
    reduc_t threadB = threadA;
    if (int32_t(threadIdx.x) & 1) threadA = reduc_t(); else threadB = reduc_t();
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(threadA, u64_add());
    threadB = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(threadB, u64_add());
    if (threadIdx.x == 0) { conv_acc<ORDER, 0>(threadB, conv_acc<ORDER, 0>(threadA, &vec_sum[blockIdx.x], int64_t(gridDim.x)), int64_t(gridDim.x)); }
  }
  else {
    __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, u64_add());
    if (threadIdx.x == 0) { conv_acc<ORDER, 1>(threadA, &vec_sum[blockIdx.x], int64_t(gridDim.x)); }
  }
}

template <class real_t, class matrix_t>
inline void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
  constexpr int32_t block_threads = 512, Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  int64_t M64 = int64_t(M), lda64 = int64_t(lda); if constexpr(Complex) { M64 <<= 1; lda64 <<= 1; }
  if (umax < 63) switch(order) {
    case 1: vector_sum_kernel<1, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 2: vector_sum_kernel<2, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 3: vector_sum_kernel<3, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    default: return;
  } else switch(order) {
    case 1: vector_sum_kernel<1, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 2: vector_sum_kernel<2, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 3: vector_sum_kernel<3, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    default: return;
  }
}

inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, int32_t K, int32_t moduli, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t iter, int32_t beta, uint64_t* C, int32_t orderC, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, beta, N, orderA, iter, W, M, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[k], *AN_k = &A[k];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, k == 0 ? beta : 1, N, orderA, iter, W, M, orderC, C);
    }

    const int8_t* AT_k = &AT[range_k], *AN_k = &A[range_k];
    if (rem <= iter_h) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, range_k == 0 ? beta : 1, N, orderA, iter, W, M, orderC, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, range_k == 0 ? beta : 1, N, orderA, iter, W, M, orderC, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, mode, 1, N, orderA, iter, W, M, orderC, C);
    }
  }
}

template <class real_t, class matrix_t>
inline void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(N) * int64_t(N) * int64_t(orderC);
  uint64_t mod_len = uint64_t(std::min(orderA, 8)), w_len = uint64_t(strideA) * mod_len, w_pad = uint64_t(algnN - N) * uint64_t(algnM);
  if constexpr(Complex) { w_len *= uint64_t(3); }

  int8_t* W = nullptr; int32_t* scratch = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&W, w_len + w_pad, stream))
    throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(N) * mod_len * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    internal::int8::quantize(stream, i, M, A, lda, umax, vexp, moduli, N, algnM, W);
    if constexpr(Complex) {
      int64_t strideW = int64_t(moduli) * strideA, strideW2 = strideW * int64_t(2);
      gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, &W[strideW2], &W[strideW2], i, beta, C, orderC, scratch);
      gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, moduli, orderA, W, &W[strideW], i, beta, &C[strideC], orderC, scratch);
    } else { gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, orderA, W, W, i, beta, C, orderC, scratch); }
  }
  cudaFreeAsync(W, stream); cudaFreeAsync(scratch, stream);

  if constexpr(Complex) { C = &C[strideC * int64_t(2)]; } else { C = &C[strideC]; }
  vector_sums<real_t>(stream, M, N, A, lda, umax, vexp, orderC, C);
}

namespace internal::int8 {

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt<double>(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt<float>(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt<__half>(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt<double>(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt<float>(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t orderC, uint64_t* C)
  { AHA_crt<__half>(stream, handle, M, N, orderA, A, lda, umax, vexp, orderC, C); }

}
