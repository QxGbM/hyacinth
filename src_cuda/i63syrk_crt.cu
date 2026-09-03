
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdexcept>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
struct u64_add {
  __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b) { a.x += b.x; a.y += b.y; return a; }
  __device__ __forceinline__ ulonglong3 operator()(ulonglong3 a, ulonglong3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
};

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, ulonglong2& acc) {
  uint64_t q = uint64_t(device::int8::round_i64(x, expon, expon));
  uint64_t lo = q << expon, sign = (-(q >> 63) << 32);
  acc.x += uint32_t(lo); acc.y += ((lo >> 32) | sign);
}

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, ulonglong3& acc) {
  uint64_t q = uint64_t(device::int8::round_i64(x, expon, expon));
  uint64_t lo = uint64_t(q) << expon, sign = (-(q >> 63)) << expon;
  uint64_t hi = expon ? (q >> (64 - expon)) : uint64_t(0);
  acc.x += uint32_t(lo); acc.y += uint32_t(lo >> 32); acc.z += (hi | sign);
}

template <int32_t ORDER, int32_t sign, class reduc_t>
__device__ __forceinline__ uint64_t* conv_acc(reduc_t acc, int32_t M, uint32_t corr, uint64_t* out, int32_t stride) {
  uint64_t a[ORDER]{ uint64_t(acc.x) };
  device::int8::add_shifted(a, int64_t(M), corr);
  device::int8::add_shifted(a, int64_t(acc.y), uint32_t(32));
  if constexpr(std::is_same_v<reduc_t, ulonglong3>) { device::int8::add_shifted(a, int64_t(acc.z), uint32_t(64)); }

  if constexpr(sign) { *out = -a[0]; } else { *out = a[0]; }
  if constexpr(1 < ORDER && sign) { *(out += stride) = -a[1]; } else if constexpr(1 < ORDER) { *(out += stride) = a[1]; }
  if constexpr(2 < ORDER && sign) { *(out += stride) = -a[2]; } else if constexpr(2 < ORDER) { *(out += stride) = a[2]; }
  return &out[stride];
}

template <int32_t ORDER, int32_t Complex, int32_t BLOCK_THREADS, class reduc_t, class real_t>
__global__ void vector_sum_kernel(int32_t M, const real_t* __restrict__ A, int64_t lda, uint32_t corr, const int32_t* __restrict__ vexp, uint64_t* __restrict__ vec_sum) {
  int32_t expon = vexp[blockIdx.x]; M = (expon == int_max) ? 0 : M;
  reduc_t threadA = reduc_t(); A = &A[int64_t(blockIdx.x) * lda];

  for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    accumulate(A[i], expon, threadA);

  if constexpr(Complex) {
    __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
    reduc_t threadB = threadA;
    if (int32_t(threadIdx.x) & 1) threadA = reduc_t(); else threadB = reduc_t();
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(threadA, u64_add());
    threadB = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(threadB, u64_add());
    if (threadIdx.x == 0) { conv_acc<ORDER, 0>(threadB, M, corr, conv_acc<ORDER, 0>(threadA, M, corr, &vec_sum[blockIdx.x], int32_t(gridDim.x)), int32_t(gridDim.x)); }
  } else {
    __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, u64_add());
    if (threadIdx.x == 0) { conv_acc<ORDER, 1>(threadA, M, corr, &vec_sum[blockIdx.x], int32_t(gridDim.x)); }
  }
}

template <class real_t, class matrix_t>
inline void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
  constexpr int32_t block_threads = 512, Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int64_t M64 = int64_t(M), lda64 = int64_t(lda); if constexpr(Complex) { M64 <<= 1; lda64 <<= 1; }
  if (corr < uint32_t(63)) switch(order) {
    case 1: vector_sum_kernel<1, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, corr, vexp, vec_sum); return;
    case 2: vector_sum_kernel<2, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, corr, vexp, vec_sum); return;
    case 3: vector_sum_kernel<3, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, corr, vexp, vec_sum); return;
    default: return;
  } else switch(order) {
    case 1: vector_sum_kernel<1, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, corr, vexp, vec_sum); return;
    case 2: vector_sum_kernel<2, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, corr, vexp, vec_sum); return;
    case 3: vector_sum_kernel<3, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, corr, vexp, vec_sum); return;
    default: return;
  }
}

inline void gemm_accum_crt(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, int32_t K, int32_t moduli, int32_t iter, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t beta, int32_t orderC, uint64_t* C, int32_t* W) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, W, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, mode, beta, N, orderA, iter, W, M, orderC, C);
  } else {
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
    } else {
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
inline void crt_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C) {
  constexpr int32_t Complex = !std::is_same_v<real_t, matrix_t>;
  int32_t orderB = (63 + (orderA << 3)) / 63;
  int32_t algnM = (M + 255) & (~255), algnN = (N + 63) & (~63);
  int64_t strideW = int64_t(algnM) * int64_t(N), strideB = int64_t(N) * int64_t(N) * int64_t(orderB);
  uint64_t mod_len = uint64_t(std::min(orderA, 8)), w_len = uint64_t(strideW) * mod_len, w_pad = uint64_t(algnN - N) * uint64_t(algnM), b_len = uint64_t(strideB) + (uint64_t(N) * uint64_t(orderB));
  if constexpr(Complex) { w_len *= uint64_t(3); b_len *= uint64_t(2); }

  int8_t* W = nullptr; int32_t* scratch = nullptr; uint64_t* B = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&W, w_len + w_pad, stream))
    throw std::runtime_error("Workspace (i8) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&scratch, uint64_t(algnN) * uint64_t(N) * mod_len * sizeof(int32_t), stream))
    throw std::runtime_error("Workspace (i32) allocation failed at Integer SY/HERK.");
  if (cudaSuccess != cudaMallocAsync((void**)&B, b_len * sizeof(uint64_t), stream))
    throw std::runtime_error("Workspace (u64) allocation failed at Integer SY/HERK.");

  for (int32_t i = 0; (i << 3) < orderA; ++i) {
    int32_t moduli = std::min(orderA - (i << 3), 8);
    int32_t beta = int32_t(0 < i);
    internal::int8::quantize(stream, i, M, A, lda, corr, vexp, moduli, N, algnM, W);
    if constexpr(Complex) {
      int64_t strideWm = int64_t(moduli) * strideW, strideW2 = strideWm * int64_t(2);
      gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, i, orderA, &W[strideW2], &W[strideW2], beta, orderB, B, scratch);
      gemm_accum_crt(stream, handle, 'A', algnN, N, algnM, moduli, i, orderA, W, &W[strideWm], beta, orderB, &B[strideB], scratch);
    } else { gemm_accum_crt(stream, handle, 'U', algnN, N, algnM, moduli, i, orderA, W, W, beta, orderB, B, scratch); }
  }
  cudaFreeAsync(W, stream); cudaFreeAsync(scratch, stream);

  if constexpr(Complex) { vector_sums<real_t>(stream, M, N, A, lda, corr, vexp, orderB, &B[strideB * int64_t(2)]); }
    else { vector_sums<real_t>(stream, M, N, A, lda, corr, vexp, orderB, &B[strideB]); }
  internal::int8::triangle_pack(stream, Complex, M, N, orderB, B, corr, beta, orderC, C);
  cudaFreeAsync(B, stream);
}

namespace internal::int8 {

  void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { crt_dispatcher<double>(stream, handle, M, N, orderA, A, lda, corr, vexp, beta, orderC, C); }

  void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { crt_dispatcher<float>(stream, handle, M, N, orderA, A, lda, corr, vexp, beta, orderC, C); }

  void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { crt_dispatcher<__half>(stream, handle, M, N, orderA, A, lda, corr, vexp, beta, orderC, C); }

  void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { crt_dispatcher<double>(stream, handle, M, N, orderA, A, lda, corr, vexp, beta, orderC, C); }

  void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { crt_dispatcher<float>(stream, handle, M, N, orderA, A, lda, corr, vexp, beta, orderC, C); }

  void AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, int32_t beta, int32_t orderC, uint64_t* C)
  { crt_dispatcher<__half>(stream, handle, M, N, orderA, A, lda, corr, vexp, beta, orderC, C); }

}
