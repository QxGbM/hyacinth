
#include <internal.hpp>

template <int32_t sign> __global__ void cf16_to_planar_kernel(int64_t M, const __half2* __restrict__ A, int64_t lda, __half* __restrict__ B, int64_t ldb, int64_t strideB) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x); 
  if (y < M) { int64_t x = int64_t(blockIdx.y); __half2 a = A[y + x * lda]; B = &B[y + x * ldb]; *B = a.x; if constexpr(sign) { B[strideB] = __hneg(a.y); } else { B[strideB] = a.y; }} else
  if (y < ldb) { int64_t x = int64_t(blockIdx.y); B = &B[y + x * ldb]; *B = CUDART_ZERO_FP16; *(B += strideB) = CUDART_ZERO_FP16; }
}

__global__ void cf16_to_planar_ext_kernel(int64_t M, const __half2* __restrict__ A, int64_t lda, __half* __restrict__ B, int64_t ldb, int64_t strideB) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x); 
  if (y < M) { int64_t x = int64_t(blockIdx.y); __half2 a = A[y + x * lda]; B = &B[y + x * ldb]; *B = __hneg(a.y); *(B += strideB) = a.x; *(B += strideB) = a.y; } else
  if (y < ldb) { int64_t x = int64_t(blockIdx.y); B = &B[y + x * ldb]; *B = CUDART_ZERO_FP16; *(B += strideB) = CUDART_ZERO_FP16; *(B += strideB) = CUDART_ZERO_FP16; }
}

__global__ void planar_to_cf16_kernel(int64_t M, const __half* __restrict__ A, int64_t lda, int64_t strideA, __half2* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x); 
  if (y < M) { int64_t x = int64_t(blockIdx.y); A = &A[y + x * lda]; B[y + x * ldb] = make_half2(*A, A[strideA]); }
}

void internal::matmul_cf16_cn(cudaStream_t stream, cublasHandle_t handle, cudaMemPool_t mempool, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, const __half2* B, int32_t ldb, __half2* C, int32_t ldc) {
  __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16;
  constexpr int32_t block = 8192; int32_t algnK = (K + 7) & (~7);
  __half* matA = nullptr, *matB = nullptr, *matC = nullptr;
  int64_t K64 = int64_t(K), N64 = int64_t(N), algnK64 = int64_t(algnK), strideB = N64 * algnK64, lda64 = int64_t(lda), ldc64 = int64_t(ldc), strideA = int64_t(block) * algnK64, strideC = int64_t(block) * N64;
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matA, uint64_t(2) * uint64_t(strideA) * uint64_t(sizeof(__half)), mempool, stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matB, uint64_t(3) * uint64_t(strideB) * uint64_t(sizeof(__half)), mempool, stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matC, uint64_t(2) * uint64_t(strideC) * uint64_t(sizeof(__half)), mempool, stream));
  cf16_to_planar_ext_kernel <<< dim3(uint32_t(algnK + 127) >> 7, uint32_t(N)), 128, 0, stream >>> (K64, B, int64_t(ldb), matB, algnK64, strideB);

  int32_t N_mul2 = N + N;
  for (int32_t i = 0; i < M; i += block) {
    int32_t rows = std::min(M - i, block);
    cf16_to_planar_kernel<1> <<< dim3(uint32_t(algnK + 127) >> 7, uint32_t(rows)), 128, 0, stream >>> (K64, &A[int64_t(i) * lda64], lda64, matA, algnK, strideA);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, N_mul2, algnK, &one, matA, CUDA_R_16F, algnK, &matB[strideB], CUDA_R_16F, algnK, &zero, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, N_mul2, algnK, &one, &matA[strideA], CUDA_R_16F, algnK, matB, CUDA_R_16F, algnK, &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    planar_to_cf16_kernel <<< dim3(uint32_t(rows + 127) >> 7, uint32_t(N)), 128, 0, stream >>> (int64_t(rows), matC, block, strideC, &C[int64_t(i)], ldc64);
  }
  CUDA_CHECK(cudaFreeAsync(matA, stream));
  CUDA_CHECK(cudaFreeAsync(matB, stream));
  CUDA_CHECK(cudaFreeAsync(matC, stream));
}

void internal::matmul_cf16_tn(cudaStream_t stream, cublasHandle_t handle, cudaMemPool_t mempool, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, const __half2* B, int32_t ldb, __half2* C, int32_t ldc) {
  __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16;
  constexpr int32_t block = 8192; int32_t algnK = (K + 7) & (~7);
  __half* matA = nullptr, *matB = nullptr, *matC = nullptr;
  int64_t K64 = int64_t(K), N64 = int64_t(N), algnK64 = int64_t(algnK), strideB = N64 * algnK64, lda64 = int64_t(lda), ldc64 = int64_t(ldc), strideA = int64_t(block) * algnK64, strideC = int64_t(block) * N64;
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matA, uint64_t(2) * uint64_t(strideA) * uint64_t(sizeof(__half)), mempool, stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matB, uint64_t(3) * uint64_t(strideB) * uint64_t(sizeof(__half)), mempool, stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matC, uint64_t(2) * uint64_t(strideC) * uint64_t(sizeof(__half)), mempool, stream));
  cf16_to_planar_ext_kernel <<< dim3(uint32_t(algnK + 127) >> 7, uint32_t(N)), 128, 0, stream >>> (K64, B, int64_t(ldb), matB, algnK64, strideB);

  int32_t N_mul2 = N + N;
  for (int32_t i = 0; i < M; i += block) {
    int32_t rows = std::min(M - i, block);
    cf16_to_planar_kernel<0> <<< dim3(uint32_t(algnK + 127) >> 7, uint32_t(rows)), 128, 0, stream >>> (K64, &A[int64_t(i) * lda64], lda64, matA, algnK, strideA);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, N_mul2, algnK, &one, matA, CUDA_R_16F, algnK, &matB[strideB], CUDA_R_16F, algnK, &zero, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, N_mul2, algnK, &one, &matA[strideA], CUDA_R_16F, algnK, matB, CUDA_R_16F, algnK, &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    planar_to_cf16_kernel <<< dim3(uint32_t(rows + 127) >> 7, uint32_t(N)), 128, 0, stream >>> (int64_t(rows), matC, block, strideC, &C[int64_t(i)], ldc64);
  }
  CUDA_CHECK(cudaFreeAsync(matA, stream));
  CUDA_CHECK(cudaFreeAsync(matB, stream));
  CUDA_CHECK(cudaFreeAsync(matC, stream));
}

void internal::matmul_cf16_nn(cudaStream_t stream, cublasHandle_t handle, cudaMemPool_t mempool, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, const __half2* B, int32_t ldb, __half2* C, int32_t ldc) {
  __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16;
  constexpr int32_t block = 8192; int32_t algnK = (K + 7) & (~7);
  __half* matA = nullptr, *matB = nullptr, *matC = nullptr;
  int64_t K64 = int64_t(K), N64 = int64_t(N), algnK64 = int64_t(algnK), strideB = N64 * algnK64, lda64 = int64_t(lda), ldc64 = int64_t(ldc), strideA = int64_t(block) * algnK64, strideC = int64_t(block) * N64;
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matA, uint64_t(2) * uint64_t(strideA) * uint64_t(sizeof(__half)), mempool, stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matB, uint64_t(3) * uint64_t(strideB) * uint64_t(sizeof(__half)), mempool, stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void**)&matC, uint64_t(2) * uint64_t(strideC) * uint64_t(sizeof(__half)), mempool, stream));
  cf16_to_planar_ext_kernel <<< dim3(uint32_t(algnK + 127) >> 7, uint32_t(N)), 128, 0, stream >>> (K64, B, int64_t(ldb), matB, algnK64, strideB);
  if (K < algnK) { cudaMemset2DAsync(&matA[int64_t(block) * K64], uint64_t(strideA) * uint64_t(sizeof(__half)), 0, uint64_t(block) * uint64_t(algnK64 - K64) * uint64_t(sizeof(__half)), 2, stream); }

  int32_t N_mul2 = N + N;
  for (int32_t i = 0; i < M; i += block) {
    int32_t rows = std::min(M - i, block);
    cf16_to_planar_kernel<0> <<< dim3(uint32_t(rows + 127) >> 7, uint32_t(K)), 128, 0, stream >>> (int64_t(rows), &A[int64_t(i)], lda64, matA, block, strideA);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, N_mul2, algnK, &one, matA, CUDA_R_16F, block, &matB[strideB], CUDA_R_16F, algnK, &zero, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, N_mul2, algnK, &one, &matA[strideA], CUDA_R_16F, block, matB, CUDA_R_16F, algnK, &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    planar_to_cf16_kernel <<< dim3(uint32_t(rows + 127) >> 7, uint32_t(N)), 128, 0, stream >>> (int64_t(rows), matC, block, strideC, &C[int64_t(i)], ldc64);
  }
  CUDA_CHECK(cudaFreeAsync(matA, stream));
  CUDA_CHECK(cudaFreeAsync(matB, stream));
  CUDA_CHECK(cudaFreeAsync(matC, stream));
}
