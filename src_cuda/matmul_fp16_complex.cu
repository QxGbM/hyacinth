
#include <internal.hpp>
#include <stdexcept>

template <int64_t ld, int64_t stride> __global__ void cf16_to_planar_kernel(int64_t M, const __half2* __restrict__ A, int64_t lda, __half* __restrict__ B) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x); 
  if (y < M) { int64_t x = int64_t(blockIdx.y); __half2 a = A[y + x * lda]; B = &B[y + x * ld]; *B = a.x; B[stride] = a.y; }
}

template <int64_t ld, int64_t stride> __global__ void planar_to_cf16_kernel(int64_t M, const __half* __restrict__ A, __half2* __restrict__ B, int64_t ldb) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x); 
  if (y < M) { int64_t x = int64_t(blockIdx.y); A = &A[y + x * ld]; B[y + x * ldb] = make_half2(*A, A[stride]); }
}

void internal::matmul_cf16(cudaStream_t stream, cublasHandle_t handle, cudaMemPool_t mempool, char opA, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, const __half2* B, int32_t ldb, __half2* C, int32_t ldc) {
  __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16, minus_one = __float2half(-1.f);
  constexpr int32_t block = 4096; constexpr int64_t stride = int64_t(block) * int64_t(block); constexpr uint64_t bytes = uint64_t(stride) * uint64_t(sizeof(__half2));
  __half* matA = nullptr, *matB = nullptr, *matC = nullptr;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matA, bytes, mempool, stream))
    throw std::runtime_error("Workspace allocation failed at CF16 GEMM");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matB, bytes, mempool, stream))
    throw std::runtime_error("Workspace allocation failed at CF16 GEMM");
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matC, bytes, mempool, stream))
    throw std::runtime_error("Workspace allocation failed at CF16 GEMM");
  if (opA == 'C') for (int32_t i = 0; i < M; i += block) { int32_t rows = std::min(M - i, block);
    for (int32_t j = 0; j < N; j += block) { int32_t cols = std::min(N - j, block);
      for (int32_t k = 0; k < K; k += block) { int32_t reduc = std::min(K - k, block);
        cf16_to_planar_kernel<block, stride> <<< dim3(uint32_t(reduc + 511) >> 9, uint32_t(rows)), 512, 0, stream >>> (int64_t(reduc), &A[int64_t(k) + int64_t(i) * int64_t(lda)], int64_t(lda), matA);
        cf16_to_planar_kernel<block, stride> <<< dim3(uint32_t(reduc + 511) >> 9, uint32_t(cols)), 512, 0, stream >>> (int64_t(reduc), &B[int64_t(k) + int64_t(j) * int64_t(ldb)], int64_t(ldb), matB);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &one, matA, CUDA_R_16F, block, matB, CUDA_R_16F, block, k == 0 ? &zero : &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &one, matA, CUDA_R_16F, block, &matB[stride], CUDA_R_16F, block, k == 0 ? &zero : &one, &matC[stride], CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &minus_one, &matA[stride], CUDA_R_16F, block, matB, CUDA_R_16F, block, &one, &matC[stride], CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &one, &matA[stride], CUDA_R_16F, block, &matB[stride], CUDA_R_16F, block, &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
      }
      planar_to_cf16_kernel<block, stride> <<< dim3(uint32_t(rows + 511) >> 9, uint32_t(cols)), 512, 0, stream >>> (int64_t(rows), matC, &C[int64_t(i) + int64_t(j) * int64_t(ldc)], ldc);
    }
  }
  else if (opA == 'T') for (int32_t i = 0; i < M; i += block) { int32_t rows = std::min(M - i, block);
    for (int32_t j = 0; j < N; j += block) { int32_t cols = std::min(N - j, block);
      for (int32_t k = 0; k < K; k += block) { int32_t reduc = std::min(K - k, block);
        cf16_to_planar_kernel<block, stride> <<< dim3(uint32_t(reduc + 511) >> 9, uint32_t(rows)), 512, 0, stream >>> (int64_t(reduc), &A[int64_t(k) + int64_t(i) * int64_t(lda)], int64_t(lda), matA);
        cf16_to_planar_kernel<block, stride> <<< dim3(uint32_t(reduc + 511) >> 9, uint32_t(cols)), 512, 0, stream >>> (int64_t(reduc), &B[int64_t(k) + int64_t(j) * int64_t(ldb)], int64_t(ldb), matB);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &one, matA, CUDA_R_16F, block, matB, CUDA_R_16F, block, k == 0 ? &zero : &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &one, matA, CUDA_R_16F, block, &matB[stride], CUDA_R_16F, block, k == 0 ? &zero : &one, &matC[stride], CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &one, &matA[stride], CUDA_R_16F, block, matB, CUDA_R_16F, block, &one, &matC[stride], CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, reduc, &minus_one, &matA[stride], CUDA_R_16F, block, &matB[stride], CUDA_R_16F, block, &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
      }
      planar_to_cf16_kernel<block, stride> <<< dim3(uint32_t(rows + 511) >> 9, uint32_t(cols)), 512, 0, stream >>> (int64_t(rows), matC, &C[int64_t(i) + int64_t(j) * int64_t(ldc)], ldc);
    }
  }
  else for (int32_t i = 0; i < M; i += block) { int32_t rows = std::min(M - i, block);
    for (int32_t j = 0; j < N; j += block) { int32_t cols = std::min(N - j, block);
      for (int32_t k = 0; k < K; k += block) { int32_t reduc = std::min(K - k, block);
        cf16_to_planar_kernel<block, stride> <<< dim3(uint32_t(rows + 511) >> 9, uint32_t(reduc)), 512, 0, stream >>> (int64_t(rows), &A[int64_t(i) + int64_t(k) * int64_t(lda)], int64_t(lda), matA);
        cf16_to_planar_kernel<block, stride> <<< dim3(uint32_t(reduc + 511) >> 9, uint32_t(cols)), 512, 0, stream >>> (int64_t(reduc), &B[int64_t(k) + int64_t(j) * int64_t(ldb)], int64_t(ldb), matB);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, reduc, &one, matA, CUDA_R_16F, block, matB, CUDA_R_16F, block, k == 0 ? &zero : &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, reduc, &one, matA, CUDA_R_16F, block, &matB[stride], CUDA_R_16F, block, k == 0 ? &zero : &one, &matC[stride], CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, reduc, &one, &matA[stride], CUDA_R_16F, block, matB, CUDA_R_16F, block, &one, &matC[stride], CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, reduc, &minus_one, &matA[stride], CUDA_R_16F, block, &matB[stride], CUDA_R_16F, block, &one, matC, CUDA_R_16F, block, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
      }
      planar_to_cf16_kernel<block, stride> <<< dim3(uint32_t(rows + 511) >> 9, uint32_t(cols)), 512, 0, stream >>> (int64_t(rows), matC, &C[int64_t(i) + int64_t(j) * int64_t(ldc)], ldc);
    }
  }
  cudaFreeAsync(matA, stream); cudaFreeAsync(matB, stream); cudaFreeAsync(matC, stream);
}
