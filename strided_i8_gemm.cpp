
#include <internal.hpp>

void internal::int8::strided_r8i_ATA_gemm(cublasHandle_t handle, int32_t order, int32_t M, int32_t N, const int8_t* A, int32_t* C) {
  int32_t strideA = N * M, strideC = N * N;
  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);
  cudaMemsetAsync(C, 0, strideC * (2 * order - 1) * sizeof(int32_t), stream);

  int32_t one = 1;
  for (int32_t i = 0; i < order; ++i) {
    const int8_t* AT = &A[i * strideA];
    int32_t* C_i = &C[i * strideC];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N * order, M, &one, 
      AT, CUDA_R_8I, M, A, CUDA_R_8I, M, &one, C_i, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }
}

void internal::int8::strided_c8i_AHA_gemm(cublasHandle_t handle, int32_t order, int32_t M, int32_t N, const int8_t* A, int32_t* C) {
  int32_t strideA = N * M, strideC = N * N;
  cudaStream_t stream = nullptr;
  cublasGetStream(handle, &stream);
  cudaMemsetAsync(C, 0, 2 * strideC * (2 * order - 1) * sizeof(int32_t), stream);

  int32_t one = 1;
  for (int32_t i = 0; i < order; ++i) {
    const int8_t* rAH = &A[(i * 2) * strideA];
    int32_t* C_i = &C[(i * 2) * strideC];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N * order * 2, M, &one, 
      rAH, CUDA_R_8I, M, A, CUDA_R_8I, M, &one, C_i, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }

  const int8_t* iA = &A[strideA];
  int32_t minus_one = -1;
  for (int32_t i = 0; i < order; ++i) {
    const int8_t* iAH = &A[(i * 2 + 1) * strideA];
    int32_t* iC = &C[(i * 2 + 1) * strideC];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &minus_one, 
      iAH, CUDA_R_8I, M, 0, A, CUDA_R_8I, M, strideA * 2, &one, iC, CUDA_R_32I, N, strideC * 2, order, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

    int32_t* rC = &C[(i * 2) * strideC];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, M, &one, 
      iAH, CUDA_R_8I, M, 0, iA, CUDA_R_8I, M, strideA * 2, &one, rC, CUDA_R_32I, N, strideC * 2, order, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  }
}
