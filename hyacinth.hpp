
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <float4.hpp>

int32_t align_up(
  int32_t ld,
  int32_t align
);

int32_t align_c_fp32(int32_t ld);
int32_t align_c_fp64(int32_t ld);
int32_t align_c_i8(int32_t ld);

int32_t dpotrfp_gpu(cudaStream_t stream, int32_t N, double* A, int32_t lda, int32_t* ipiv);

int32_t spotrfp_gpu(cudaStream_t stream, int32_t N, float* A, int32_t lda, int32_t* ipiv);

int32_t float4_potrfp_gpu(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int32_t* ipiv);

int32_t zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv);

int32_t cpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<float>* A, int32_t lda, int32_t* ipiv);

int32_t complex_float4_potrfp_gpu(cudaStream_t stream, int32_t N, complex_float4* A, int32_t lda, int32_t* ipiv);
