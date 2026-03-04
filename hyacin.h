
#pragma once

#include <stdint.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#ifndef NO_NCCL
#include <nccl.h>
#endif

typedef enum { 
  HYACIN_F64,
  HYACIN_F32,
  HYACIN_DD,
  HYACIN_QF,
  HYACIN_F64_COMPLEX,
  HYACIN_F32_COMPLEX,
  HYACIN_DD_COMPLEX,
  HYACIN_QF_COMPLEX
} hyacinPrecision_t;

typedef enum {
  HYACIN_ALG_LIMBS,
  HYACIN_ALG_CRT
} hyacinAlgorithm_t;

#ifdef __cplusplus
extern "C" {
#endif

void hyacinXcpqrk_autoTune(
  double epi,
  int32_t globalM,
  int32_t u_extra,
  int32_t* umax,
  hyacinPrecision_t Atype,
  hyacinPrecision_t* ComputeType,
  hyacinAlgorithm_t* alg
);

void hyacinXcpqrk_bufferSize(
  int32_t M,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t ComputeType,
  hyacinAlgorithm_t alg,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXcpqrk(
  cublasHandle_t handle,
  char mode,
  double epi,
  int32_t M,
  int32_t N,
  int32_t K,
  int32_t p,
  int32_t umax,
  hyacinPrecision_t Atype,
  const void* A,
  int32_t lda,
  int32_t* jpiv,
  hyacinPrecision_t Rtype,
  void* R,
  int32_t ldr,
  hyacinPrecision_t ComputeType,
  void* dev_work,
  void* pinned_work,
  hyacinAlgorithm_t alg
);

void hyacinXutvk_bufferSize(
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  double epi,
  int32_t N,
  int32_t K,
  hyacinPrecision_t ComputeType,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXutvk(
  cublasHandle_t handle,
  cusolverDnHandle_t s_handle,
  cusolverDnParams_t params,
  double epi,
  int32_t M,
  int32_t N,
  int32_t K,
  int32_t p,
  void* UA,
  int32_t ldu,
  void* RJ,
  int32_t ldr,
  hyacinPrecision_t ComputeType,
  uint64_t dev_work_bytes,
  void* dev_work,
  uint64_t pinned_work_bytes,
  void* pinned_work
);

void hyacinXcpqrk1Drow_bufferSize(
  int32_t localM,
  int32_t globalM,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t ComputeType,
  hyacinAlgorithm_t alg,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

void hyacinXAllGatherV1Dcol_bufferSize(
  int32_t M,
  int32_t maxK,
  int32_t lenK,
  hyacinPrecision_t Atype,
  uint64_t* dev_work_bytes
);

#ifndef NO_NCCL

int32_t hyacinXcpqrk1Drow(
  cublasHandle_t handle,
  char mode,
  double epi,
  int32_t localM,
  int32_t globalM,
  int32_t N,
  int32_t K,
  int32_t p,
  int32_t umax,
  hyacinPrecision_t Atype,
  const void* A,
  int32_t lda,
  int32_t* jpiv,
  hyacinPrecision_t Rtype,
  void* R,
  int32_t ldr,
  hyacinPrecision_t ComputeType,
  void* dev_work,
  void* pinned_work,
  hyacinAlgorithm_t alg,
  ncclComm_t col_comm
);

void hyacinXAllGatherV1Dcol(
  cublasHandle_t handle,
  int32_t M,
  int32_t iK,
  int32_t lenK,
  const int32_t* allK,
  hyacinPrecision_t Atype,
  void* A,
  int32_t lda,
  void* dev_work,
  ncclComm_t row_comm
);

#endif

#ifdef __cplusplus
}
#endif
