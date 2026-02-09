
#pragma once

#include <stdint.h>
#include <cublas_v2.h>

#ifndef NO_NCCL
#include <nccl.h>
#endif

typedef enum { 
  HYACIN_F64 = 0,
  HYACIN_F32 = 1,
  HYACIN_DD = 2,
  HYACIN_QF = 3,
  HYACIN_F64_COMPLEX = 8,
  HYACIN_F32_COMPLEX = 9,
  HYACIN_DD_COMPLEX = 10,
  HYACIN_QF_COMPLEX = 11
} hyacinPrecision_t;

typedef enum {
  HYACIN_ALG_LIMBS = 0,
  HYACIN_ALG_CRT = 1
} hyacinAlgorithm_t;

#ifdef __cplusplus
extern "C" {
#endif

void hyacinXcpqrk_autoTune(
  double epi,
  int64_t globalM,
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

#ifndef NO_NCCL

void hyacinXcpqrkD_bufferSize(
  int32_t localM,
  int64_t globalM,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t ComputeType,
  hyacinAlgorithm_t alg,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXcpqrkD(
  cublasHandle_t handle,
  ncclComm_t comm,
  char mode,
  double epi,
  int32_t localM,
  int64_t globalM,
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

#endif

#ifdef __cplusplus
}
#endif
