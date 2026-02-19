
#pragma once

#include <stdint.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

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

void hyacinXutvk_bufferSize(
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  double epi,
  int32_t N,
  int32_t K,
  int32_t ldr,
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
  const void* A,
  int32_t lda,
  void* RV,
  int32_t ldr,
  void* UT,
  int32_t ldu,
  hyacinPrecision_t ComputeType,
  uint64_t dev_work_bytes,
  void* dev_work,
  uint64_t pinned_work_bytes,
  void* pinned_work
);

void hyacinXcpqrk1Dcol_bufferSize(
  int32_t localM,
  int64_t globalM,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t ComputeType,
  hyacinAlgorithm_t alg,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

#ifndef NO_NCCL

int32_t hyacinXcpqrk1Dcol(
  cublasHandle_t handle,
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
  hyacinAlgorithm_t alg,
  ncclComm_t col_comm
);

#endif

#ifdef __cplusplus
}
#endif
