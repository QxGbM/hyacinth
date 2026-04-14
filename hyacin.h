
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
  HYACIN_F64_COMPLEX = 4,
  HYACIN_F32_COMPLEX = 5,
  HYACIN_DD_COMPLEX = 6,
  HYACIN_QF_COMPLEX = 7
} hyacinPrecision_t;

typedef enum {
  HYACIN_ALG_LIMBS = 0,
  HYACIN_ALG_CRT = 1,
  HYACIN_ALG_LIMBS_ND = 2,
  HYACIN_ALG_CRT_ND = 3,
  CUBLAS_FLOAT_ND = 4
} hyacinAlgorithm_t;

#ifdef __cplusplus
extern "C" {
#endif

void hyacinXelem(
  char sel,
  hyacinPrecision_t Atype,
  hyacinPrecision_t* type,
  int32_t* bytes,
  cudaDataType_t* cutype
);

void hyacinXsyherk_autoTune(
  double epi,
  int32_t use_nd_allreduce,
  int32_t u_extra,
  int32_t* umax,
  hyacinPrecision_t Atype,
  hyacinPrecision_t* ComputeType,
  hyacinAlgorithm_t* alg
);

int32_t hyacinXsyherk(
  cublasHandle_t handle,
  int32_t M,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t Atype,
  const void* A,
  int32_t lda,
  hyacinPrecision_t Gtype,
  void* G,
  int32_t ldg,
  hyacinAlgorithm_t alg
);

char hyacinXGevPcsvd_autoTune(
  int32_t N,
  int32_t K,
  hyacinPrecision_t Gtype,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXGevPcsvd(
  cublasHandle_t handle,
  cusolverDnHandle_t s_handle,
  cusolverDnParams_t params,
  char use_evd,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t AXtype,
  void* S,
  void* X,
  int32_t ldx,
  hyacinPrecision_t Gtype,
  void* G,
  int32_t ldg,
  void* pinned_work
);

int32_t hyacinXGinterp(
  cublasHandle_t handle,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t AXtype,
  void* X,
  int32_t ldx,
  int32_t* jpiv,
  hyacinPrecision_t Gtype,
  void* G,
  int32_t ldg,
  void* pinned_work
);

void hyacinXtransform(
  cublasHandle_t handle,
  int32_t M,
  int32_t N,
  int32_t K,
  hyacinPrecision_t AXtype,
  void* A,
  int32_t lda,
  const void* X,
  int32_t ldx
);

#ifndef NO_NCCL

int32_t hyacinXsyherk1Drow(
  cublasHandle_t handle,
  int32_t localM,
  int32_t globalM,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t Atype,
  const void* A,
  int32_t lda,
  hyacinPrecision_t Gtype,
  void* G,
  int32_t ldg,
  hyacinAlgorithm_t alg,
  ncclComm_t col_comm
);

int32_t hyacinXAllGatherV1Dcol(
  cudaStream_t stream,
  int32_t M,
  int32_t* K,
  hyacinPrecision_t Atype,
  void* A,
  int32_t lda,
  ncclComm_t row_comm
);

#endif

void hyacinSync_TimerSegments(
  cudaStream_t stream,
  double* kernel_time,
  double* comm_time
);

#ifdef __cplusplus
}
#endif
