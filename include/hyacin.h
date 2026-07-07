
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
  HYACIN_F16 = 2,
  HYACIN_DD = 3,
  HYACIN_QF = 4,
  HYACIN_F64_COMPLEX = 5,
  HYACIN_F32_COMPLEX = 6,
  HYACIN_F16_COMPLEX = 7,
  HYACIN_DD_COMPLEX = 8,
  HYACIN_QF_COMPLEX = 9
} hyacinPrecision_t;

typedef enum {
  HYACIN_ALG_LIMBS = 0,
  HYACIN_ALG_CRT = 1,
  HYACIN_ALG_LIMBS_ND = 2,
  HYACIN_ALG_CRT_ND = 3,
  CUBLAS_FLOAT_ND = 4
} hyacinAlgorithm_t;

typedef struct {
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  cusolverDnParams_t cusolverParams;
  void* pinnedWorkspace; // A 128-byte pinned workspace on host for host reduction
  void* timer;
} hyacinHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

void hyacinCreate(
  hyacinHandle_t* handle, // host-pointer
  int32_t create_timer
);

void hyacinDestroy(
  hyacinHandle_t handle
);

void hyacinXelem(
  char sel,
  hyacinPrecision_t Atype,
  hyacinPrecision_t* type, // host-pointer
  int32_t* bytes, // host-pointer
  cudaDataType_t* cutype // host-pointer
);

void hyacinXsyherk_autoTune(
  double epi,
  int32_t use_nd_allreduce,
  int32_t u_extra,
  int32_t* umax, // host-pointer
  hyacinPrecision_t Atype,
  hyacinPrecision_t* ComputeType, // host-pointer
  hyacinAlgorithm_t* alg // host-pointer
);

int32_t hyacinXsyherk(
  hyacinHandle_t handle,
  int32_t M,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg,
  hyacinAlgorithm_t alg
);

char hyacinXGevPcsvd_autoTune(
  int32_t N,
  int32_t K,
  hyacinPrecision_t Gtype
);

int32_t hyacinXGevPcsvd(
  hyacinHandle_t handle,
  char use_evd,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t Xtype,
  void* S, // device-pointer
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg
);

int32_t hyacinXGinterp(
  hyacinHandle_t handle,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t Atype,
  void* X, // device-pointer
  int32_t ldx,
  int32_t* jpiv, // device-pointer
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg
);

void hyacinXtransform(
  hyacinHandle_t handle,
  int32_t M,
  int32_t N,
  int32_t K,
  hyacinPrecision_t AXtype,
  void* A, // device-pointer
  int32_t lda,
  const void* X, // device-pointer
  int32_t ldx
);

#ifndef NO_NCCL

int32_t hyacinXsyherk1Drow(
  hyacinHandle_t handle,
  int32_t localM,
  int32_t globalM,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg,
  hyacinAlgorithm_t alg,
  ncclComm_t col_comm
);

int32_t hyacinXAllGatherV1Dcol(
  hyacinHandle_t handle,
  int32_t M,
  int32_t* K, // host-pointer
  hyacinPrecision_t Atype,
  void* A, // device-pointer
  int32_t lda,
  ncclComm_t row_comm
);

#endif

void hyacinSync_TimerSegments(
  hyacinHandle_t handle,
  double* kernelMs, // host-pointer
  double* commMs // host-pointer
);

#ifdef __cplusplus
}
#endif
