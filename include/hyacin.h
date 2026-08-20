
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
  HYACIN_ALG_AUTO = 0,
  HYACIN_ALG_LIMBS = 1,
  HYACIN_ALG_CRT = 2
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

int32_t hyacinXelem(
  char sel,
  hyacinPrecision_t* Atype // host-pointer
);

void hyacinXsyherk_autoTune(
  double epi,
  int32_t u_extra,
  int32_t* umax, // host-pointer
  hyacinPrecision_t Atype,
  hyacinPrecision_t* ComputeType // host-pointer
);

void hyacinXsyherk(
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

void hyacinXherk(
  hyacinHandle_t handle,
  int32_t M,
  int32_t N,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  const int32_t* vexp, // device-pointer
  int32_t beta,
  int32_t orderC,
  uint64_t* C, // device-pointer
  hyacinAlgorithm_t alg
);

void hyacinXdequantize(
  hyacinHandle_t handle,
  int32_t N,
  int32_t orderC,
  const uint64_t* C, // device-pointer
  const int32_t* vexp, // device-pointer
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg
);

int32_t hyacinXGevPcsvd(
  hyacinHandle_t handle,
  char use_evd,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t Atype,
  void* X,
  int32_t ldx,
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
  hyacinPrecision_t Atype,
  void* A, // device-pointer
  int32_t lda,
  const void* X, // device-pointer
  int32_t ldx
);

#ifndef NO_NCCL

void hyacinXsyherk1Drow(
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

void hyacinXAllReduce1Drow(
  hyacinHandle_t handle,
  int32_t orderA,
  int32_t Complex,
  int64_t N,
  uint64_t* A, // device-pointer
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
