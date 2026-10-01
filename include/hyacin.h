
#pragma once

#include <stdint.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#define HYACIN_QUERY_U (int32_t)(0x80000000)

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

typedef struct {
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  cusolverDnParams_t cusolverParams;
  cudaMemPool_t mempool;
  void* pinnedWorkspace; // A 128-byte pinned workspace on host for host reduction
  void* timer;
#ifndef NO_NCCL
  ncclComm_t col_comm, row_comm;
#endif
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

void hyacinXGautoType(
  double epi,
  int32_t u_corr,
  int32_t g_corr,
  uint64_t M,
  int32_t N,
  hyacinPrecision_t Atype,
  int32_t* uA, // host-pointer
  int32_t* cPanels, // host-pointer
  int32_t* lPanels, // host-pointer
  uint64_t* stride, // host-pointer
  hyacinPrecision_t* Gtype, // host-pointer
  int32_t* gElemBytes // host-pointer
);

void hyacinXquantizeScale(
  hyacinHandle_t handle,
  int32_t M,
  int32_t N,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  int32_t uA,
  int32_t beta,
  int32_t* vexp // device-pointer
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

void* hyacinXherkBatchCreate(
  hyacinHandle_t handle,
  char alg,
  double epi,
  int32_t u_corr,
  int32_t batchK,
  int32_t N,
  hyacinPrecision_t Atype
);

void hyacinXherkBatchDestroy(
  hyacinHandle_t handle,
  void* batch // host-pointer
);

void hyacinXherkBatch(
  hyacinHandle_t handle,
  char alg,
  int32_t M,
  int32_t N,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  int32_t u_hint, // HYACIN_QUERY_U for query
  const int32_t* vexp, // device-pointer
  int32_t* beta, // host-pointer
  int32_t orderC,
  uint64_t* C, // device-pointer
  void* batch // host-pointer
);

void hyacinXherkBatchFlush(
  hyacinHandle_t handle,
  int32_t N,
  hyacinPrecision_t Atype,
  const int32_t* vexp, // device-pointer
  int32_t beta,
  int32_t orderC,
  uint64_t* C, // device-pointer
  void* batch // host-pointer
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
); // returns rank

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
); // returns rank

void hyacinXtransform(
  hyacinHandle_t handle,
  int32_t M,
  int32_t N,
  int32_t K,
  hyacinPrecision_t Atype,
  const void* Ain, // device-pointer
  int32_t lda_in,
  void* Aout, // device-pointer
  int32_t lda_out,
  const void* X, // device-pointer
  int32_t ldx
); // In-place mode: Ain == Aout && lda_in == lda_out; Identity mode: N <= 0

void hyacinAllReduce1Drow(
  hyacinHandle_t handle,
  int32_t Complex,
  int32_t orderA,
  uint64_t N,
  uint64_t* A // device-pointer
);

extern "C" void hyacinAllReduceVExp(
  hyacinHandle_t handle,
  uint64_t N,
  int32_t* vexp // device-pointer
);

int32_t hyacinXAllGatherV1Dcol(
  hyacinHandle_t handle,
  int32_t M,
  int32_t* K, // host-pointer
  int32_t AElemBytes,
  void* A, // device-pointer
  int32_t lda
); // returns local Koffset

#ifndef NO_NCCL

void hyacinCreate2D(
  hyacinHandle_t* handle, // host-pointer
  ncclComm_t col_comm,
  ncclComm_t row_comm,
  int32_t create_timer
);

#endif

void hyacinSync_TimerSegments(
  hyacinHandle_t handle,
  double* eventMs, // host-pointer
  int32_t lenMs
);

#ifdef __cplusplus
}
#endif
