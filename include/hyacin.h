
#pragma once
#include <stdint.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#ifndef NO_NCCL
#include <nccl.h>
#endif

typedef enum { 
  HYACIN_F64 = 0, HYACIN_F64_COMPLEX = 1,
  HYACIN_F32 = 2, HYACIN_F32_COMPLEX = 3,
  HYACIN_F16 = 4, HYACIN_F16_COMPLEX = 5,
  HYACIN_DD = 6, HYACIN_DD_COMPLEX = 7,
  HYACIN_TF = 8, HYACIN_TF_COMPLEX = 9,
  HYACIN_QF = 10, HYACIN_QF_COMPLEX = 11
} hyacinPrecision_t;

typedef struct {
  char GramMatrixAlgorithm;
  int32_t QuantizeBitCorrection, GramBitCorrection, JacobiSVDSweeps, RankOversampling;
  int32_t DeviceID, DeviceSMs, DeviceIsF64Capable;
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  cudaMemPool_t mempool;
  void* timer, *pinnedWorkspace;
#ifndef NO_NCCL
  ncclComm_t ncclComm;
#endif
} hyacinHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

void hyacinCreate(
  hyacinHandle_t* handle
);

void hyacinDestroy(
  hyacinHandle_t* handle
);

void hyacinXherkBatchInit(
  const hyacinHandle_t* handle,
  double epi,
  uint64_t M,
  int32_t N,
  hyacinPrecision_t Atype,
  int32_t* uA, // host-pointer
  int32_t* cPanels, // host-pointer
  int32_t* lPanels, // host-pointer
  uint64_t* stride // host-pointer
);

void hyacinXherkBatch(
  const hyacinHandle_t* handle,
  int32_t batches,
  int32_t segM,
  int32_t lastM, 
  int32_t N,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  int32_t u,
  const int32_t* vexp, // device-pointer
  int32_t beta,
  int32_t orderC,
  uint64_t* C // device-pointer
);

void hyacinXquantizeScaleBatch(
  const hyacinHandle_t* handle,
  int32_t batches,
  int32_t segM,
  int32_t lastM, 
  int32_t N,
  hyacinPrecision_t Atype,
  const void* A, // device-pointer
  int32_t lda,
  int32_t uA,
  int32_t* vexp // device-pointer
);

void hyacinXdequantizeInit(
  const hyacinHandle_t* handle,
  int32_t N,
  int32_t orderC,
  const uint64_t* C, // device-pointer
  const int32_t* vexp, // device-pointer
  hyacinPrecision_t Atype,
  hyacinPrecision_t* Gtype, // host-pointer
  int32_t* gElemBytes // host-pointer
);

void hyacinXdequantize(
  const hyacinHandle_t* handle,
  int32_t N,
  int32_t orderC,
  const uint64_t* C, // device-pointer
  const int32_t* vexp, // device-pointer
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg
);

int32_t hyacinXGevd(
  const hyacinHandle_t* handle,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  hyacinPrecision_t Atype,
  void* X,
  int32_t ldx,
  void* S, // device-pointer
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg
); // returns rank

int32_t hyacinXGinterp(
  const hyacinHandle_t* handle,
  char fillmode,
  double epi,
  int32_t N,
  int32_t K,
  hyacinPrecision_t Atype,
  void* X, // device-pointer
  int32_t ldx,
  int32_t* jpiv, // device-pointer
  hyacinPrecision_t Gtype,
  void* G, // device-pointer
  int32_t ldg
); // returns rank

void hyacinAllReduce1Drow(
  const hyacinHandle_t* handle,
  int32_t Complex,
  int32_t orderA,
  uint64_t N,
  uint64_t* A // device-pointer
);

void hyacinAllReduceVExp(
  const hyacinHandle_t* handle,
  uint64_t N,
  int32_t* vexp // device-pointer
);

void hyacinSync_TimerSegments(
  const hyacinHandle_t* handle,
  double* eventMs, // host-pointer
  int32_t lenMs
);

#ifdef __cplusplus
}
#endif
