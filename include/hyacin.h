
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
  char GramMatrixAlgorithm;
  int32_t BatchK, QuantizeBitCorrection, GramBitCorrection, JacobiSVDSweeps, RankOversampling;
  int32_t Batches; struct { int32_t U, Order, SegK, Prefix, Rows; char Algorithm; } *BatchTensor;
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  cudaMemPool_t mempool;
  void* timer, *pinnedWorkspace;
#ifndef NO_NCCL
  ncclComm_t col_comm, row_comm;
#endif
} hyacinHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

void hyacinCreate(
  hyacinHandle_t* handle // host-pointer
);

void hyacinDestroy(
  hyacinHandle_t* handle
);

void hyacinXherkBatchCreate(
  hyacinHandle_t* handle,
  double epi,
  int32_t N,
  hyacinPrecision_t Atype,
  uint64_t* Bbytes // host-pointer
);

void hyacinXherkBatch(
  hyacinHandle_t* handle,
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
  int8_t* Bdata // device-pointer
);

void hyacinXherkBatchFlush(
  hyacinHandle_t* handle,
  int32_t N,
  hyacinPrecision_t Atype,
  const int32_t* vexp, // device-pointer
  int32_t beta,
  int32_t orderC,
  uint64_t* C, // device-pointer
  int8_t* Bdata // device-pointer
);

void hyacinXGautoType(
  const hyacinHandle_t* handle,
  double epi,
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
  const hyacinHandle_t* handle,
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

void hyacinXtransform(
  const hyacinHandle_t* handle,
  int32_t M,
  int32_t N,
  int32_t K,
  hyacinPrecision_t Atype,
  const void* Ain, // device-pointer
  int32_t lda_in,
  void* Aout, // device-pointer
  int32_t lda_out,
  char Xtype,
  const void* X, // device-pointer
  int32_t ldx
); // In-place mode: Ain == Aout && lda_in == lda_out; Gather mode: Xtype == 'I'/'J', Identity mode: N <= 0

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

int32_t hyacinXAllGatherV1Dcol(
  const hyacinHandle_t* handle,
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
  ncclComm_t row_comm
);

#endif

void hyacinSync_TimerSegments(
  const hyacinHandle_t* handle,
  double* eventMs, // host-pointer
  int32_t lenMs
);

#ifdef __cplusplus
}
#endif
