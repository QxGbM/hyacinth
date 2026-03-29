
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
  HYACIN_ALG_CRT,
  HYACIN_ALG_LIMBS_ND,
  HYACIN_ALG_CRT_ND
} hyacinAlgorithm_t;

#ifdef __cplusplus
extern "C" {
#endif

int32_t hyacinXelem_bytes(
  hyacinPrecision_t Atype
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

void hyacinXsyherk_bufferSize(
  int32_t M,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t Gtype,
  hyacinAlgorithm_t alg,
  uint64_t* dev_work_bytes
);

void hyacinXsyherk(
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
  void* dev_work,
  hyacinAlgorithm_t alg
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

void hyacinXGinterp_bufferSize(
  int32_t N,
  int32_t K,
  hyacinPrecision_t AXtype,
  hyacinPrecision_t Gtype,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXGinterp(
  cublasHandle_t handle,
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
  void* dev_work,
  void* pinned_work
);

void hyacinXGsqr_bufferSize(
  cusolverDnHandle_t s_handle,
  int32_t N,
  int32_t K,
  hyacinPrecision_t AXtype,
  int32_t ldx,
  hyacinPrecision_t Gtype,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXGsqr(
  cublasHandle_t handle,
  cusolverDnHandle_t s_handle,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t AXtype,
  void* X,
  int32_t ldx,
  hyacinPrecision_t Gtype,
  void* G,
  int32_t ldg,
  void* dev_work,
  uint64_t dev_work_bytes,
  void* pinned_work
);

void hyacinXGevd_bufferSize(
  cusolverDnHandle_t s_handle,
  cusolverDnParams_t params,
  int32_t N,
  int32_t K,
  hyacinPrecision_t AXGtype,
  int32_t ldg,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXGevd(
  cusolverDnHandle_t s_handle,
  cusolverDnParams_t params,
  double epi,
  int32_t N,
  int32_t K,
  int32_t p,
  hyacinPrecision_t AXGtype,
  void* S,
  void* X,
  int32_t ldx,
  void* G,
  int32_t ldg,
  void* dev_work,
  uint64_t dev_work_bytes,
  void* pinned_work,
  uint64_t pinned_work_bytes
);

void hyacinXGsvd_bufferSize(
  cusolverDnHandle_t s_handle,
  cusolverDnParams_t params,
  int32_t N,
  int32_t K,
  hyacinPrecision_t AXtype,
  int32_t ldx,
  hyacinPrecision_t Gtype,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

int32_t hyacinXGsvd(
  cublasHandle_t handle,
  cusolverDnHandle_t s_handle,
  cusolverDnParams_t params,
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
  void* dev_work,
  uint64_t dev_work_bytes,
  void* pinned_work,
  uint64_t pinned_work_bytes
);

void hyacinXtransform_bufferSize(
  int32_t K,
  hyacinPrecision_t AXtype,
  uint64_t* dev_work_bytes
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
  int32_t ldx,
  void* dev_work,
  uint64_t dev_work_bytes
);

void hyacinXsyherk1Drow_bufferSize(
  int32_t localM,
  int32_t globalM,
  int32_t N,
  int32_t umax,
  hyacinPrecision_t Gtype,
  hyacinAlgorithm_t alg,
  uint64_t* dev_work_bytes
);

void hyacinXAllGatherV1Dcol_bufferSize(
  int32_t M,
  int32_t comm_size,
  hyacinPrecision_t Atype,
  uint64_t* dev_work_bytes,
  uint64_t* pinned_work_bytes
);

#ifndef NO_NCCL

void hyacinXsyherk1Drow(
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
  void* dev_work,
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
  uint64_t dev_work_bytes,
  void* dev_work,
  void* pinned_work,
  ncclComm_t row_comm
);

#endif

#ifdef __cplusplus
}
#endif
