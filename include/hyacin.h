
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
  /* Performance parameters */
  /* ENV{HYACIN_GRAM_ALGORITHM}: ['L'/'l'] to use only int-limb decomposition; ['C'/'c'] to use only CRT; Others for auto; Default: 'A' */
  /* ENV{HYACIN_STRIDE_BATCH_PERF_PENALTY}: Runtime multiplier for using StridedBatchGEMM over GEMM; only used for auto; Default: 1.3 */
  /* ENV{HYACIN_GRAM_SEGMENT_K}: Segment length for partioning K-dimension in HERK for local exponents; Default: 8192 */
  /* ENV{HYACIN_TCGEMM_TARGET_K}: Long-reduction CRT TC-GEMM will segment K, but not smaller than TargetK; not used for limbs; Default 8192 */
  /* Tuning guide: Default options performs generally well across systems. */
  char GramMatrixAlgorithm;
  double StrideBatchPerfPenalty;
  int32_t GramSegmentK, TCGemmTargetK;

  /* Spectral parameters */
  /* ENV{HYACIN_GRAM_SVD_EPI_CORR}: Epsilon multiplier for SVD Preconditioning, eps_precond = eps_requested * GramSVDEpiCorrection; Default: 0.5 */
  /* ENV{HYACIN_QUANTIZE_BITS_CORR}: Extra quantization resolution, add this many bits to the estimated quantization resolution; Default: 7 */
  /* ENV{HYACIN_JACOBI_SVD_SWEEPS}: Maximum Jacobi iteration number, set to 0 to select the bidiagonalization-based SVD backend instead of Jacobi SVD; Default: 5; */
  /* Tuning guide:
    (1) Insufficient quantization resolution may contaminate the small-eigenvalue spectrum of the Gram matrix, causing excessive rank retention, expensive downstream factorization, or numerical failure.
    (2) We therefore recommend a modest positive resolution margin, with Gram algorithm set to CRT or auto.
    (3) Usually, 2-3 Jacobi Sweeps produce basis with excellent quality, 1 Jacobi Sweep is usable with slightly worse error bound. */
  double GramSVDEpiCorrection;
  int32_t QuantizeBitCorrection, JacobiSVDSweeps;

  /* Device parameters */
  int32_t DeviceID, DeviceSMs, DeviceIsF64Capable;
  uint32_t HandleOwnedResource;
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  cudaMemPool_t mempool;
  void* timer, *pinnedWorkspace;
#ifndef NO_NCCL
  ncclComm_t ncclComm;
#endif
} hyacinHandle_t; // Expect initialization to all zero before calling hyacinCreate()

#ifdef __cplusplus
extern "C" {
#endif

void hyacinCreate(
  hyacinHandle_t* handle
);

void hyacinDestroy(
  hyacinHandle_t* handle
);

int32_t hyacinInterpolateFit(
  const hyacinHandle_t* handle,
  double epi,
  char opA,
  int32_t M, // Rows of matrix op(A)
  int32_t N, // Cols of matrix op(A), Rows of matrix V, length of jpvt
  int32_t K, // Cols of matrix V
  const void* A, // device-pointer
  int32_t lda,
  int32_t* jpvt, // device-pointer, LAPACK 1-based index
  void* V, // device-pointer, only columns [1:rank] will be overwritten
  int32_t ldv,
  int64_t globalM, // Total rows of op(A) if distributed
  hyacinPrecision_t ComputeType
); // returns rank, either K or epi is reached

int32_t hyacinTsvdFit(
  const hyacinHandle_t* handle,
  double epi,
  char opA,
  int32_t M, // Rows of matrix op(A)
  int32_t N, // Cols of matrix op(A), Rows of matrix V
  int32_t K, // Length of vector S, Cols of matrix V
  const void* A, // device-pointer
  int32_t lda,
  void* S, // device-pointer
  void* V, // device-pointer, only columns [1:rank] will be overwritten
  int32_t ldv,
  int64_t globalM, // Total rows of op(A) if distributed
  hyacinPrecision_t ComputeType
); // returns rank, either K or epi is reached

int32_t hyacinTsvdFitSegmentN(
  const hyacinHandle_t* handle,
  double epi,
  char opA,
  int32_t M, // Rows of matrix op(A)
  int32_t N, // Cols of matrix op(A), Rows of matrix V
  int32_t K, // Length of vector S, Cols of matrix V
  const void* A, // device-pointer
  int32_t lda,
  void* S, // device-pointer
  void* V, // device-pointer, only columns [1:rank] will be overwritten
  int32_t ldv,
  int64_t globalM, // Total rows of op(A) if distributed
  hyacinPrecision_t ComputeType
); // returns rank, either K or epi is reached

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

double hyacinCheckNorm(
  const hyacinHandle_t* handle,
  char opA,
  int32_t M,
  int32_t N,
  const void* A, // device-pointer
  int32_t lda,
  int32_t rank, // Can be zero, V won't be accessed if rank <= 0
  const void* V, // device-pointer
  int32_t ldv,
  hyacinPrecision_t ComputeType
); // returns ||A-A*orth(V)*orth(V)^H||^2_F

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
