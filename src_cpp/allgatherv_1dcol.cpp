
#include <hyacin.h>
#include <cuComplex.h>

inline int32_t shifts(hyacinPrecision_t prec) {
  switch(prec) 
  { case HYACIN_F32: return 0; case HYACIN_F64: case HYACIN_F32_COMPLEX: return 1; case HYACIN_F64_COMPLEX: return 2; default: return 0; }
}

extern "C" void hyacinXAllGatherV1Dcol_bufferSize(int32_t M, int32_t maxK, int32_t lenK, hyacinPrecision_t Atype, uint64_t* dev_work_bytes) {
  uint64_t stride = (uint64_t(M) * uint64_t(maxK) + uint64_t(63)) & (~uint64_t(63));
  *dev_work_bytes = stride * uint64_t(lenK) * (uint64_t(sizeof(int32_t)) << shifts(Atype));
}

#ifndef NO_NCCL

template <hyacinPrecision_t prec>
inline void matrix_copy(cublasHandle_t handle, int32_t M, int32_t N, const void* A, int32_t lda, void* B, int32_t ldb) {
  if constexpr(prec == HYACIN_F64)
  { double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, (const double*)A, lda, &zero, (double*)B, ldb, (double*)B, ldb); }
  else if constexpr(prec == HYACIN_F32)
  { float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, (const float*)A, lda, &zero, (float*)B, ldb, (float*)B, ldb); }
  else if constexpr(prec == HYACIN_F64_COMPLEX) {
    cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, (const cuDoubleComplex*)A, lda, &zero, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)B, ldb);
  }
  else if constexpr(prec == HYACIN_F32_COMPLEX) {
    cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
    cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, (const cuComplex*)A, lda, &zero, (cuComplex*)B, ldb, (cuComplex*)B, ldb);
  }
}

template <class T, hyacinPrecision_t prec>
inline void allgatherv_1dcol(cublasHandle_t handle, int32_t M, int32_t maxK, int32_t iK, int32_t lenK, const int32_t* allK, T* A, int32_t lda, T* dev_work, ncclComm_t row_comm) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  uint64_t stride = (uint64_t(M) * uint64_t(maxK) + uint64_t(63)) & (~uint64_t(63));
  matrix_copy<prec>(handle, M, allK[iK], A, lda, &dev_work[int64_t(iK) * stride], M);
  ncclAllGather(&dev_work[int64_t(iK) * stride], dev_work, stride << shifts(prec), ncclInt32, row_comm, stream);

  for (int32_t i = 0; i < lenK; ++i) {
    matrix_copy<prec>(handle, M, allK[i], &dev_work[int64_t(i) * stride], M, A, lda);
    A = &A[int64_t(allK[i]) * int64_t(lda)];
  }
}

extern "C" void hyacinXAllGatherV1Dcol(cublasHandle_t handle, int32_t M, int32_t maxK, int32_t iK, int32_t lenK, const int32_t* allK, hyacinPrecision_t Atype, void* A, int32_t lda, void* dev_work, ncclComm_t row_comm) {
  switch(Atype) {
    case HYACIN_F64: allgatherv_1dcol<double, HYACIN_F64>(handle, M, maxK, iK, lenK, allK, (double*)A, lda, (double*)dev_work, row_comm); break;
    case HYACIN_F32: allgatherv_1dcol<float, HYACIN_F32>(handle, M, maxK, iK, lenK, allK, (float*)A, lda, (float*)dev_work, row_comm); break;
    case HYACIN_F64_COMPLEX: allgatherv_1dcol<cuDoubleComplex, HYACIN_F64_COMPLEX>(handle, M, maxK, iK, lenK, allK, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)dev_work, row_comm); break;
    case HYACIN_F32_COMPLEX: allgatherv_1dcol<cuComplex, HYACIN_F32_COMPLEX>(handle, M, maxK, iK, lenK, allK, (cuComplex*)A, lda, (cuComplex*)dev_work, row_comm); break;
    default: break;
  }
}

#endif
