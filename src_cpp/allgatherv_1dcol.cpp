
#include <hyacin.h>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cuComplex.h>
#include <numeric>

extern "C" void hyacinXAllGatherV1Dcol_bufferSize(int32_t M, int32_t maxK, int32_t lenK, hyacinPrecision_t Atype, uint64_t* dev_work_bytes) {
  uint64_t stride = (uint64_t(M) * uint64_t(maxK) + uint64_t(63)) & (~uint64_t(63)), elem_bytes = 0;
  switch(Atype) {
    case HYACIN_F64: elem_bytes = sizeof(double); break; case HYACIN_F32: elem_bytes = sizeof(float); break;
    case HYACIN_DD: elem_bytes = sizeof(double2); break; case HYACIN_QF: elem_bytes = sizeof(float4); break;
    case HYACIN_F64_COMPLEX: elem_bytes = sizeof(cuDoubleComplex); break; case HYACIN_F32_COMPLEX: elem_bytes = sizeof(cuComplex); break;
    case HYACIN_DD_COMPLEX: elem_bytes = sizeof(complex_double2); break; case HYACIN_QF_COMPLEX: elem_bytes = sizeof(complex_float4); break;
  }
  *dev_work_bytes = stride * uint64_t(lenK) * elem_bytes;
}

#ifndef NO_NCCL

inline void matrix_copy(cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, double* B, int32_t ldb)
{ double one = 1., zero = 0.; cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matrix_copy(cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, float* B, int32_t ldb)
{ float one = 1.f, zero = 0.f; cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matrix_copy(cublasHandle_t handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void matrix_copy(cublasHandle_t handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &one, A, lda, &zero, B, ldb, B, ldb); }

template <class T>
inline void allgatherv_1dcol(cublasHandle_t handle, int32_t M, int32_t iK, int32_t lenK, const int32_t* allK, T* A, int32_t lda, T* dev_work, ncclComm_t row_comm) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t maxK = std::reduce(allK, &allK[lenK], 0, [](int32_t i, int32_t j) { return std::max(i, j); });
  uint64_t stride = (uint64_t(M) * uint64_t(maxK) + uint64_t(63)) & (~uint64_t(63));
  constexpr uint64_t shifts = sizeof(T) / sizeof(int32_t);
  matrix_copy(handle, M, allK[iK], A, lda, &dev_work[int64_t(iK) * stride], M);
  ncclAllGather(&dev_work[int64_t(iK) * stride], dev_work, stride * shifts, ncclInt32, row_comm, stream);

  for (int32_t i = 0; i < lenK; ++i) {
    matrix_copy(handle, M, allK[i], &dev_work[int64_t(i) * stride], M, A, lda);
    A = &A[int64_t(allK[i]) * int64_t(lda)];
  }
}

extern "C" void hyacinXAllGatherV1Dcol(cublasHandle_t handle, int32_t M, int32_t iK, int32_t lenK, const int32_t* allK, hyacinPrecision_t Atype, void* A, int32_t lda, void* dev_work, ncclComm_t row_comm) {
  switch(Atype) {
    case HYACIN_F64: allgatherv_1dcol(handle, M, iK, lenK, allK, (double*)A, lda, (double*)dev_work, row_comm); break;
    case HYACIN_F32: allgatherv_1dcol(handle, M, iK, lenK, allK, (float*)A, lda, (float*)dev_work, row_comm); break;
    case HYACIN_F64_COMPLEX: allgatherv_1dcol(handle, M, iK, lenK, allK, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)dev_work, row_comm); break;
    case HYACIN_F32_COMPLEX: allgatherv_1dcol(handle, M, iK, lenK, allK, (cuComplex*)A, lda, (cuComplex*)dev_work, row_comm); break;
    default: break;
  }
}

#endif
