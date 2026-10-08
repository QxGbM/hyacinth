
#pragma once
#include <hyacin.h>

namespace hyacinLRA {

  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, int32_t* jpvt, double* U, int32_t ldu, double* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, int32_t* jpvt, float* U, int32_t ldu, float* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);
  
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, int32_t* jpvt, __half* U, int32_t ldu, __half* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, int32_t* jpvt, cuDoubleComplex* U, int32_t ldu, cuDoubleComplex* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, int32_t* jpvt, cuComplex* U, int32_t ldu, cuComplex* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);
  
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, int32_t* jpvt, __half2* U, int32_t ldu, __half2* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, double* U, int32_t ldu, double* S, double* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, float* U, int32_t ldu, float* S, float* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);
  
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, __half* U, int32_t ldu, __half* S, __half* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* U, int32_t ldu, double* S, cuDoubleComplex* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, cuComplex* U, int32_t ldu, float* S, cuComplex* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);
  
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, __half2* U, int32_t ldu, __half* S, __half2* V, int32_t ldv,
    int64_t globalM = 0ll, int32_t batchIter = 8192, int32_t localMv = 0, int32_t Nv = 0, int32_t localVoffset = 0);

  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t rank = 0, const double* U = nullptr, int32_t ldu = 0, const double* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t rank = 0, const float* U = nullptr, int32_t ldu = 0, const float* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t rank = 0, const __half* U = nullptr, int32_t ldu = 0, const __half* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t rank = 0, const cuDoubleComplex* U = nullptr, int32_t ldu = 0, const cuDoubleComplex* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t rank = 0, const cuComplex* U = nullptr, int32_t ldu = 0, const cuComplex* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t rank = 0, const __half2* U = nullptr, int32_t ldu = 0, const __half2* V = nullptr, int32_t ldv = 0);

};
