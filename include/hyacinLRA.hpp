
#pragma once
#include <hyacin.h>

namespace hyacinLRA {

  int32_t interp_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, int32_t* jpvt, double* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t interp_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, int32_t* jpvt, float* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t interp_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, int32_t* jpvt, __half* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t interp_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, int32_t* jpvt, cuDoubleComplex* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t interp_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, int32_t* jpvt, cuComplex* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t interp_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, int32_t* jpvt, __half2* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);

  int32_t tsvd_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, double* S, double* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, float* S, float* V, int32_t ldv,  int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, __half* S, __half* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, double* S, cuDoubleComplex* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, float* S, cuComplex* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, __half* S, __half2* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);

  int32_t tsvd_fit_segmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, double* S, double* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit_segmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, float* S, float* V, int32_t ldv,  int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit_segmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, __half* S, __half* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit_segmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, double* S, cuDoubleComplex* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit_segmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, float* S, cuComplex* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);
  int32_t tsvd_fit_segmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, __half* S, __half2* V, int32_t ldv, int64_t globalM = 0ll, int32_t batchIter = 8192);

  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t rank = 0, const double* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t rank = 0, const float* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t rank = 0, const __half* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t rank = 0, const cuDoubleComplex* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t rank = 0, const cuComplex* V = nullptr, int32_t ldv = 0);
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t rank = 0, const __half2* V = nullptr, int32_t ldv = 0);

};
