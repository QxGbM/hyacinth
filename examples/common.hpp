
#include <hyacin.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

double check_answer_dlra(int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, const int32_t* jpiv, const double* R, int32_t ldr) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<double> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  for (int32_t i = 0; i < N; ++i) {
    std::copy_n(&A[int64_t(i) * int64_t(lda)], M, &matB[int64_t(i) * int64_t(M)]);
    std::copy_n(&R[int64_t(i) * int64_t(ldr)], rank, &matR[int64_t(jpiv[i] - 1) * int64_t(rank)]);
    if (i < rank)
      std::copy_n(&A[int64_t(jpiv[i] - 1) * int64_t(lda)], M, &matC[int64_t(i) * int64_t(M)]);
  }

  double one = 1., minus_one = -1.;
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, N, one, R, ldr, &matR[0], rank);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, minus_one, &matC[0], M, &matR[0], rank, one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::sqrt(err / nrm);
}

double check_answer_slra(int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, const int32_t* jpiv, const float* R, int32_t ldr) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<float> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  for (int32_t i = 0; i < N; ++i) {
    std::copy_n(&A[int64_t(i) * int64_t(lda)], M, &matB[int64_t(i) * int64_t(M)]);
    std::copy_n(&R[int64_t(i) * int64_t(ldr)], rank, &matR[int64_t(jpiv[i] - 1) * int64_t(rank)]);
    if (i < rank)
      std::copy_n(&A[int64_t(jpiv[i] - 1) * int64_t(lda)], M, &matC[int64_t(i) * int64_t(M)]);
  }

  float one = 1.f, minus_one = -1.f;
  cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, N, one, R, ldr, &matR[0], rank);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, minus_one, &matC[0], M, &matR[0], rank, one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::sqrt(err / nrm);
}

double check_answer_zlra(int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const int32_t* jpiv, const std::complex<double>* R, int32_t ldr) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<std::complex<double>> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  for (int32_t i = 0; i < N; ++i) {
    std::copy_n(&A[int64_t(i) * int64_t(lda)], M, &matB[int64_t(i) * int64_t(M)]);
    std::copy_n(&R[int64_t(i) * int64_t(ldr)], rank, &matR[int64_t(jpiv[i] - 1) * int64_t(rank)]);
    if (i < rank)
      std::copy_n(&A[int64_t(jpiv[i] - 1) * int64_t(lda)], M, &matC[int64_t(i) * int64_t(M)]);
  }

  std::complex<double> one(1., 0.), minus_one(-1., 0.);
  cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, N, &one, R, ldr, &matR[0], rank);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, &minus_one, &matC[0], M, &matR[0], rank, &one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::sqrt(err / nrm);
}

double check_answer_clra(int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, const int32_t* jpiv, const std::complex<float>* R, int32_t ldr) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<std::complex<float>> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  for (int32_t i = 0; i < N; ++i) {
    std::copy_n(&A[int64_t(i) * int64_t(lda)], M, &matB[int64_t(i) * int64_t(M)]);
    std::copy_n(&R[int64_t(i) * int64_t(ldr)], rank, &matR[int64_t(jpiv[i] - 1) * int64_t(rank)]);
    if (i < rank)
      std::copy_n(&A[int64_t(jpiv[i] - 1) * int64_t(lda)], M, &matC[int64_t(i) * int64_t(M)]);
  }

  std::complex<float> one(1.f, 0.f), minus_one(-1.f, 0.f);
  cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, N, &one, R, ldr, &matR[0], rank);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, &minus_one, &matC[0], M, &matR[0], rank, &one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::sqrt(err / nrm);
}

std::pair<double, double> check_answer_dsvd(int32_t M, int32_t N, int32_t rank, const double* U, int32_t ldu, const double* V, int32_t ldv, const double* B, int32_t ldb) {
  if (rank <= 0)
    return std::make_pair(0., 0.);
  std::vector<double> matB(M * N);
  for (int32_t i = 0; i < N; ++i)
    std::copy_n(&B[int64_t(i) * int64_t(ldb)], M, &matB[int64_t(i) * int64_t(M)]);

  double one = 1., minus_one = -1.;
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, minus_one, U, ldu, V, ldv, one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::make_pair(err, nrm);
}

std::pair<double, double> check_answer_ssvd(int32_t M, int32_t N, int32_t rank, const float* U, int32_t ldu, const float* V, int32_t ldv, const float* B, int32_t ldb) {
  if (rank <= 0)
    return std::make_pair(0., 0.);
  std::vector<float> matB(M * N);
  for (int32_t i = 0; i < N; ++i)
    std::copy_n(&B[int64_t(i) * int64_t(ldb)], M, &matB[int64_t(i) * int64_t(M)]);

  float one = 1.f, minus_one = -1.f;
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, minus_one, U, ldu, V, ldv, one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::make_pair(err, nrm);
}

std::pair<double, double> check_answer_zsvd(int32_t M, int32_t N, int32_t rank, const std::complex<double>* U, int32_t ldu, const std::complex<double>* V, int32_t ldv, const std::complex<double>* B, int32_t ldb) {
  if (rank <= 0)
    return std::make_pair(0., 0.);
  std::vector<std::complex<double>> matB(M * N);
  for (int32_t i = 0; i < N; ++i)
    std::copy_n(&B[int64_t(i) * int64_t(ldb)], M, &matB[int64_t(i) * int64_t(M)]);

  std::complex<double> one(1., 0.), minus_one(-1., 0.);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, &minus_one, U, ldu, V, ldv, &one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::make_pair(err, nrm);
}

std::pair<double, double> check_answer_csvd(int32_t M, int32_t N, int32_t rank, const std::complex<float>* U, int32_t ldu, const std::complex<float>* V, int32_t ldv, const std::complex<float>* B, int32_t ldb) {
  if (rank <= 0)
    return std::make_pair(0., 0.);
  std::vector<std::complex<float>> matB(M * N);
  for (int32_t i = 0; i < N; ++i)
    std::copy_n(&B[int64_t(i) * int64_t(ldb)], M, &matB[int64_t(i) * int64_t(M)]);

  std::complex<float> one(1.f, 0.f), minus_one(-1.f, 0.f);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, &minus_one, U, ldu, V, ldv, &one, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::make_pair(err, nrm);
}
