
#include <hyacin.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <complex>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_2D_oscillatory(double w, int32_t om, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(-(j + height));
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + om);
      double d = std::abs(vi - vj);
      A[i + j * lda] = std::cos(w * d) / d;
    }
  }
}

std::pair<double, double> check_answer(int32_t M, int32_t N, int32_t rank, const double* U, int32_t ldu, const double* V, int32_t ldv, const double* B, int32_t ldb) {
  if (rank <= 0)
    return std::make_pair(0., 0.);
  std::vector<double> matQ(M * N, 0.);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, M, N, rank, 1., U, ldu, V, ldv, 0., &matQ[0], M);

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) {
      err += std::norm(matQ[i + j * M] - B[i + j * ldb]);
      nrm += std::norm(B[i + j * ldb]);
  }
  return std::make_pair(err, nrm);
}

int32_t utv_factorize(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, double epi, int32_t M, int64_t gM, int32_t N, const double* A, int32_t lda, double* UT, int32_t ldu, double* V, int32_t ldv, MPI_Comm comm) {
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, gM, 6, &umax, HYACIN_F64, &precC, &alg);
  hyacinXcpqrk1Dcol_bufferSize(M, gM, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk1Dcol(cublasH, 'J', epi, M, gM, N, N, p, umax, HYACIN_F64, A, lda, (int32_t*)jpiv, HYACIN_F64, V, ldv, precC, dev_work, pinned_work, alg, comm);

  cusolverDnParams_t params; cusolverDnCreateParams(&params);
  uint64_t dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, N, HYACIN_F64, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaDeviceSynchronize(); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaDeviceSynchronize(); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, dev_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, UT, ldu, HYACIN_F64, dev_work_bytes_new, dev_work, pinned_work_bytes_new, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

int32_t main(int32_t argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int32_t mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  auto cu_err = cudaSetDevice(mpi_rank);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t gM = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(gM, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  double omega = 4 < argc ? std::atof(argv[4]) : 1.;

  int64_t lM = (gM + mpi_size - 1) / mpi_size;
  int64_t lS = lM * mpi_rank; lM = std::min(lM, gM - lS);

  std::vector<double> matA(lM * N);
  make_2D_oscillatory(omega, lS, lM, N, &matA[0], lM);

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A = nullptr, *d_U = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), lM * N * sizeof(double));
  cudaMalloc((void**)(&d_U), lM * N * sizeof(double));
  cudaMalloc((void**)(&d_V), N * N * sizeof(double));
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(double), cudaMemcpyHostToDevice);
  utv_factorize(stream, cublasH, cusolverH, epi, lM, gM, N, d_A, lM, d_U, lM, d_V, N, MPI_COMM_WORLD);

  cudaEventRecord(start, stream);
  int32_t rank = utv_factorize(stream, cublasH, cusolverH, epi, lM, gM, N, d_A, lM, d_U, lM, d_V, N, MPI_COMM_WORLD);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<double> matU(lM * N), matV(N * N);
  cudaMemcpy(matU.data(), d_U, lM * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, N * N * sizeof(double), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer(lM, N, rank, &matU[0], lM, &matV[0], N, &matA[0], lM);
  MPI_Allreduce(MPI_IN_PLACE, &ret, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double err = std::sqrt(ret.first / ret.second);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(gM) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(gM) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "D-UTVK," << gM << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_U);
  cudaFree(d_V);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;

  MPI_Finalize();
  return 0;
}
