
#include <hyacin.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <complex>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_2D_oscillatory(double w, int32_t iA, int32_t jA, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + jA + height);
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + iA);
      double d = std::abs(vi + std::conj(vj));
      A[i + j * lda] = std::cos(w * d) / d;
    }
  }
}

std::pair<double, double> check_answer(int32_t M, int32_t N, int32_t rank, const double* U, int32_t ldu, const double* V, int32_t ldv, const double* B, int32_t ldb) {
  if (rank <= 0)
    return std::make_pair(0., 0.);
  std::vector<double> matQ(M * N, 0.);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, 1., U, ldu, V, ldv, 0., &matQ[0], M);

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) {
      err += std::norm(matQ[i + j * M] - B[i + j * ldb]);
      nrm += std::norm(B[i + j * ldb]);
  }
  return std::make_pair(err, nrm);
}

int32_t utv_factorize_phase1(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, double* A, int32_t lda, double* V, int32_t ldv) {
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes, dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, HYACIN_F64, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, K, p, umax, HYACIN_F64, A, lda, (int32_t*)jpiv, HYACIN_F64, V, ldv, precC, dev_work, pinned_work, alg);

  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, HYACIN_F64, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, HYACIN_F64, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

std::pair<int32_t, int32_t> utv_factorize_phase2(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, double* A, int32_t lda, double* V, int32_t ldv, ncclComm_t comm, MPI_Comm mpi_comm) {
  int32_t mpi_rank, mpi_size; MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  std::vector<int32_t> ranks_arr(mpi_size);
  MPI_Allgather(&N, 1, MPI_INT32_T, ranks_arr.data(), 1, MPI_INT32_T, mpi_comm);
  int32_t rank_max = std::reduce(ranks_arr.begin(), ranks_arr.end(), 0, [](int32_t i, int32_t j) { return std::max(i, j); });
  int32_t v_offset = std::reduce(ranks_arr.begin(), ranks_arr.begin() + mpi_rank, 0, std::plus<int32_t>());
  N = std::reduce(ranks_arr.begin() + mpi_rank, ranks_arr.end(), v_offset, std::plus<int32_t>());

  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes, dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, HYACIN_F64, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);
  hyacinXAllGatherV1Dcol_bufferSize(M, rank_max, mpi_size, HYACIN_F64, &dev_work_bytes_new);
  dev_work_bytes = std::max(dev_work_bytes, dev_work_bytes_new);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  hyacinXAllGatherV1Dcol(cublasH, M, mpi_rank, mpi_size, ranks_arr.data(), HYACIN_F64, A, lda, dev_work, comm);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, K, p, umax, HYACIN_F64, A, lda, (int32_t*)jpiv, HYACIN_F64, V, ldv, precC, dev_work, pinned_work, alg);

  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, HYACIN_F64, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, HYACIN_F64, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return std::make_pair(rank, v_offset);
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

  ncclUniqueId id; ncclComm_t comm;
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, mpi_size, id, mpi_rank);

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t gN = 2 < argc ? std::atoi(argv[2]) : 2048;
  gN = std::min(M, gN);

  int64_t K = 3 < argc ? std::atoi(argv[3]) : 1500;
  double epi = 4 < argc ? std::atof(argv[4]) : 1.e-12;
  double omega = 5 < argc ? std::atof(argv[5]) : 1.;
  K = std::min(gN, K);

  int64_t lN = (gN + mpi_size - 1) / mpi_size;
  int64_t lS = lN * mpi_rank; lN = std::min(lN, gN - lS);
  int64_t gK = K * mpi_size;

  std::vector<double> matA(M * lN);
  make_2D_oscillatory(omega, 0, lS, M, lN, &matA[0], M);

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;
  cusolverDnParams_t params;

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);
  cusolverDnCreateParams(&params);

  double* d_A = nullptr, *d_V1 = nullptr, *d_V2 = nullptr;
  cudaMalloc((void**)(&d_A), M * std::max(gK, lN) * sizeof(double));
  cudaMalloc((void**)(&d_V1), K * lN * sizeof(double));
  cudaMalloc((void**)(&d_V2), K * gK * sizeof(double));
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(double), cudaMemcpyHostToDevice);

  int32_t r1, r2, offset;
  r1 = utv_factorize_phase1(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(r2, offset) = utv_factorize_phase2(stream, cublasH, cusolverH, params, epi, M, r1, K, d_A, M, d_V2, K, comm, MPI_COMM_WORLD);
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(double), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  r1 = utv_factorize_phase1(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(r2, offset) = utv_factorize_phase2(stream, cublasH, cusolverH, params, epi, M, r1, K, d_A, M, d_V2, K, comm, MPI_COMM_WORLD);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  std::vector<double> matU(M * K), matV1(K * lN), matV2(K * gK), matV(K * lN);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV1.data(), d_V1, K * lN * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV2.data(), d_V2, K * gK * sizeof(double), cudaMemcpyDeviceToHost);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r2, lN, r1, 1., &matV2[int64_t(offset) * int64_t(K)], K, &matV1[0], K, 0., &matV[0], K);
  std::pair<double, double> ret = check_answer(M, lN, r2, &matU[0], M, &matV[0], K, &matA[0], M);
  MPI_Allreduce(MPI_IN_PLACE, &ret, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double err = std::sqrt(ret.first / ret.second);

  double milliseconds = 1.e3 * (end - start);
  int64_t flops = ((int64_t(M) + int64_t(gN)) * int64_t(r2) * int64_t(2)) + (int64_t(M) * int64_t(gN) * int64_t(r2) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "D-UTVK," << M << "," << gN << "," << epi << "," << err << "," << r1 << "," << r2 << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V1);
  cudaFree(d_V2);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
  ncclCommDestroy(comm);
  MPI_Finalize();

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
