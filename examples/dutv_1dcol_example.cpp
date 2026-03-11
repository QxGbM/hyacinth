
#include <common.hpp>
#include <iostream>

int32_t main(int32_t argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int32_t mpi_rank = 0, local_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  MPI_Comm shmcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_free(&shmcomm);

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; MPI_Finalize(); return -1; }

  ncclUniqueId id; ncclComm_t comm;
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, mpi_size, id, mpi_rank);

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t gN = 2 < argc ? std::atoi(argv[2]) : 2048;
  gN = std::min(M, gN);

  int64_t K = 3 < argc ? std::atoi(argv[3]) : 1500;
  double epi = 4 < argc ? std::atof(argv[4]) : 1.e-12;
  double omega = 5 < argc ? std::atof(argv[5]) : 1.;
  K = std::min(gN, K);
  int64_t gK = K * mpi_size;

  int64_t nb = 6 < argc ? std::atoi(argv[6]) : 512;
  int64_t lN = nb * (gN / (nb * mpi_size));
  lN += std::max(int64_t(0), std::min(nb, gN - lN * mpi_size - nb * mpi_rank));

  std::vector<double> matA(M * lN);
  for (int64_t j = mpi_rank * nb, x = 0; j < gN; j += mpi_size * nb) {
    make_2D_oscillatory(omega, 0, j, M, std::min(gN - j, nb), &matA[x * M], M);
    x += nb;
  }

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
  std::tie(r2, offset) = utv_factorize_phase2_1dc(stream, cublasH, cusolverH, params, epi, M, r1, K, d_A, M, d_V2, K, comm, MPI_COMM_WORLD);
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(double), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  r1 = utv_factorize_phase1(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(r2, offset) = utv_factorize_phase2_1dc(stream, cublasH, cusolverH, params, epi, M, r1, K, d_A, M, d_V2, K, comm, MPI_COMM_WORLD);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  std::vector<double> matU(M * K), matV1(K * lN), matV2(K * gK);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV1.data(), d_V1, K * lN * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV2.data(), d_V2, K * gK * sizeof(double), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(M, lN, r2, r1, &matU[0], M, &matV2[int64_t(offset) * int64_t(K)], K, &matV1[0], K, &matA[0], M);
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
