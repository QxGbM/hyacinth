
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

  int64_t gM = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(gM, N);

  int64_t K = 3 < argc ? std::atoi(argv[3]) : N;
  double epi = 4 < argc ? std::atof(argv[4]) : 1.e-12;
  double omega = 5 < argc ? std::atof(argv[5]) : 1.;
  K = std::min(N, K);

  int64_t mb = 6 < argc ? std::atoi(argv[6]) : 512;
  int64_t lM = mb * (gM / (mb * mpi_size));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * mpi_size - mb * mpi_rank));

  std::vector<double> matA(lM * N);
  for (int64_t i = mpi_rank * mb, y = 0; i < gM; i += mpi_size * mb) {
    make_2D_oscillatory(omega, i, 0, std::min(gM - i, mb), N, &matA[y], lM);
    y += mb;
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

  double* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), lM * N * sizeof(double));
  cudaMalloc((void**)(&d_V), K * N * sizeof(double));
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(double), cudaMemcpyHostToDevice);

  utv_factorize_phase1_1dr(stream, cublasH, cusolverH, params, epi, lM, gM, N, K, d_A, lM, d_V, K, comm);
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(double), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  int32_t rank = utv_factorize_phase1_1dr(stream, cublasH, cusolverH, params, epi, lM, gM, N, K, d_A, lM, d_V, K, comm);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  std::vector<double> matU(lM * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, lM * K * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(double), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(lM, N, rank, &matU[0], lM, &matV[0], K, &matA[0], lM);
  MPI_Allreduce(MPI_IN_PLACE, &ret, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double err = std::sqrt(ret.first / ret.second);

  double milliseconds = 1.e3 * (end - start);
  int64_t flops = ((int64_t(gM) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(gM) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "D-UTVK," << gM << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V);
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
