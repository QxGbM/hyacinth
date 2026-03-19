
#include <common.hpp>
#include <iostream>
#include <mpi.h>

template <class T> inline void run(char prec, int64_t M, int64_t gN, int64_t K, int64_t nb, double epi, const std::string& file) {
  int32_t mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  ncclUniqueId id; ncclComm_t comm;
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void*)&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, mpi_size, id, mpi_rank);

  int64_t gK = K * mpi_size;
  int64_t lN = nb * (gN / (nb * mpi_size));
  lN += std::max(int64_t(0), std::min(nb, gN - lN * mpi_size - nb * mpi_rank));

  std::vector<T> matA(M * lN);
  if (!file.empty())
    matrix_from_row_major_csv(M, gN, 512, nb, matA.data(), M, file, 0, mpi_rank, 1, mpi_size);
  else for (int64_t j = mpi_rank * nb, x = 0; j < gN; j = mpi_rank * nb + mpi_size * (x += nb))
    make_2D_oscillatory(1., 0, j, M, std::min(gN - j, nb), &matA[x * M], M);

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

  T* d_A = nullptr, *d_V1 = nullptr, *d_V2 = nullptr;
  cudaMalloc((void**)(&d_A), M * std::max(gK, lN) * sizeof(T));
  cudaMalloc((void**)(&d_V1), K * lN * sizeof(T));
  cudaMalloc((void**)(&d_V2), K * gK * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(T), cudaMemcpyHostToDevice);

  int32_t r1, r2, N2, offset;
  r1 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(N2, offset) = allgatherv_1dc(stream, M, r1, d_A, M, comm);
  r2 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N2, K, d_A, M, d_V2, K);
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(T), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  r1 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(N2, offset) = allgatherv_1dc(stream, M, r1, d_A, M, comm);
  r2 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N2, K, d_A, M, d_V2, K);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  std::vector<T> matU(M * K), matV1(K * lN), matV2(K * gK);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV1.data(), d_V1, K * lN * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV2.data(), d_V2, K * gK * sizeof(T), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(M, lN, r2, r1, &matU[0], M, &matV2[int64_t(offset) * int64_t(K)], K, &matV1[0], K, &matA[0], M);
  MPI_Allreduce(MPI_IN_PLACE, &ret, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double err = std::sqrt(ret.first / ret.second);

  double milliseconds = 1.e3 * (end - start);
  int64_t flops = ((int64_t(M) + int64_t(gN)) * int64_t(r2) * int64_t(2)) + (int64_t(M) * int64_t(gN) * int64_t(r2) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << prec << "-UTVK," << M << "," << gN << "," << epi << "," << err << "," << r1 << "," << r2 << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V1);
  cudaFree(d_V2);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
  ncclCommDestroy(comm);
}

int32_t main(int32_t argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  char prec = 'D'; std::string file;
  int64_t M = 2048, gN = 2048, K = 1500, nb = 512;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &M); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &gN); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "nb=", 3) == 0) { std::sscanf(argv[i], "nb=%ld", &nb); }
    else if (std::strncmp(argv[i], "file=", 5) == 0) { file.resize(std::strlen(argv[i])); std::sscanf(argv[i], "file=%s", file.data()); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  gN = std::min(M, gN); K = std::min(gN, K);

  int32_t local_rank = 0;
  MPI_Comm shmcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_free(&shmcomm);

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; MPI_Finalize(); return -1; }

  switch(prec) {
    case 'D': run<double>(prec, M, gN, K, nb, epi, file); break;
    case 'S': run<float>(prec, M, gN, K, nb, epi, file); break;
    case 'Z': run<std::complex<double>>(prec, M, gN, K, nb, epi, file); break;
    case 'C': run<std::complex<float>>(prec, M, gN, K, nb, epi, file); break;
    default: break;
  }
  MPI_Finalize();

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
