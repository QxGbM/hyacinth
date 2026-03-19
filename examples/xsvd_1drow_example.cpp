
#include <common.hpp>
#include <iostream>
#include <mpi.h>

template <class T> inline void run(char prec, int64_t gM, int64_t N, int64_t K, int64_t mb, double epi, const std::string& file) {
  int32_t mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  ncclUniqueId id; ncclComm_t comm;
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void*)&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, mpi_size, id, mpi_rank);

  int64_t lM = mb * (gM / (mb * mpi_size));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * mpi_size - mb * mpi_rank));

  std::vector<T> matA(lM * N);
  if (!file.empty())
    matrix_from_row_major_csv(gM, N, mb, N, matA.data(), lM, file, mpi_rank, 0, mpi_size, 1);
  else for (int64_t i = mpi_rank * mb, y = 0; i < gM; i = mpi_rank * mb + mpi_size * (y += mb))
    make_2D_oscillatory(1., i, 0, std::min(gM - i, mb), N, &matA[y], lM);

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

  T* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), lM * N * sizeof(T));
  cudaMalloc((void**)(&d_V), K * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);

  svd_fit_transform_1dr(stream, cublasH, cusolverH, params, epi, lM, gM, N, K, d_A, lM, d_V, K, comm);
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  int32_t rank = svd_fit_transform_1dr(stream, cublasH, cusolverH, params, epi, lM, gM, N, K, d_A, lM, d_V, K, comm);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  std::vector<T> matU(lM * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, lM * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(T), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(lM, N, rank, &matU[0], lM, &matV[0], K, &matA[0], lM);
  MPI_Allreduce(MPI_IN_PLACE, &ret, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double err = std::sqrt(ret.first / ret.second);

  double milliseconds = 1.e3 * (end - start);
  int64_t flops = ((int64_t(gM) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(gM) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << prec << "-UTVK," << gM << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
  ncclCommDestroy(comm);
}

int32_t main(int32_t argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  char prec = 'D'; std::string file;
  int64_t gM = 2048, N = 2048, K = 2048, mb = 512;
  double epi = 1.e-12;

  for (int32_t i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "M=", 2) == 0) { std::sscanf(argv[i], "M=%ld", &gM); }
    else if (std::strncmp(argv[i], "N=", 2) == 0) { std::sscanf(argv[i], "N=%ld", &N); }
    else if (std::strncmp(argv[i], "K=", 2) == 0) { std::sscanf(argv[i], "K=%ld", &K); }
    else if (std::strncmp(argv[i], "data=", 5) == 0) { std::sscanf(argv[i], "data=%c", &prec); }
    else if (std::strncmp(argv[i], "epi=", 4) == 0) { std::sscanf(argv[i], "epi=%lf", &epi); }
    else if (std::strncmp(argv[i], "mb=", 3) == 0) { std::sscanf(argv[i], "mb=%ld", &mb); }
    else if (std::strncmp(argv[i], "file=", 5) == 0) { file.resize(std::strlen(argv[i])); std::sscanf(argv[i], "file=%s", file.data()); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(gM, N); K = std::min(N, K);

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
    case 'D': run<double>(prec, gM, N, K, mb, epi, file); break;
    case 'S': run<float>(prec, gM, N, K, mb, epi, file); break;
    case 'Z': run<std::complex<double>>(prec, gM, N, K, mb, epi, file); break;
    case 'C': run<std::complex<float>>(prec, gM, N, K, mb, epi, file); break;
    default: break;
  }
  MPI_Finalize();

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
