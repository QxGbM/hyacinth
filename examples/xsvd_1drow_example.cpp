
#include <common.hpp>
#include <iostream>

template <class T> inline void run(char prec, int64_t gM, int64_t N, int64_t K, int64_t mb, double epi, int32_t grid_row, int32_t tile_m, ncclComm_t comm, const std::string& file) {
  int64_t lM = mb * (gM / (mb * tile_m));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));

  std::vector<T> matA(lM * N);
  if (!file.empty())
    matrix_from_row_major_csv(gM, N, mb, N, matA.data(), lM, file, grid_row, 0, tile_m, 1);
  else for (int64_t i = grid_row * mb, y = 0; i < gM; i = grid_row * mb + tile_m * (y += mb))
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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int32_t* d_barrier = nullptr;
  T* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_barrier), sizeof(double2));
  cudaMalloc((void**)(&d_A), lM * N * sizeof(T));
  cudaMalloc((void**)(&d_V), K * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_barrier, 0xDEADBEEF, sizeof(double2));

  svd_fit_transform_1dr(stream, cublasH, cusolverH, params, epi, lM, gM, N, K, d_A, lM, d_V, K, comm);
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMax, comm, stream);
  cudaStreamSynchronize(stream);
  cudaEventRecord(start, stream);

  int32_t rank = svd_fit_transform_1dr(stream, cublasH, cusolverH, params, epi, lM, gM, N, K, d_A, lM, d_V, K, comm);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMax, comm, stream);
  cudaStreamSynchronize(stream);
  cudaEventRecord(stop, stream);

  std::vector<T> matU(lM * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, lM * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(T), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(lM, N, rank, &matU[0], lM, &matV[0], K, &matA[0], lM);
  cudaMemcpy(d_barrier, &ret, sizeof(double2), cudaMemcpyHostToDevice);
  ncclAllReduce(d_barrier, d_barrier, 2, ncclDouble, ncclSum, comm, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(&ret, d_barrier, sizeof(double2), cudaMemcpyDeviceToHost);
  double err = std::sqrt(ret.first / ret.second);

  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(gM) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(gM) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / double(milliseconds);

  std::cout << prec << "-SVD," << gM << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_barrier);
  cudaFree(d_A);
  cudaFree(d_V);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
  ncclCommDestroy(comm);
}

#include <mpi.h>
#include <filesystem>

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D'; std::string file, id_path("id.out");
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
    else if (std::strncmp(argv[i], "ccl=", 4) == 0) { id_path.resize(std::strlen(argv[i])); std::sscanf(argv[i], "ccl=%s", id_path.data()); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(gM, N); K = std::min(N, K);

  MPI_Init(&argc, &argv);
  int32_t world_rank = 0, local_rank = 0, world_size = 1;
  MPI_Comm shmcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_free(&shmcomm);
  MPI_Finalize();

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  ncclUniqueId id = world_rank == 0 ? nccl_id_to_file(id_path) : nccl_id_from_file(id_path); ncclComm_t comm;
  ncclCommInitRank(&comm, world_size, id, world_rank);

  switch(prec) {
    case 'D': run<double>(prec, gM, N, K, mb, epi, world_rank, world_size, comm, file); break;
    case 'S': run<float>(prec, gM, N, K, mb, epi, world_rank, world_size, comm, file); break;
    case 'Z': run<std::complex<double>>(prec, gM, N, K, mb, epi, world_rank, world_size, comm, file); break;
    case 'C': run<std::complex<float>>(prec, gM, N, K, mb, epi, world_rank, world_size, comm, file); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;

  ncclCommDestroy(comm);
  if (world_rank == 0) { std::filesystem::remove(id_path); }
  return 0;
}
