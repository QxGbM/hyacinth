
#include <common.hpp>
#include <iostream>

template <class T> inline void run(char prec, int64_t M, int64_t gN, int64_t K, int64_t nb, double epi, int32_t grid_col, int32_t tile_n, ncclComm_t comm, const std::string& file) {
  int64_t gK = K * tile_n;
  int64_t lN = nb * (gN / (nb * tile_n));
  lN += std::max(int64_t(0), std::min(nb, gN - lN * tile_n - nb * grid_col));

  std::vector<T> matA(M * lN);
  if (!file.empty())
    matrix_from_row_major_csv(M, gN, 512, nb, matA.data(), M, file, 0, grid_col, 1, tile_n);
  else for (int64_t j = grid_col * nb, x = 0; j < gN; j = grid_col * nb + tile_n * (x += nb))
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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int32_t* d_barrier = nullptr;
  T* d_A = nullptr, *d_V1 = nullptr, *d_V2 = nullptr;
  cudaMalloc((void**)(&d_barrier), sizeof(double2));
  cudaMalloc((void**)(&d_A), M * std::max(gK, lN) * sizeof(T));
  cudaMalloc((void**)(&d_V1), K * lN * sizeof(T));
  cudaMalloc((void**)(&d_V2), K * gK * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_barrier, 0xDEADBEEF, sizeof(double2));

  int32_t r1, r2, N2, offset;
  r1 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(N2, offset) = allgatherv_1dc(stream, M, r1, d_A, M, comm);
  r2 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N2, K, d_A, M, d_V2, K);
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(T), cudaMemcpyHostToDevice);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream);
  cudaStreamSynchronize(stream);
  cudaEventRecord(start, stream);

  r1 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V1, K);
  std::tie(N2, offset) = allgatherv_1dc(stream, M, r1, d_A, M, comm);
  r2 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N2, K, d_A, M, d_V2, K);

  ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream);
  cudaStreamSynchronize(stream);
  cudaEventRecord(stop, stream);

  std::vector<T> matU(M * K), matV1(K * lN), matV2(K * gK);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV1.data(), d_V1, K * lN * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV2.data(), d_V2, K * gK * sizeof(T), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(M, lN, r2, r1, &matU[0], M, &matV2[int64_t(offset) * int64_t(K)], K, &matV1[0], K, &matA[0], M);
  cudaMemcpy(d_barrier, &ret, sizeof(double2), cudaMemcpyHostToDevice);
  ncclAllReduce(d_barrier, d_barrier, 2, ncclDouble, ncclSum, comm, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(&ret, d_barrier, sizeof(double2), cudaMemcpyDeviceToHost);
  double err = std::sqrt(ret.first / ret.second);

  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(M) + int64_t(gN)) * int64_t(r2) * int64_t(2)) + (int64_t(M) * int64_t(gN) * int64_t(r2) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / double(milliseconds);

  printf("%c-SVD,%ld,%ld,%.1le,%.12le,%d,%d,%f,%lf\n", prec, M, gN, epi, err, r1, r2, milliseconds, gflops);

  cudaFree(d_barrier);
  cudaFree(d_A);
  cudaFree(d_V1);
  cudaFree(d_V2);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
}

int32_t main(int32_t argc, char* argv[]) {
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

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  __bootstrap(world_rank, world_size, local_rank, id);

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  ncclComm_t comm;
  ncclCommInitRank(&comm, world_size, id, world_rank);

  switch(prec) {
    case 'D': run<double>(prec, M, gN, K, nb, epi, world_rank, world_size, comm, file); break;
    case 'S': run<float>(prec, M, gN, K, nb, epi, world_rank, world_size, comm, file); break;
    case 'Z': run<std::complex<double>>(prec, M, gN, K, nb, epi, world_rank, world_size, comm, file); break;
    case 'C': run<std::complex<float>>(prec, M, gN, K, nb, epi, world_rank, world_size, comm, file); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;

  ncclCommDestroy(comm);
  return 0;
}
