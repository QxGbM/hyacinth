
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T> inline void run(char prec, int64_t M, int64_t gN, int64_t K, int64_t nb, double epi, int32_t grid_col, int32_t tile_n, ncclUniqueId id, const std::string& file) {
  int64_t gK = K * tile_n;
  int64_t lN = nb * (gN / (nb * tile_n));
  lN += std::max(int64_t(0), std::min(nb, gN - lN * tile_n - nb * grid_col));

  std::vector<T> matA(M * lN);
  if (!file.empty())
    matrix_from_row_major_csv(M, gN, 512, nb, matA.data(), M, file, 0, grid_col, 1, tile_n);
  else
    matrix_generator<T>(M, gN).generate_block(1., 512, nb, &matA[0], M, 0, grid_col, 1, tile_n);

  T* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), M * std::max(gK, lN) * sizeof(T));
  cudaMalloc((void**)(&d_V), K * lN * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * lN * sizeof(T), cudaMemcpyHostToDevice);

  /* Timed region start */
  auto host_start = std::chrono::high_resolution_clock::now();

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;
  cusolverDnParams_t params;
  ncclComm_t comm;

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);
  cusolverDnCreateParams(&params);
  ncclCommInitRank(&comm, tile_n, id, grid_col);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int32_t* d_barrier = nullptr;
  int32_t r1, r2, N2, offset;
  if (time_kernel) {
    cudaMalloc((void**)(&d_barrier), sizeof(int32_t));
    cudaMemset(d_barrier, 0xDEADBEEF, sizeof(int32_t));
    r1 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V, lN, lN);
    std::tie(N2, offset) = allgatherv_1dc(stream, comm, M, r1, d_A, M);
    r2 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N2, K, d_A, M, d_V, lN, lN, r1, offset);
    cudaMemcpy(d_A, matA.data(), M * lN * sizeof(T), cudaMemcpyHostToDevice);
    ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream);
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(start, stream);

  r1 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, lN, K, d_A, M, d_V, lN, lN);
  std::tie(N2, offset) = allgatherv_1dc(stream, comm, M, r1, d_A, M);
  r2 = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N2, K, d_A, M, d_V, lN, lN, r1, offset);

  if (time_kernel)
    ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream);
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);

  if (time_kernel)
    cudaFree(d_barrier);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
  ncclCommDestroy(comm);

  /* Timed region end */
  auto host_end = std::chrono::high_resolution_clock::now();

  std::vector<T> matU(M * K), matV(K * lN);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * lN * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_V);

  double err = check_answer_svd(M, lN, r2, &matU[0], M, &matV[0], lN, &matA[0], M);
  std::chrono::duration<double, std::milli> host_wtime = host_end - host_start;
  double duration = time_kernel ? double(milliseconds) : host_wtime.count();
  int64_t flops = ((int64_t(M) + int64_t(gN)) * int64_t(r2) * int64_t(2)) + (int64_t(M) * int64_t(gN) * int64_t(r2) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / double(duration);

  printf("%c-SVD#%d,%ld,%ld,%.1le,%.12le,%d,%d,%lf,%lf\n", prec, grid_col, M, gN, epi, err, r1, r2, duration, gflops);
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

  switch(prec) {
    case 'D': run<double>(prec, M, gN, K, nb, epi, world_rank, world_size, id, file); break;
    case 'S': run<float>(prec, M, gN, K, nb, epi, world_rank, world_size, id, file); break;
    case 'Z': run<std::complex<double>>(prec, M, gN, K, nb, epi, world_rank, world_size, id, file); break;
    case 'C': run<std::complex<float>>(prec, M, gN, K, nb, epi, world_rank, world_size, id, file); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
