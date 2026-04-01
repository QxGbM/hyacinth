
#include <common.hpp>
#include <iostream>
#include <chrono>

template <class T> inline void run(char prec, int64_t gM, int64_t N, int64_t K, int64_t mb, double epi, int32_t grid_row, int32_t tile_m, ncclUniqueId id, const std::string& file, const std::string& out) {
  int64_t lM = mb * (gM / (mb * tile_m));
  lM += std::max(int64_t(0), std::min(mb, gM - lM * tile_m - mb * grid_row));

  std::vector<T> matA(lM * N);
  if (!file.empty())
    matrix_from_row_major_csv(gM, N, mb, 512, matA.data(), lM, file, grid_row, 0, tile_m, 1);
  else
    matrix_generator<T>(gM, N).generate_block(1., mb, 512, &matA[0], lM, grid_row, 0, tile_m, 1);

  T* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), lM * N * sizeof(T));
  cudaMalloc((void**)(&d_V), K * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);

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
  ncclCommInitRank(&comm, tile_m, id, grid_row);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int32_t* d_barrier = nullptr;
  if (time_kernel) {
    cudaMalloc((void**)(&d_barrier), sizeof(int32_t));
    cudaMemset(d_barrier, 0xDEADBEEF, sizeof(int32_t));
    svd_fit_transform_1dr(stream, cublasH, cusolverH, params, comm, epi, lM, gM, N, K, d_A, lM, d_V, N, N);
    cudaMemcpy(d_A, matA.data(), lM * N * sizeof(T), cudaMemcpyHostToDevice);
    ncclAllReduce(d_barrier, d_barrier, 1, ncclInt32, ncclMin, comm, stream);
    cudaStreamSynchronize(stream);
  }
  cudaEventRecord(start, stream);

  int32_t rank = svd_fit_transform_1dr(stream, cublasH, cusolverH, params, comm, epi, lM, gM, N, K, d_A, lM, d_V, N, N);

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

  std::vector<T> matU(lM * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, lM * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_V);

  double err = check_answer_svd(lM, N, rank, &matU[0], lM, &matV[0], N, &matA[0], lM);
  std::chrono::duration<double, std::milli> host_wtime = host_end - host_start;
  double duration = time_kernel ? double(milliseconds) : host_wtime.count();
  int64_t flops = ((int64_t(gM) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(gM) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / double(duration);

  printf("%c-SVD#%d,%ld,%ld,%.1le,%.12le,%d,%lf,%lf\n", prec, grid_row, gM, N, epi, err, rank, duration, gflops);

  if (!out.empty() && grid_row == 0)
    write_matrix_to_csv(rank, N, &matV[0], K, out);
}

int32_t main(int32_t argc, char* argv[]) {
  char prec = 'D'; std::string file, out;
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
    else if (std::strncmp(argv[i], "out=", 4) == 0) { out.resize(std::strlen(argv[i])); std::sscanf(argv[i], "out=%s", out.data()); }
    else { std::cerr << "Ignored parameter: " << argv[i] << std::endl; }
  }
  N = std::min(gM, N); K = std::min(N, K);

  int32_t world_rank, world_size, local_rank; ncclUniqueId id;
  __bootstrap(world_rank, world_size, local_rank, id);

  int32_t device_count = 0; cudaGetDeviceCount(&device_count);
  auto cu_err = cudaSetDevice(1 < device_count ? local_rank : 0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  switch(prec) {
    case 'D': run<double>(prec, gM, N, K, mb, epi, world_rank, world_size, id, file, out); break;
    case 'S': run<float>(prec, gM, N, K, mb, epi, world_rank, world_size, id, file, out); break;
    case 'Z': run<std::complex<double>>(prec, gM, N, K, mb, epi, world_rank, world_size, id, file, out); break;
    case 'C': run<std::complex<float>>(prec, gM, N, K, mb, epi, world_rank, world_size, id, file, out); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
