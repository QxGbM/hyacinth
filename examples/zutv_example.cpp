
#include <common.hpp>
#include <iostream>

void make_2D_oscillatory(double w, int32_t iA, int32_t jA, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + jA + height);
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + iA);
      double d = std::abs(vi + std::conj(vj));
      A[i + j * lda] = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
  }
}

int32_t utv_factorize(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, std::complex<double>* A, int32_t lda, std::complex<double>* V, int32_t ldv) {
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, HYACIN_F64_COMPLEX, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, K, p, umax, HYACIN_F64_COMPLEX, A, lda, (int32_t*)jpiv, HYACIN_F64_COMPLEX, V, ldv, precC, dev_work, pinned_work, alg);

  uint64_t dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, HYACIN_F64_COMPLEX, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, HYACIN_F64_COMPLEX, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(M, N);

  int64_t K = 3 < argc ? std::atoi(argv[3]) : N;
  double epi = 4 < argc ? std::atof(argv[4]) : 1.e-12;
  double omega = 5 < argc ? std::atof(argv[5]) : 1.;
  K = std::min(K, N);

  std::vector<std::complex<double>> matA(M * N);
  make_2D_oscillatory(omega, 0, 0, M, N, &matA[0], M);

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

  std::complex<double>* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&d_V), K * N * sizeof(std::complex<double>));
  cudaMemcpy(d_A, &matA[0], M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  utv_factorize(stream, cublasH, cusolverH, params, epi, M, N, K, d_A, M, d_V, K);
  cudaMemcpy(d_A, &matA[0], M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = utv_factorize(stream, cublasH, cusolverH, params, epi, M, N, K, d_A, M, d_V, K);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<std::complex<double>> matU(M * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_zsvd(M, N, rank, &matU[0], M, &matV[0], K, &matA[0], M);
  double err = std::sqrt(ret.first / ret.second);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(M) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(M) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "Z-UTVK," << M << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
