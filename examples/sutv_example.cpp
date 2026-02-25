
#include <hyacin.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <complex>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_2D_oscillatory(double w, int32_t iA, int32_t jA, int32_t M, int32_t N, float* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + jA + height);
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + iA);
      double d = std::abs(vi + std::conj(vj));
      A[i + j * lda] = float(std::cos(w * d) / d);
    }
  }
}

double check_answer(int32_t M, int32_t N, int32_t rank, const float* U, int32_t ldu, const float* V, int32_t ldv, const float* B, int32_t ldb) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();
  std::vector<float> matQ(M * N, 0.);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, 1.f, U, ldu, V, ldv, 0.f, &matQ[0], M);

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) {
      err += std::norm(matQ[i + j * M] - B[i + j * ldb]);
      nrm += std::norm(B[i + j * ldb]);
  }
  return std::sqrt(err / nrm);
}

int32_t utv_factorize(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, double epi, int32_t M, int32_t N, float* A, int32_t lda, float* V, int32_t ldv) {
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, HYACIN_F32, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, N, p, umax, HYACIN_F32, A, lda, (int32_t*)jpiv, HYACIN_F32, V, ldv, precC, dev_work, pinned_work, alg);

  cusolverDnParams_t params; cusolverDnCreateParams(&params);
  uint64_t dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, HYACIN_F32, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaDeviceSynchronize(); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaDeviceSynchronize(); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, dev_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, HYACIN_F32, dev_work_bytes_new, dev_work, pinned_work_bytes_new, pinned_work);

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

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-6;
  double omega = 4 < argc ? std::atof(argv[4]) : 1.;

  std::vector<float> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, 0, 0, M, N, &matA[0], M);

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(float));
  cudaMalloc((void**)(&d_V), N * N * sizeof(float));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  utv_factorize(stream, cublasH, cusolverH, epi, M, N, d_A, M, d_V, N);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = utv_factorize(stream, cublasH, cusolverH, epi, M, N, d_A, M, d_V, N);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<float> matU(M * N), matV(N * N);
  cudaMemcpy(matU.data(), d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  double err = check_answer(M, N, rank, &matU[0], M, &matV[0], N, &matA[0], M);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(M) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(M) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "S-UTKV," << M << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
