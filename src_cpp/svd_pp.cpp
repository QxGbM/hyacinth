
#include <hyacin.h>
#include <cuComplex.h>
#include <tuple>
#include <algorithm>

const double f32_polar_svd_cutoff_epi = 1.e-2;
const double f64_polar_svd_cutoff_epi = 1.e-12; // for epi < cutoff, use gesvd over gesvdp

inline std::tuple<cudaDataType_t, cudaDataType_t, double, size_t, size_t> convert(hyacinPrecision_t type) {
  switch(type) {
    case HYACIN_F64: return std::make_tuple(CUDA_R_64F, CUDA_R_64F, f64_polar_svd_cutoff_epi, sizeof(double), sizeof(double));
    case HYACIN_F32: return std::make_tuple(CUDA_R_32F, CUDA_R_32F, f32_polar_svd_cutoff_epi, sizeof(float), sizeof(float));
    case HYACIN_F64_COMPLEX: return std::make_tuple(CUDA_C_64F, CUDA_R_64F, f64_polar_svd_cutoff_epi, sizeof(cuDoubleComplex), sizeof(double));
    case HYACIN_F32_COMPLEX: return std::make_tuple(CUDA_C_32F, CUDA_R_32F, f32_polar_svd_cutoff_epi, sizeof(cuComplex), sizeof(float));
    default: return std::make_tuple(cudaDataType_t(-1), cudaDataType_t(-1), 0., size_t(0), size_t(0));
  }
}

extern "C" void hyacinXutvk_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t ldr, hyacinPrecision_t ComputeType, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  std::tuple<cudaDataType_t, cudaDataType_t, size_t, size_t, double> type = convert(ComputeType);
  cudaDataType_t type_c = std::get<0>(type), type_r = std::get<1>(type);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost; uint64_t basis_bytes;

  if (epi < std::get<2>(type)) {
    cusolverDnXgesvd_bufferSize(handle, params, 'S', 'N', N, K,
      type_c, nullptr, algnN, type_r, nullptr, type_c, nullptr, ldr, type_c, nullptr, algnK, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    basis_bytes = uint64_t(std::get<3>(type)) * uint64_t(algnN) * uint64_t(K);
  }
  else {
    cusolverDnXgesvdp_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      type_c, nullptr, algnN, type_r, nullptr, type_c, nullptr, ldr, type_c, nullptr, algnK, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    basis_bytes = uint64_t(std::get<3>(type)) * (uint64_t(algnN) + uint64_t(algnK)) * uint64_t(K);
  }

  uint64_t s_bytes = uint64_t(std::get<4>(type)) * uint64_t(algnK);
  *dev_work_bytes = uint64_t(workspaceInBytesOnDevice) + basis_bytes + s_bytes + uint64_t(sizeof(int32_t));
  *pinned_work_bytes = uint64_t(workspaceInBytesOnHost) + s_bytes;
}

inline int32_t hyacinDutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  const double* A, int32_t lda, double* RV, int32_t ldr, double* UT, int32_t ldu, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {
  
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(double) * int64_t(algnK);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];
  double one = 1., zero = 0.;

  int64_t r_bytes = sizeof(double) * int64_t(algnN) * int64_t(K);
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, RV, ldr, &zero, (double*)R, algnN, (double*)R, algnN);

  if (epi < f64_polar_svd_cutoff_epi) {
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* d_info = &bufferOnDevice[workspaceInBytesOnDevice];

    cusolverDnXgesvd(s_handle, params, 'S', 'N', N, K,
      CUDA_R_64F, R, algnN, CUDA_R_64F, S, CUDA_R_64F, RV, ldr, CUDA_R_64F, nullptr, algnK, CUDA_R_64F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info);
  }
  else {
    int64_t u_bytes = sizeof(double) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + u_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* U = &bufferOnDevice[workspaceInBytesOnDevice], *d_info = &U[u_bytes];

    double h_err_sigma = 0.;
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      CUDA_R_64F, R, algnN, CUDA_R_64F, S, CUDA_R_64F, RV, ldr, CUDA_R_64F, U, algnK, CUDA_R_64F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info, &h_err_sigma);
  }
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  double *Svec = (double*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](double s) { return s < s0; }))));
  if (0 < K)
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &one, A, lda, RV, ldr, &zero, UT, ldu);
  return K;
}

inline int32_t hyacinSutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  const float* A, int32_t lda, float* RV, int32_t ldr, float* UT, int32_t ldu, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {
  
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(float) * int64_t(algnK);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];
  float one = 1., zero = 0.;

  int64_t r_bytes = sizeof(float) * int64_t(algnN) * int64_t(K);
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, RV, ldr, &zero, (float*)R, algnN, (float*)R, algnN);

  if (epi < f64_polar_svd_cutoff_epi) {
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* d_info = &bufferOnDevice[workspaceInBytesOnDevice];

    cusolverDnXgesvd(s_handle, params, 'S', 'N', N, K,
      CUDA_R_32F, R, algnN, CUDA_R_32F, S, CUDA_R_32F, RV, ldr, CUDA_R_32F, nullptr, algnK, CUDA_R_32F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info);
  }
  else {
    int64_t u_bytes = sizeof(float) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + u_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* U = &bufferOnDevice[workspaceInBytesOnDevice], *d_info = &U[u_bytes];

    double h_err_sigma = 0.;
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      CUDA_R_32F, R, algnN, CUDA_R_32F, S, CUDA_R_32F, RV, ldr, CUDA_R_32F, U, algnK, CUDA_R_32F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info, &h_err_sigma);
  }
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  float *Svec = (float*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](float s) { return s < s0; }))));
  if (0 < K)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &one, A, lda, RV, ldr, &zero, UT, ldu);
  return K;
}

inline int32_t hyacinZutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* RV, int32_t ldr, cuDoubleComplex* UT, int32_t ldu, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {
  
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(double) * int64_t(algnK);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);

  int64_t r_bytes = sizeof(cuDoubleComplex) * int64_t(algnN) * int64_t(K);
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, RV, ldr, &zero, (cuDoubleComplex*)R, algnN, (cuDoubleComplex*)R, algnN);

  if (epi < f64_polar_svd_cutoff_epi) {
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* d_info = &bufferOnDevice[workspaceInBytesOnDevice];

    cusolverDnXgesvd(s_handle, params, 'S', 'N', N, K,
      CUDA_C_64F, R, algnN, CUDA_R_64F, S, CUDA_C_64F, RV, ldr, CUDA_C_64F, nullptr, algnK, CUDA_C_64F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info);
  }
  else {
    int64_t u_bytes = sizeof(cuDoubleComplex) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + u_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* U = &bufferOnDevice[workspaceInBytesOnDevice], *d_info = &U[u_bytes];

    double h_err_sigma = 0.;
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      CUDA_C_64F, R, algnN, CUDA_R_64F, S, CUDA_C_64F, RV, ldr, CUDA_C_64F, U, algnK, CUDA_C_64F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info, &h_err_sigma);
  }
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  double *Svec = (double*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](double s) { return s < s0; }))));
  if (0 < K)
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &one, A, lda, RV, ldr, &zero, UT, ldu);
  return K;
}

inline int32_t hyacinCutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  const cuComplex* A, int32_t lda, cuComplex* RV, int32_t ldr, cuComplex* UT, int32_t ldu, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {
  
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(double) * int64_t(algnK);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];
  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);

  int64_t r_bytes = sizeof(cuComplex) * int64_t(algnN) * int64_t(K);
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  cublasCgeam(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, K, &one, RV, ldr, &zero, (cuComplex*)R, algnN, (cuComplex*)R, algnN);

  if (epi < f64_polar_svd_cutoff_epi) {
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* d_info = &bufferOnDevice[workspaceInBytesOnDevice];

    cusolverDnXgesvd(s_handle, params, 'S', 'N', N, K,
      CUDA_C_32F, R, algnN, CUDA_R_32F, S, CUDA_C_32F, RV, ldr, CUDA_C_32F, nullptr, algnK, CUDA_C_32F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info);
  }
  else {
    int64_t u_bytes = sizeof(cuComplex) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + u_bytes + s_bytes + int64_t(sizeof(int32_t))));
    int8_t* U = &bufferOnDevice[workspaceInBytesOnDevice], *d_info = &U[u_bytes];

    double h_err_sigma = 0.;
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      CUDA_C_32F, R, algnN, CUDA_R_32F, S, CUDA_C_32F, RV, ldr, CUDA_C_32F, U, algnK, CUDA_C_32F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info, &h_err_sigma);
  }
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  float *Svec = (float*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](float s) { return s < s0; }))));
  if (0 < K)
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &one, A, lda, RV, ldr, &zero, UT, ldu);
  return K;
}

extern "C" int32_t hyacinXutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  const void* A, int32_t lda, void* RV, int32_t ldr, void* UT, int32_t ldu, hyacinPrecision_t ComputeType, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  switch(ComputeType) {
    case HYACIN_F64: return hyacinDutvk(handle, s_handle, params, epi, M, N, K, p, (const double*)A, lda, (double*)RV, ldr, (double*)UT, ldu, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32: return hyacinSutvk(handle, s_handle, params, epi, M, N, K, p, (const float*)A, lda, (float*)RV, ldr, (float*)UT, ldu, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F64_COMPLEX: return hyacinZutvk(handle, s_handle, params, epi, M, N, K, p, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)RV, ldr, (cuDoubleComplex*)UT, ldu, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32_COMPLEX: return hyacinCutvk(handle, s_handle, params, epi, M, N, K, p, (const cuComplex*)A, lda, (cuComplex*)RV, ldr, (cuComplex*)UT, ldu, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    default: return 0;
  }
}
