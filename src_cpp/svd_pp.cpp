
#include <hyacin.h>
#include <cuComplex.h>
#include <tuple>
#include <algorithm>

const double f32_polar_svd_cutoff_epi = 1.e-5;
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
      type_c, nullptr, algnN, type_r, nullptr, type_c, nullptr, ldr, type_c, nullptr, K, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    basis_bytes = uint64_t(std::get<3>(type)) * uint64_t(algnN) * uint64_t(K);
  }
  else {
    cusolverDnXgesvdp_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, K, N,
      type_c, nullptr, algnK, type_r, nullptr, type_c, nullptr, algnK, type_c, nullptr, ldr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    basis_bytes = uint64_t(std::get<3>(type)) * uint64_t(algnK) * (uint64_t(K) + uint64_t(N));
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
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];
  double one = 1., zero = 0.;

  if (epi < f64_polar_svd_cutoff_epi) {
    int64_t r_bytes = sizeof(double) * int64_t(algnN) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes + int64_t(sizeof(int32_t))));
    size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));

    int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes], *d_info = &bufferOnDevice[workspaceInBytesOnDevice];
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &one, RV, ldr, &zero, (double*)R, algnN, (double*)R, algnN);
    cusolverDnXgesvd(s_handle, params, 'S', 'N', N, K,
      CUDA_R_64F, R, algnN, CUDA_R_64F, S, CUDA_R_64F, RV, ldr, CUDA_R_64F, nullptr, K, CUDA_R_64F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info);
    cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  }
  else {
    int64_t r_bytes = sizeof(double) * int64_t(algnK) * int64_t(N);
    int64_t u_bytes = sizeof(double) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + u_bytes + s_bytes + int64_t(sizeof(int32_t))));
    size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
    
    int8_t* R = (int8_t*)dev_work, *U = &R[r_bytes], *S = &U[u_bytes], *bufferOnDevice = &S[s_bytes], *d_info = &bufferOnDevice[workspaceInBytesOnDevice];
    double h_err_sigma = 0.;
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, &one, RV, ldr, &zero, (double*)R, algnK, (double*)R, algnK);
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, K, N,
      CUDA_R_64F, R, algnK, CUDA_R_64F, S, CUDA_R_64F, U, algnK, CUDA_R_64F, RV, ldr, CUDA_R_64F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, (int32_t*)d_info, &h_err_sigma);
    cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  }
  
  cudaStreamSynchronize(stream);
  double *Svec = (double*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](double s) { return s < s0; }))));
  if (0 < K)
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &one, A, lda, RV, ldr, &zero, UT, ldu);
  return K;
}

extern "C" int32_t hyacinXutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  const void* A, int32_t lda, void* RV, int32_t ldr, void* UT, int32_t ldu, hyacinPrecision_t ComputeType, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  switch(ComputeType) {
    case HYACIN_F64: return hyacinDutvk(handle, s_handle, params, epi, M, N, K, p, (const double*)A, lda, (double*)RV, ldr, (double*)UT, ldu, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    default: return 0;
  }
}
