
#include <hyacin.h>
#include <cuComplex.h>
#include <algorithm>
#include <stdexcept>

template <class T> __device__ __forceinline__ T float_relu_sqrt(T a);
template <> __device__ __forceinline__ double float_relu_sqrt<double>(double a) { return sqrt(fmax(a, 0.)); };
template <> __device__ __forceinline__ float float_relu_sqrt<float>(float a) { return sqrtf(fmaxf(a, 0.f)); };

template <class real_t, class complex_t, class constCptr, class Cptr, class constRptr, class Rptr>
__global__ void evd_reorder_kernel(int64_t M, int64_t N_minus_one, constCptr A, int64_t lda, Cptr B, int64_t ldb, constRptr s_in, Rptr s_out) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y < M) {
    if (x <= N_minus_one) B[y + int64_t(N_minus_one - x) * ldb] = A[y + x * lda];
      else if (y <= N_minus_one) s_out[N_minus_one - y] = float_relu_sqrt(s_in[y]);
  }
};

template <class T> inline cudaDataType_t cuda_type();
template <> inline cudaDataType_t cuda_type<double>() { return CUDA_R_64F; }
template <> inline cudaDataType_t cuda_type<float>() { return CUDA_R_32F; }
template <> inline cudaDataType_t cuda_type<cuDoubleComplex>() { return CUDA_C_64F; }
template <> inline cudaDataType_t cuda_type<cuComplex>() { return CUDA_C_32F; }

template <class real_t, class complex_t, class constCptr, class Cptr, class constRptr, class Rptr>
inline int32_t tevd(cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p,
  void* X, int32_t ldx, void* G, int32_t ldg, void* S, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {
  cudaStream_t stream; cusolverDnGetStream(s_handle, &stream);
  cudaDataType_t type_c = cuda_type<complex_t>(), type_r = cuda_type<real_t>();
  K = std::min(K, N); uint64_t s_bytes = uint64_t(sizeof(real_t)) * K;
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXsyevd_bufferSize(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, type_c, nullptr, ldx, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  
  if ((uint64_t(workspaceInBytesOnDevice) + s_bytes) <= dev_work_bytes && uint64_t(workspaceInBytesOnHost) <= pinned_work_bytes) {
    real_t* dev_S = (real_t*)(&((int8_t*)dev_work)[workspaceInBytesOnDevice]);
    int64_t x_start = int64_t(N - K);
    cusolverDnXsyevd(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, type_c, G, ldg, type_r, dev_S, type_c, dev_work, workspaceInBytesOnDevice, pinned_work, workspaceInBytesOnHost, nullptr);
    evd_reorder_kernel <real_t, complex_t, constCptr, Cptr, constRptr, Rptr> <<< dim3(uint32_t((N + 511) >> 9), uint32_t(K + 1)), 512, 0, stream >>>
      (int64_t(N), int64_t(K - 1), &((const complex_t*)G)[x_start * int64_t(ldg)], int64_t(ldg), (complex_t*)X, int64_t(ldx), &((const real_t*)dev_S)[x_start], (real_t*)S); 
    cudaMemcpyAsync(pinned_work, S, int64_t(K) * sizeof(real_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    real_t *Svec = (real_t*)pinned_work;
    double s0 = epi * double(Svec[0]);
    return std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return double(s) < s0; }))));
  }
  else throw std::runtime_error("Insufficient workspace provided for SYEVD.");
}

extern "C" void hyacinXGevd_bufferSize(cusolverDnHandle_t s_handle, cusolverDnParams_t params, int32_t N, int32_t K, hyacinPrecision_t AXGtype, int32_t ldg, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  int64_t x_real_bytes = int64_t(0);
  cudaDataType_t type_c = cudaDataType_t(), type_r = cudaDataType_t();
  switch (AXGtype) {
    case HYACIN_F64: { type_c = type_r = cuda_type<double>(); x_real_bytes = sizeof(double); break; }
    case HYACIN_F32: { type_c = type_r = cuda_type<float>(); x_real_bytes = sizeof(float); break; } 
    case HYACIN_F64_COMPLEX: { type_c = cuda_type<cuDoubleComplex>(); type_r = cuda_type<double>(); x_real_bytes = sizeof(double); break; }
    case HYACIN_F32_COMPLEX: { type_c = cuda_type<cuComplex>(); type_r = cuda_type<float>(); x_real_bytes = sizeof(float); break; }
    default: break;
  }

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXsyevd_bufferSize(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, type_c, nullptr, ldg, type_r, nullptr, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  *dev_work_bytes = uint64_t(workspaceInBytesOnDevice) + uint64_t(x_real_bytes * int64_t(K));
  *pinned_work_bytes = std::max(uint64_t(workspaceInBytesOnHost), uint64_t(x_real_bytes * int64_t(K)));
}

extern "C" int32_t hyacinXGevd(cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, int32_t p,
  hyacinPrecision_t AXGtype, void* S, void* X, int32_t ldx, void* G, int32_t ldg, void* dev_work, uint64_t dev_work_bytes, void* pinned_work, uint64_t pinned_work_bytes) {

  switch (AXGtype) {
    case HYACIN_F64: return tevd<double, double, const double* __restrict__, double* __restrict__, const double* __restrict__, double* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32: return tevd<float, float, const float* __restrict__, float* __restrict__, const float* __restrict__, float* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F64_COMPLEX: return tevd<double, cuDoubleComplex, const cuDoubleComplex* __restrict__, cuDoubleComplex* __restrict__, const double* __restrict__, double* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    case HYACIN_F32_COMPLEX: return tevd<float, cuComplex, const cuComplex* __restrict__, cuComplex* __restrict__, const float* __restrict__, float* __restrict__>
      (s_handle, params, epi, N, K, p, X, ldx, G, ldg, S, dev_work, dev_work_bytes, pinned_work, pinned_work_bytes);
    default: return 0;
  }
}
