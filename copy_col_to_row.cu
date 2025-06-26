
#include <hyacinth.hpp>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <cuComplex.h>
#include <float4.hpp>

struct conj {
  __device__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ complex_float4 operator()(complex_float4 f) { return device::f4::conj(f); }
};

struct stride {
  int32_t ld;
  stride(int32_t ld) : ld(ld) {}
  __device__ int32_t operator()(int32_t i) { return i * ld; }
};

template<class T>
inline void copy_col_to_row_real(cudaStream_t stream, int32_t i, int32_t N, T* A, int32_t lda) {
  thrust::device_ptr<const T> Acol(&A[i * lda]);
  thrust::device_ptr<T> Arow(&A[i]);
  auto Arow_idx = thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride(lda));
  thrust::copy(thrust::cuda::par_nosync.on(stream), Acol, Acol + N, thrust::make_permutation_iterator(Arow, Arow_idx));
}

template<class T>
inline void copy_col_to_row_complex(cudaStream_t stream, int32_t i, int32_t N, T* A, int32_t lda) {
  thrust::device_ptr<const T> Acol(&A[i * lda]);
  thrust::device_ptr<T> Arow(&A[i]);
  auto Arow_idx = thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride(lda));
  thrust::transform(thrust::cuda::par_nosync.on(stream), Acol, Acol + N, thrust::make_permutation_iterator(Arow, Arow_idx), conj());
}

void copy_col_to_row_double(cudaStream_t stream, int32_t i, int32_t N, double* A, int32_t lda) {
  copy_col_to_row_real<double>(stream, i, N, A, lda);
}

void copy_col_to_row_double_complex(cudaStream_t stream, int32_t i, int32_t N, std::complex<double>* A, int32_t lda) {
  copy_col_to_row_complex<cuDoubleComplex>(stream, i, N, (cuDoubleComplex*)A, lda);
}
