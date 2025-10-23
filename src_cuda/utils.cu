
#include <hyacin.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <class elem_t, class sign_t, sign_t one> struct vector_signs {
  const elem_t* __restrict__ X;
  sign_t* __restrict__ signs;
  int32_t incx;
  vector_signs(const elem_t* X, int32_t incx, sign_t* signs) : X(X), signs(signs), incx(incx) {}

  __device__ __forceinline__ void operator()(int32_t i) {
    signs[i] = signbit(X[i * incx]) ? one : sign_t();
  }
};

void device::copy_signs(cudaStream_t stream, int32_t N, const void* X, int32_t incx, Precision precX, void* signs, int32_t sign_bytes) {
  thrust::counting_iterator<int32_t> iter(0);
  if (precX == Precision::FP64 && sign_bytes == 1)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, N, vector_signs<double, uint8_t, uint8_t(0xFF)>((const double*)X, incx, (uint8_t*)signs));
  else if (precX == Precision::FP32 && sign_bytes == 1)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, N, vector_signs<float, uint8_t, uint8_t(0xFF)>((const float*)X, incx, (uint8_t*)signs));
  else if (precX == Precision::FP64 && sign_bytes == 2)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, N, vector_signs<double, uint16_t, uint16_t(0xFFFF)>((const double*)X, incx, (uint16_t*)signs));
  else if (precX == Precision::FP32 && sign_bytes == 2)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, N, vector_signs<float, uint16_t, uint16_t(0xFFFF)>((const float*)X, incx, (uint16_t*)signs));
}

template <class typeA, class typeB> struct convert_copy {
  const typeA* __restrict__ A;
  const int8_t* __restrict__ signs;
  typeB* __restrict__ B;
  int64_t M, lda, ldb;
  char mode;
  convert_copy(char mode, int64_t M, const typeA* A, int64_t lda, typeB* B, int64_t ldb, const int8_t* signs) :
    A(A), signs(signs), B(B), M(M), lda(lda), ldb(ldb), mode(mode) {}

  __device__ __forceinline__ void conv(double a, double& b, int8_t sign) { b = sign ? -a : a; }
  __device__ __forceinline__ void conv(float a, double& b, int8_t sign) { b = double(sign ? -a : a); }
  __device__ __forceinline__ void conv(double2 a, double& b, int8_t sign) { double c = device::dd::dd2double(a); b = sign ? -c : c; }
  __device__ __forceinline__ void conv(float4 a, double& b, int8_t sign) { double c = device::qf::qf2double(a); b = sign ? -c : c; }

  __device__ __forceinline__ void conv(double a, float& b, int8_t sign) { float c = float(a); b = sign ? -c : c; }
  __device__ __forceinline__ void conv(float a, float& b, int8_t sign) { b = sign ? -a : a; }
  __device__ __forceinline__ void conv(double2 a, float& b, int8_t sign) { float c = float(a.x) + float(a.y); b = sign ? -c : c; }
  __device__ __forceinline__ void conv(float4 a, float& b, int8_t sign) { float c = a.x + a.y + a.z + a.w; b = sign ? -c : c; }

  __device__ __forceinline__ void conv(double a, double2& b, int8_t sign) { b = device::dd::double2dd(sign ? -a : a); }
  __device__ __forceinline__ void conv(float a, double2& b, int8_t sign) { b = make_double2(double(sign ? -a : a), 0.); }
  __device__ __forceinline__ void conv(double2 a, double2& b, int8_t sign) { b = sign ? device::dd::negate(a) : a; }
  __device__ __forceinline__ void conv(float4 a, double2& b, int8_t sign) { b = device::dd::qf2dd(sign ? device::qf::negate(a) : a); }

  __device__ __forceinline__ void conv(double a, float4& b, int8_t sign) { b = device::qf::double2qf(sign ? -a : a); }
  __device__ __forceinline__ void conv(float a, float4& b, int8_t sign) { b = make_float4(sign ? -a : a, 0.f, 0.f, 0.f); }
  __device__ __forceinline__ void conv(double2 a, float4& b, int8_t sign) { b = device::dd::dd2qf(sign ? device::dd::negate(a) : a); }
  __device__ __forceinline__ void conv(float4 a, float4& b, int8_t sign) { b = sign ? device::qf::negate(a) : a; }

  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    if ((mode != 'u' || y <= x) && (mode != 'U' || (y >> 1) <= x))
      conv(A[y + x * lda], B[y + x * ldb], signs[y]);
  }
};

template <class typeA>
inline void conv_copy_dispatcher(cudaStream_t stream, char mode, int32_t M, int32_t N, const typeA* A, int32_t lda, void* B, int32_t ldb, device::Precision precB, const int8_t* signs) {
  thrust::counting_iterator<int64_t> iter(0);
  int64_t len = int64_t(M) * int64_t(N);
  switch (precB) {
    case device::Precision::FP64:
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, len, convert_copy<typeA, double>(mode, M, A, lda, (double*)B, ldb, signs)); break;
    case device::Precision::FP32:
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, len, convert_copy<typeA, float>(mode, M, A, lda, (float*)B, ldb, signs)); break;
    case device::Precision::FP128_DD:
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, len, convert_copy<typeA, double2>(mode, M, A, lda, (double2*)B, ldb, signs)); break;
    case device::Precision::FP128_QF:
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, len, convert_copy<typeA, float4>(mode, M, A, lda, (float4*)B, ldb, signs)); break;
    default: break;
  }
}

void device::convert_and_copy(cudaStream_t stream, char mode, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB, const int8_t* signs) {
  switch (precA) {
    case Precision::FP64: conv_copy_dispatcher<double>(stream, mode, M, N, (const double*)A, lda, B, ldb, precB, signs); break;
    case Precision::FP32: conv_copy_dispatcher<float>(stream, mode, M, N, (const float*)A, lda, B, ldb, precB, signs); break;
    case Precision::FP128_DD: conv_copy_dispatcher<double2>(stream, mode, M, N, (const double2*)A, lda, B, ldb, precB, signs); break;
    case Precision::FP128_QF: conv_copy_dispatcher<float4>(stream, mode, M, N, (const float4*)A, lda, B, ldb, precB, signs); break;
    default: break;
  }
}

const int32_t clen = 16384;
__constant__ int32_t cpiv[clen];

template <int32_t sc0ga1, class elem_t> struct permute_copy {
  const elem_t* __restrict__ A;
  elem_t* __restrict__ B;
  int64_t M, lda, ldb;
  permute_copy(int64_t M, const elem_t* A, int64_t lda, elem_t* B, int64_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int64_t px = int64_t(cpiv[x] - 1);
    if constexpr(sc0ga1) B[y + x * ldb] = A[y + px * lda];
      else B[y + px * ldb] = A[y + x * lda];
  }
};

void device::copy_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec) {
  thrust::counting_iterator<int64_t> iter(0);
  for (int32_t i = 0; i < N; i += clen) {
    int32_t cols = std::min(N - i, clen);
    int64_t elements = int64_t(cols) * int64_t(M);
    cudaMemcpyToSymbolAsync(cpiv, &jpiv[i], int64_t(cols) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);

    if (prec == Precision::FP64) {
      permute_copy<1, int64_t> perm(M, (const int64_t*)A, lda, &((int64_t*)B)[int64_t(i) * int64_t(ldb)], ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
    else if (prec == Precision::FP32) {
      permute_copy<1, int32_t> perm(M, (const int32_t*)A, lda, &((int32_t*)B)[int64_t(i) * int64_t(ldb)], ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
  }
}

void device::inplace_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, void* A, int32_t lda, void* workspace, Precision prec) {
  const int32_t ws_rows = 8192;
  int64_t elem_bytes = prec == Precision::FP32 ? sizeof(float) : sizeof(double);
  thrust::counting_iterator<int64_t> iter(0);
  if (N <= clen)
    cudaMemcpyToSymbolAsync(cpiv, jpiv, int64_t(N) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);

  for (int32_t i = 0; i < M; i += ws_rows) {
    int32_t rows = std::min(M - i, ws_rows);
    uint8_t* A_i = &((uint8_t*)A)[elem_bytes * int64_t(i)];

    if (N <= clen && prec == Precision::FP64) {
      permute_copy<1, int64_t> perm(rows, (const int64_t*)A_i, lda, (int64_t*)workspace, rows);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(rows), perm);
    }
    else if (N <= clen && prec == Precision::FP32) {
      permute_copy<1, int32_t> perm(rows, (const int32_t*)A_i, lda, (int32_t*)workspace, rows);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, int64_t(N) * int64_t(rows), perm);
    }
    else
      copy_gather(stream, rows, N, jpiv, A_i, lda, workspace, rows, prec);
    cudaMemcpy2DAsync(A_i, elem_bytes * int64_t(lda), workspace, elem_bytes * int64_t(rows), elem_bytes * int64_t(rows), N, cudaMemcpyDeviceToDevice, stream);
  }
}

void device::copy_scatter(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec) {
  thrust::counting_iterator<int64_t> iter(0);
  for (int32_t i = 0; i < N; i += clen) {
    int32_t cols = std::min(N - i, clen);
    int64_t elements = int64_t(cols) * int64_t(M);
    cudaMemcpyToSymbolAsync(cpiv, &jpiv[i], int64_t(cols) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);

    if (prec == Precision::FP64) {
      permute_copy<0, int64_t> perm(M, &((const int64_t*)A)[int64_t(i) * int64_t(lda)], lda, (int64_t*)B, ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
    else if (prec == Precision::FP32) {
      permute_copy<0, int32_t> perm(M, &((const int32_t*)A)[int64_t(i) * int64_t(lda)], lda, (int32_t*)B, ldb);
      thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
    }
  }
}

template <class elem_t, elem_t one> struct identity {
  elem_t* __restrict__ A;
  int64_t M, lda, strideD;
  identity(int64_t M, elem_t* A, int64_t lda, int64_t strideD) :
    A(A), M(M), lda(lda), strideD(strideD) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int32_t diag = int32_t(i % strideD == int64_t(0));
    A[y + x * lda] = diag ? one : elem_t();
  }
};

void device::strided_identity(cudaStream_t stream, int32_t M, int32_t N, int32_t strideD, void* A, int32_t lda, Precision prec) {
  int64_t iter_items = int64_t(N) * int64_t(M);
  thrust::counting_iterator<int64_t> iter(0);

  if (prec == Precision::FP64) {
    const int64_t f64_one = 0x3FF0000000000000LL;
    identity<int64_t, f64_one> id(M, (int64_t*)A, lda, strideD);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
  else if (prec == Precision::FP32) {
    const int32_t f32_one = 0x3F800000;
    identity<int32_t, f32_one> id(M, (int32_t*)A, lda, strideD);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
}
