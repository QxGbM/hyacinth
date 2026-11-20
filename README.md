# Hyacin

Hyacin computes low-rank approximations on one or more NVIDIA GPUs. It fits a
right basis for truncated singular value decomposition (SVD) or interpolative
decomposition, with a requested tolerance and a maximum rank.

The input can be a matrix, its transpose, or its conjugate transpose. Real and
complex FP16, FP32, and FP64 storage are supported. For distributed matrices,
Hyacin combines local Gram contributions through NCCL, using a one-dimensional
row distribution.

Hyacin accelerates Gram construction with integer Tensor Core arithmetic, using
integer-limb decomposition or the Chinese remainder theorem (CRT). Quantization
resolution is chosen from the requested tolerance, with a configurable accuracy
margin. The resulting Gram representation supplies the right basis; applications
can then form the compact left factor with a matrix multiplication.

## Build and install

You need a C++17 compiler, CMake 3.18 or newer, and a CUDA toolkit with cuBLAS and
cuSOLVER. NCCL is enabled by default. The CUDA compiler must support the GPU
architectures selected for the build.

```bash
git clone https://github.com/QxGbM/hyacinth.git
cd hyacinth

# If NCCL is outside the standard search paths:
export NCCL_PATH=/path/to/nccl

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/path/to/hyacin-install
cmake --build build -j8
cmake --install build
```

To build for a specific GPU architecture, pass `-DCMAKE_CUDA_ARCHITECTURES=86`,
for example. This also avoids compiling for architectures you do not use.
For a single-GPU build without NCCL, pass `-DUSE_NCCL=OFF`.

Installation provides the shared library, `hyacin.h`, and the CMake package.
Link your application through the exported target:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_app LANGUAGES CXX)

find_package(Hyacin REQUIRED PATHS "/path/to/hyacin-install/lib/cmake")

add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_17)
target_link_libraries(my_app PRIVATE Hyacin::hyacin)
```

`Hyacin::hyacin` supplies the include paths and links the CUDA runtime, cuBLAS,
cuSOLVER, and, when enabled, NCCL. A build without NCCL also exports the
`HYACIN_BUILD_NO_NCCL` definition to consumers of this target.

## Choose a fitting routine

| Routine | Result |
| --- | --- |
| `hyacinTsvdFit` | Truncated-SVD right basis and singular values. |
| `hyacinTsvdFitSegmentN` | The same outputs, with automatic column segmentation when the matrix dimensions and rank cap make it useful. |
| `hyacinInterpolateFit` | An interpolation basis and column pivots in LAPACK's one-based indexing. |

Each routine returns the retained rank `r`. The tolerance `eps` controls
truncation, and `K` caps the number of basis columns. A rank cap can stop the fit
before the requested tolerance is reached. Set `K = min(M, N)` when you want to
allow the full possible rank.

For SVD fitting, the output `V` contains the right singular vectors as columns.
For `B = op(A)`, the approximation is

```text
B ≈ (B V) Vᴴ,    V has shape N × r.
```

Hyacin does not explicitly form the left singular vectors. When you need a
compact left factor, compute `B V` directly with GEMM. This produces the factor
corresponding to `U S` without dividing by singular values.

## Matrix layout and dimensions

Matrices use column-major storage, and matrix arguments point to device memory.
`M` and `N` always describe **`op(A)`**, the matrix being approximated:

| `opA` | Operation | Stored shape of `A` | Minimum `lda` |
| --- | --- | --- | --- |
| `'N'` | `A` | `M × N` | `M` |
| `'T'` | `Aᵀ` | `N × M` | `N` |
| `'C'` | `Aᴴ` | `N × M` | `N` |

Allocate `V` for `N × K` elements, with `ldv >= N`, and the SVD singular-value
buffer for `K` real elements in the corresponding storage precision. Only the
first `r` columns of `V` contain the fitted basis. For a single GPU, pass `M` as
`globalM`.

Input `A` is preserved when `A` and `V` use separate allocations. `V` may also
alias `A` when the shared allocation and leading dimensions accommodate the
output; this overwrites the input with the basis.

## Quick start: complex FP64

This complete example fits the right basis of a complex matrix, checks the
relative projection error, and copies the retained basis to the host. The
separate input and output buffers preserve `A` for validation.

```cpp
#include <hyacin.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  constexpr int M = 512, N = 128, K = N;
  constexpr double eps = 1e-10;
  check_cuda(cudaSetDevice(0));

  // This example matrix has a low-dimensional right subspace.
  std::vector<cuDoubleComplex> A(size_t(M) * N);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i)
      A[i + size_t(j) * M] = make_cuDoubleComplex(
        (1. + .01 * i) * (1. + .02 * j),
        std::sin(.1 * i) * std::cos(.2 * j));

  cuDoubleComplex *d_A = nullptr, *d_V = nullptr;
  double* d_S = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_A), A.size() * sizeof(*d_A)));
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_V), size_t(N) * K * sizeof(*d_V)));
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_S), K * sizeof(*d_S)));
  check_cuda(cudaMemcpy(d_A, A.data(), A.size() * sizeof(*d_A), cudaMemcpyHostToDevice));

  hyacinHandle_t handle{};
  hyacinCreate(&handle);

  const int rank = hyacinTsvdFitSegmentN(
    &handle, eps, 'N', M, N, K,
    d_A, M, d_S, d_V, N, M, HYACIN_F64_COMPLEX);

  const double norm2 = hyacinCheckNorm(
    &handle, 'N', M, N, d_A, M, 0, nullptr, 0, HYACIN_F64_COMPLEX);
  const double residual2 = hyacinCheckNorm(
    &handle, 'N', M, N, d_A, M, rank, d_V, N, HYACIN_F64_COMPLEX);
  const double relative_error = norm2 > 0. ? std::sqrt(residual2 / norm2) : 0.;

  hyacinSync_TimerSegments(&handle, nullptr, 0);
  std::vector<cuDoubleComplex> V(size_t(N) * rank);
  if (rank > 0)
    check_cuda(cudaMemcpy(V.data(), d_V, V.size() * sizeof(*d_V), cudaMemcpyDeviceToHost));
  std::printf("rank = %d, relative projection error = %.3e\n", rank, relative_error);

  hyacinDestroy(&handle);
  check_cuda(cudaFree(d_S));
  check_cuda(cudaFree(d_V));
  check_cuda(cudaFree(d_A));
}
```

To fit the conjugate transpose of a stored `rows × cols` matrix, pass
`opA = 'C'`, `M = cols`, `N = rows`, and `lda = rows`. The returned basis then has
`rows` rows. This is also a way to fit the left subspace of the original matrix.

## Reuse your application's CUDA resources

Start with a zero-initialized handle. Select the GPU before calling
`hyacinCreate`, and reuse the handle for successive fits on that GPU.

With `hyacinHandle_t handle{}`, Hyacin uses the default CUDA stream. It creates
the missing cuBLAS and cuSOLVER handles, memory pool, and pinned workspace.

To integrate with an existing application, supply its resources before creation:

```cpp
// stream, blas, and solver already belong to the application.
// Bind both library handles to stream before passing them to Hyacin.
hyacinHandle_t handle{};
handle.cudaStream = stream;
handle.cublasHandle = blas;
handle.cusolverHandle = solver;
hyacinCreate(&handle);

// Perform one or more fits using handle.

hyacinSync_TimerSegments(&handle, nullptr, 0);
hyacinDestroy(&handle);
// stream, blas, and solver remain valid and owned by the application.
```

`hyacinCreate` preserves supplied resource handles and creates missing resources
as needed. A stream supplied by the application remains application-owned;
Hyacin does not create or destroy CUDA streams. A supplied memory pool is also
borrowed. `hyacinDestroy` releases resources created by Hyacin and clears the
handle; it leaves borrowed resources intact.

Keep borrowed resources alive until all Hyacin work has completed and the handle
has been destroyed. When supplying cuBLAS or cuSOLVER handles, configure them for
the same GPU and stream as Hyacin. Ownership bookkeeping is internal: leave
`HandleOwnedResource` at its zero-initialized value before `hyacinCreate`.

## Distributed fitting with NCCL

With an NCCL-enabled build, assign an existing communicator to
`handle.ncclComm`. A null communicator selects local, single-GPU computation.
Hyacin borrows the communicator; the application creates and destroys it and
chooses the GPU used by each rank.

Distribute the rows of `B = op(A)` among ranks. Each rank supplies its local rows
and participates in Gram construction and reduction. The right basis is
replicated across the communicator. The row distribution may be contiguous or
cyclic; the local matrices must share the same column ordering.

```cpp
// d_localA stores localM rows of B in column-major order.
// All ranks use the same N, K, eps, precision, and globalM.
hyacinHandle_t handle{};
handle.cudaStream = stream;
handle.ncclComm = comm;
hyacinCreate(&handle);

const int rank = hyacinTsvdFitSegmentN(
  &handle, eps, 'N', localM, N, K,
  d_localA, localM, d_S, d_V, N, globalM, HYACIN_F64_COMPLEX);

hyacinSync_TimerSegments(&handle, nullptr, 0);
hyacinDestroy(&handle);
```

Here `globalM` is the sum of the local row counts. To allow the full possible
rank, use `K = min(globalM, N)` rather than limiting it to a local row count.
All ranks must call the fit in the same collective order. Hyacin handles the numerical communication through
NCCL; process launch and communicator bootstrap belong to the application. MPI
can be used for that bootstrap but is not a Hyacin dependency.

## Check the fitted subspace

`hyacinCheckNorm` computes the squared Frobenius projection residual:

```text
Q = orth(V)
residual² = ‖op(A) − op(A) Q Qᴴ‖²_F
```

With `rank = 0`, it returns `‖op(A)‖²_F` and does not access `V`. Divide the two
results and take the square root to obtain relative projection error, as in the
quick-start example. The check promotes stored values to real or complex FP64
for orthogonalization, products, and norms. It also works with the
nonorthogonal basis returned by interpolative fitting.

The check preserves `A` and `V`. For distributed inputs, it returns a **local**
squared norm or residual; sum those values across ranks before computing the
global relative error.

## Accuracy and performance settings

Start with the defaults. `hyacinCreate` reads the following environment
variables; set them before creating the handle. The corresponding fields in
`hyacinHandle_t` can also be adjusted after creation, between fits.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `HYACIN_GRAM_ALGORITHM` | `A` | Automatic selection; `L` selects integer limbs and `C` selects CRT. |
| `HYACIN_QUANTIZE_BITS_CORR` | `7` | Extra quantization bits beyond the estimated requirement. |
| `HYACIN_QRCP_TRUNCATE_RANK` | `1` | Enables early rank truncation in the Gram factorization. Set to `0` to retain the full factor for SVD. |
| `HYACIN_JACOBI_SVD_SWEEPS` | `3` | Jacobi sweep budget where that backend is selected. Set to `0` for bidiagonalization-based SVD; a negative value forces Jacobi with that absolute sweep count. |
| `HYACIN_GRAM_SEGMENT_K` | `8192` | Reduction segment length used for local exponent scaling. |
| `HYACIN_TCGEMM_TARGET_K` | `8192` | Target reduction segment length for CRT Tensor Core GEMM. |
| `HYACIN_STRIDE_BATCH_PERF_PENALTY` | `1.3` | Cost multiplier used by automatic Gram algorithm selection. |

Hyacin favors bidiagonalization for FP64 computations on GPUs with limited FP64
throughput. The backend also depends on early-truncation settings; see the
[tuning comments in the public header](include/hyacin.h) for details.

Quantization resolution matters most near the small end of the spectrum.
Reducing its margin too aggressively can retain unnecessary rank or impair the
factorization. The defaults use a positive margin and automatic Gram selection.
For applications that depend on small singular values, validate the returned
subspace and spectrum at the precision and tolerance you intend to use.

Set `HYACIN_VERBOSE=1` to enable diagnostic output. To collect internal GPU
timings, set `HYACIN_CREATE_TIMER=1` before creating the handle, then request
segments in milliseconds:

```cpp
double milliseconds[] = {'D', 'R', 'C'};
hyacinSync_TimerSegments(&handle, milliseconds, 3);
// Distributed kernels, replicated kernels, and communication, respectively.
```

`hyacinSync_TimerSegments` waits for the stream and clears the recorded timer
events. Passing `nullptr, 0` synchronizes and clears events without requesting
timings.

## Examples and API reference

The [numeric-bench repository](https://github.com/QxGbM/numeric-bench) includes
single-GPU and distributed fitting examples, vendor comparisons, and HSS
construction using Hyacin compression.

See [include/hyacin.h](include/hyacin.h) for the complete public API, precision
identifiers, resource fields, and lower-level Gram construction routines.

## Author

Qianxiang Ma, Ph.D., Postdoctoral Researcher, RIKEN R-CCS.

Contact: ma@rio.scrc.iir.isct.ac.jp / qianxiang.ma@riken.jp
