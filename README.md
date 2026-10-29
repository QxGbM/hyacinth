
# What this library does

This library is a single- or multi-GPU low-rank approximation driver.

The current functionality provides Truncated-SVD and Interpolative-decomposition right-side basis fitting, with A, A^T, or A^H as input.

The distributed functionality has a 1-D row-cyclic Gram matrix formulation.

# How to compile
* Have a CUDA toolkit installation and CMake (Version >= 3.18)
* NCCL location set via NCCL_PATH, or Compiler paths CPATH (nccl.h) and LD_LIBRARY_PATH (libnccl) etc.

(0) Before compiling, modify `CMakeLists.txt` if needed:

`set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90a;100a;103;120;121")`: List of SM architectures

`option(USE_NCCL "Enable NCCL backend" ON)`: ON/OFF if NCCL is not available on the system for a single-GPU build

(1) Create a build folder for CMake: `mkdir build && cd build`

(2) Create CMake files: `cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/hyacin-install`;

(3) Build the library: `make -j8`; This builds a dynamic library

(4) Install files: `cmake --install .`; installs headers, the dynamic shared library, and CMake configurations

(5) Link your own code from CMake:

Add `find_package(Hyacin REQUIRED PATHS "/path/to/hyacin-install/lib/cmake")` in your CMakeList.txt

Link your build target with `Hyacin::hyacin`, with `target_link_libraries(your_app PRIVATE Hyacin::hyacin)` for example

CMake resolves the header include path, CUDA-runtime, cuBLAS, cuSolverDn, and NCCL dependencies.

# A Quick startup guide

Simple GPU offload example for op(A) = A^H in complex double precision

```
#include <hyacin.h>
hyacinHandle_t handle = {};
// Setup and teardown functions at the start and end of the program
void libs_init() { cudaSetDevice(0); hyacinCreate(&handle); }
void libs_finalize() { hyacinDestroy(&handle); }

int tsvd_fit_gpu(double epi, int M, int N, std::complex<double>* A) {
  // GPU device memory allocations
  std::complex<double>* devA = nullptr; double* devS = nullptr; 
  cudaMalloc((void**)&devA, (long long)M * (long long)N * sizeof(std::complex<double>));
  cudaMalloc((void**)&devS, (long long)M * sizeof(double));

  // Copy the input matrix A to device
  cudaMemcpy(devA, A, (long long)M * (long long)N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  // The V-basis can safely alias with input matrix A
  int rank = hyacinTsvdFitSegmentN(&handle, epi, 'C', N, M, M, devA, M, devS, devA, M, N, HYACIN_F64_COMPLEX);

  // Copy back the V-basis from GPU device memory
  cudaMemcpy(A, devA, (long long)M * (long long)rank * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  // Clean up the device resources
  cudaFree(devA); cudaFree(devS);
  return rank;
}
```

# More advanced uses

For details, see `include/hyacin.h` for the application interfaces, performance/accuracy parameters, and tuning guide.

# Authors
Qianxiang Ma, Dr., Postdoc @ RIKEN R-CCS. ma@rio.scrc.iir.isct.ac.jp / qianxiang.ma@riken.jp
