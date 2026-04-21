
# What this library does

This library is a multi-GPU low-rank approximation driver; the user needs to assemble the proper routines to complete a low-rank approximation.

The current functionality provides Gram-EVD/SVD and Gram-ID options.

The distributed functionality has a 1-D row-cyclic Gram matrix formulation and a 1-D column-cyclic allGatherV.

For details, please refer to `hyacin.h`, the application interfaces.

# How to compile
* Have a CUDA toolkit installation, CMake (Version >= 3.18)
* NCCL location set via NCCL_PATH, or Compiler paths CPATH (nccl.h) and LD_LIBRARY_PATH (libnccl) etc.

(0) Before compiling, modify `CMakeLists.txt` if needed:

`set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120")`: List of SM architectures

`option(USE_NCCL "Enable NCCL backend" ON)`: ON/OFF if NCCL is not available on the system for a single-GPU build

(1) Create a build folder for CMake `mkdir build && cd build`

(2) Create CMake files `cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/hyacin-install`;

(3) Build library `make -j8`; This builds a dynamic library

(4) Install files `cmake --install .`; Installs header, dynamic shared library, and CMake configurations

(5) Link your own code from CMake:

Add `set(Hyacin_DIR "/path/to/hyacin-install/lib/cmake")` and `find_package(Hyacin REQUIRED)` in your CMakeList.txt

Link your build target with `Hyacin::hyacin`; CMake resolves for CUDA-runtime, cuBLAS, cuSolverDn, and NCCL dependencies.

# Authors
Qianxiang Ma Dr., Post-doc @ RIKEN R-CCS. ma@rio.scrc.iir.isct.ac.jp / qianxiang.ma@riken.jp
