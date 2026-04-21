
# How to compile
* Have a CUDA toolkit installation, CMake (Version >= 3.18)
* NCCL location set via NCCL_PATH, or Compiler paths CPATH (nccl.h) and LD_LIBRARY_PATH (libnccl) etc.

(1) Create build folder for Cmake `mkdir build && cd build`

(2) Modify `CMakeLists.txt`:

`set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120")`: List of SM architectures

`option(USE_NCCL "Enable NCCL backend" ON)`: ON/OFF if nccl is not available on system for single-GPU build

(2) Create cmake files `cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/hyacin-install`;

(3) Build library `make -j8`; This builds dynamic library

(4) Install files `cmake --install .`; Installs header, dynamic shared library, and Cmake configurations

(5) Link your own code from CMake:

add `set(Hyacin_DIR "/path/to/hyacin-install/lib/cmake")` and `find_package(Hyacin REQUIRED)` in your CMakeList.txt

Link your build target with `Hyacin::hyacin`; CMake resolves for CUDA-runtime, cuBLAS, cuSolverDn, NCCL dependencies.

# Authors
Qianxiang Ma Dr., Post-doc @ RIKEN R-CCS. ma@rio.scrc.iir.isct.ac.jp / qianxiang.ma@riken.jp
