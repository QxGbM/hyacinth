
# How to compile
* Have a cuda toolkit installation
* Have a BLAS/LAPACK installation (MKL is okay)

(1) Create build folder for Cmake `mkdir build && cd build`

(2) Create cmake files `cmake ..`; optionally add extra configurations to find packages

(3) Build library `make`; This builds dynamic library and several examples

(4) Run examples `./dlra_example `; Or link your own code

# Authors
Qianxiang Ma, RIKEN R-CCS. ma@rio.scrc.iir.isct.ac.jp
