cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(BUILD_EXAMPLES ON)

set(USE_MPI ON)
set(MPI_IS_CUDA_AWARE ON)

project ("hyacin" LANGUAGES CXX CUDA)
set(SM_TARGETS "80;86;89;90;100;120")

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES
  src_cpp/autotune.cpp
  src_cpp/gemv_cublas.cpp
  src_cpp/i63syrk.cpp
  src_cpp/imax_host.cpp
  src_cpp/potrfp_uniform.cpp
  src_cpp/svd_pp.cpp
)

set(CU_SOURCES
  src_cuda/acc_dequant_complex.cu
  src_cuda/acc_dequant_real.cu
  src_cuda/fp_exponent.cu
  src_cuda/fp_i8_quantize.cu
  src_cuda/fp_pervec_sum.cu
  src_cuda/gemv_pp_nopiv.cu
  src_cuda/gemv_pp.cu
  src_cuda/gemv_scal_f128.cu
  src_cuda/hyacin.cu
  src_cuda/i32_accum.cu
  src_cuda/i32_rem_accum.cu
  src_cuda/i63_i42_accum_conv.cu
  src_cuda/imax_diag.cu
)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-march=native;-Wall;-Wextra;-fexceptions")
set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fexceptions;--use_fast_math")

add_library(hyacin SHARED ${CPP_SOURCES} ${CU_SOURCES})
set_property(TARGET hyacin PROPERTY CUDA_ARCHITECTURES ${SM_TARGETS})

target_include_directories(hyacin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hyacin PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)

if (USE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(hyacin PRIVATE MPI::MPI_CXX)
  if (MPI_IS_CUDA_AWARE)
    target_compile_definitions(hyacin PRIVATE "MPI_IS_CUDA_AWARE")
    message(STATUS "MPI : CUDA aware")
  else()
    message(STATUS "MPI : No CUDA aware")
  endif()
else()
  message(STATUS "NO MPI")
  target_compile_definitions(hyacin PRIVATE "NO_MPI")
endif()

install(TARGETS hyacin LIBRARY DESTINATION .)
install(FILES hyacin.h DESTINATION .)
message(STATUS "Will install to: ${CMAKE_INSTALL_PREFIX}")

if (BUILD_EXAMPLES)
  message(STATUS "Will build: Example files")
  add_subdirectory(examples)
endif()

