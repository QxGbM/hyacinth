cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

option(BUILD_EXAMPLES "Example files" ON)
option(USE_NCCL "Enable NCCL backend" OFF)

project("hyacin" LANGUAGES CXX CUDA)
set(SM_TARGETS "80;86;89;90;100;120")

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES
  src_cpp/allgatherv_1dcol.cpp
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
  src_cuda/i63_u47_accum_conv.cu
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

if(USE_NCCL)
  find_package(NCCL QUIET)
  if(NCCL_FOUND)
    message(STATUS "USE NCCL: CMake Package")
  else()
    find_path(NCCL_INCLUDE_DIRS NAMES nccl.h HINTS ENV NCCL_PATH ENV CPATH ENV C_INCLUDE_PATH ENV CPLUS_INCLUDE_PATH PATH_SUFFIXES include)
    find_library(NCCL_LIBRARIES NAMES nccl HINTS ENV NCCL_PATH ENV LD_LIBRARY_PATH ENV LIBRARY_PATH PATH_SUFFIXES lib lib64)

    if (NCCL_INCLUDE_DIRS AND NCCL_LIBRARIES)
      message(STATUS "USE NCCL: ENV")
    else()
      find_package(PkgConfig QUIET)
      if(PkgConfig_FOUND)
        message(STATUS "USE NCCL: PkgConfig")
        pkg_check_modules(NCCL QUIET nccl)
      endif()
    endif()
  endif()

  if (NCCL_INCLUDE_DIRS AND NCCL_LIBRARIES)
    message(STATUS "Found NCCL include: ${NCCL_INCLUDE_DIRS}")
    message(STATUS "Found NCCL library: ${NCCL_LIBRARIES}")
    target_include_directories(hyacin PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(hyacin PRIVATE ${NCCL_LIBRARIES})
  else()
    message(FATAL_ERROR "NCCL requested but not found.")
  endif()
else()
  message(STATUS "NO NCCL")
  set_source_files_properties(src_cuda/hyacin.cu src_cpp/allgatherv_1dcol.cpp PROPERTIES COMPILE_DEFINITIONS "NO_NCCL")
endif()

install(TARGETS hyacin LIBRARY DESTINATION .)
install(FILES hyacin.h DESTINATION .)
message(STATUS "Will install to: ${CMAKE_INSTALL_PREFIX}")

if (BUILD_EXAMPLES)
  message(STATUS "Will build: Example files")
  add_subdirectory(examples)
endif()

