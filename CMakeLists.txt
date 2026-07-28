cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90a;100a;120")

option(USE_NCCL "Enable NCCL backend" ON)
project("hyacin" LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES
  src_cpp/autotune.cpp
  src_cpp/gemv_cublas.cpp
  src_cpp/gram_id.cpp
  src_cpp/handle_timer.cpp
  src_cpp/i63syrk.cpp
  src_cpp/syherk.cpp
  src_cpp/transform.cpp
)

set(CU_SOURCES
  src_cuda/acc_dequant_complex.cu
  src_cuda/acc_dequant_real.cu
  src_cuda/allgatherv_1dcol.cu
  src_cuda/fp_exponent.cu
  src_cuda/fp_i8_quantize.cu
  src_cuda/fp_pervec_sum.cu
  src_cuda/gemv_pp.cu
  src_cuda/gemv_scal_f128.cu
  src_cuda/gram_svd.cu
  src_cuda/i32_accum.cu
  src_cuda/i32_rem_accum.cu
  src_cuda/i63_accum_conv.cu
  src_cuda/matcopy.cu
  src_cuda/potrfp_uniform.cu
)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-march=native;-Wall;-Wextra;-fexceptions")
set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fexceptions;--use_fast_math")

add_library(hyacin SHARED ${CPP_SOURCES} ${CU_SOURCES})
target_include_directories(hyacin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_link_libraries(hyacin PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)

if(USE_NCCL)
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

  if (NCCL_INCLUDE_DIRS AND NCCL_LIBRARIES)
    message(STATUS "Found NCCL include: ${NCCL_INCLUDE_DIRS}")
    message(STATUS "Found NCCL library: ${NCCL_LIBRARIES}")
    target_include_directories(hyacin PUBLIC ${NCCL_INCLUDE_DIRS})
    target_link_libraries(hyacin PUBLIC ${NCCL_LIBRARIES})
  else()
    message(FATAL_ERROR "NCCL requested but not found.")
  endif()
else()
  message(STATUS "NCCL: Disabled")
  target_compile_definitions(hyacin PUBLIC "NO_NCCL")
endif()

message(STATUS "Will install to: ${CMAKE_INSTALL_PREFIX}")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/hyacin.h DESTINATION include)
install(TARGETS hyacin EXPORT HyacinTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(EXPORT HyacinTargets FILE HyacinTargets.cmake NAMESPACE Hyacin:: DESTINATION lib/cmake)
export(EXPORT HyacinTargets FILE HyacinTargets.cmake NAMESPACE Hyacin::)

include(CMakePackageConfigHelpers)
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/HyacinConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/HyacinConfig.cmake"
  INSTALL_DESTINATION lib/cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/HyacinConfig.cmake DESTINATION lib/cmake)
