cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("hyacinth" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CPP_SOURCES potrfp_real.cpp potrfp_complex.cpp qr_params_query.cpp geqp3_ronly.cpp)
set(CU_SOURCES imax_real.cu imax_complex.cu gemv_scal.cu reduce_scal.cu swap_cols.cu copy_convert_upper.cu fp_exponent.cu encode_fp_int_strided.cu decode_strided_i32.cu strided_i8_gemm_real.cu strided_i8_gemm_complex.cu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_libraries(CUDA::cudart CUDA::cublas)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Wall;-Wextra;-fexceptions")
set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fexceptions")

add_library(hyacinth SHARED ${CPP_SOURCES} ${CU_SOURCES})
set_target_properties(hyacinth PROPERTIES CUDA_ARCHITECTURES all-major)

