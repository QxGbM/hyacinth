cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("hyacinth" LANGUAGES CXX CUDA)
set (SM_TARGETS "all-major")

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES
  src_cpp/geqp3_ronly.cpp
  src_cpp/potrfp_complex.cpp
  src_cpp/potrfp_real.cpp
  src_cpp/qr_params_query.cpp
)

set(CU_SOURCES
  src_cuda/copy_convert_upper.cu
  src_cuda/decode_strided_i32.cu
  src_cuda/encode_fp_int_strided.cu
  src_cuda/fp_exponent.cu
  src_cuda/gemv_scal.cu
  src_cuda/cgemm_pp.cu
  src_cuda/imax_diag.cu
  src_cuda/reduce_scal.cu
  src_cuda/swap_cols.cu
  src_cuda/strided_i8_gemm_real.cu
)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Wall;-Wextra;-fexceptions")
set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fexceptions")

add_library(hyacinth SHARED ${CPP_SOURCES} ${CU_SOURCES})
set_target_properties(hyacinth PROPERTIES CUDA_ARCHITECTURES ${SM_TARGETS})

target_include_directories(hyacinth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hyacinth PRIVATE CUDA::cudart CUDA::cublas)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
enable_testing()

set(TEST_SOURCES
  examples/cgeqp3_example.cpp
  examples/dgeqp3_example.cpp
  examples/sgeqp3_example.cpp
  examples/zgeqp3_example.cpp
)

string(JOIN " " BLAS_LIBS ${BLAS_LIBRARIES})
if(BLAS_LIBS MATCHES "mkl")
  message(STATUS "BLAS is Intel MKL")
endif()

set_source_files_properties(${TEST_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Wall;-Wextra;-fexceptions")

foreach (test ${TEST_SOURCES})
  get_filename_component (src ${test} NAME_WE)
  add_executable(${src}.app ${test})
  add_test(NAME ${src}.app COMMAND ${src}.app)

  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${src}.app PRIVATE hyacinth BLAS::BLAS LAPACK::LAPACK CUDA::cudart CUDA::cublas)
  if(BLAS_LIBS MATCHES "mkl")
    target_compile_options(${src}.app PRIVATE "-DUSE_MKL")
  else()
    target_link_libraries(${src}.app PRIVATE lapacke)
  endif()
  set_target_properties(${src}.app PROPERTIES CUDA_ARCHITECTURES ${SM_TARGETS})
endforeach ()
