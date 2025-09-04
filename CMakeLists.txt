cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("hyacinth" LANGUAGES CXX CUDA)
set(SM_TARGETS "80;90;100;120")

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES
  src_cpp/check_id.cpp
  src_cpp/cublas_preload.cpp
  src_cpp/gemv_cublas.cpp
  src_cpp/geqp3_ronly.cpp
  src_cpp/igemm_params.cpp
  src_cpp/igemm.cpp
  src_cpp/interp_decomp.cpp
  src_cpp/potrfp_uniform.cpp
)

set(CU_SOURCES
  src_cuda/decode_strided_i32.cu
  src_cuda/encode_fp_int_strided.cu
  src_cuda/fp_exponent.cu
  src_cuda/gemv_pp_nopiv.cu
  src_cuda/gemv_pp.cu
  src_cuda/gemv_scal_f128.cu
  src_cuda/igemm_normalize.cu
  src_cuda/igemm_pp.cu
  src_cuda/imax_diag.cu
  src_cuda/utils.cu
)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Wall;-Wextra;-fexceptions")
set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fexceptions")

add_library(hyacinth SHARED ${CPP_SOURCES} ${CU_SOURCES})
set_property(TARGET hyacinth PROPERTY CUDA_ARCHITECTURES ${SM_TARGETS})

target_include_directories(hyacinth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hyacinth PRIVATE CUDA::cudart CUDA::cublas)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
enable_testing()

set(TEST_SOURCES
  examples/cgeqp3_example.cpp examples/clra_example.cpp
  examples/dgeqp3_example.cpp examples/dlra_example.cpp
  examples/sgeqp3_example.cpp examples/slra_example.cpp
  examples/zgeqp3_example.cpp examples/zlra_example.cpp
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
endforeach ()
