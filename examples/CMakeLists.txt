cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("examples" LANGUAGES CXX CUDA)
set(SM_TARGETS "80;86;89;90;100;120")

find_package(CUDAToolkit REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
enable_testing()

set(APP_SOURCES
  geqp3.cu
  interp_decomp.cu
)

add_library(example_common OBJECT ${APP_SOURCES})
set_property(TARGET example_common PROPERTY CUDA_ARCHITECTURES ${SM_TARGETS})
target_include_directories(example_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(example_common PRIVATE hyacin CUDA::cudart CUDA::cublas CUDA::cusolver)

set(TEST_SOURCES
  cgeqp3_example.cpp clra_example.cpp
  dgeqp3_example.cpp dlra_example.cpp
  sgeqp3_example.cpp slra_example.cpp
  zgeqp3_example.cpp zlra_example.cpp
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

  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${src}.app PRIVATE hyacin example_common BLAS::BLAS LAPACK::LAPACK CUDA::cudart CUDA::cublas CUDA::cusolver)

  if(BLAS_LIBS MATCHES "mkl")
    target_compile_options(${src}.app PRIVATE "-DUSE_MKL")
  else()
    target_link_libraries(${src}.app PRIVATE lapacke)
  endif()
endforeach ()

