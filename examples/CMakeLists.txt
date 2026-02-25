cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("examples" LANGUAGES CXX)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
enable_testing()

set(TEST_SOURCES
  clra_example.cpp cutv_example.cpp
  dlra_example.cpp dutv_example.cpp dutv_1dcol_example.cpp dutv_1drow_example.cpp
  slra_example.cpp sutv_example.cpp
  zlra_example.cpp zutv_example.cpp
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
  add_test(NAME ${src}.app COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${src}.app> ${MPIEXEC_POSTFLAGS})

  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${src}.app PRIVATE hyacin MPI::MPI_CXX BLAS::BLAS LAPACK::LAPACK CUDA::cudart CUDA::cublas CUDA::cusolver)

  if(USE_NCCL)
    target_include_directories(${src}.app PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(${src}.app PRIVATE ${NCCL_LIBRARIES})
  endif()

  if(BLAS_LIBS MATCHES "mkl")
    target_compile_options(${src}.app PRIVATE "-DUSE_MKL" "-DMKL_Complex8=std::complex<float>" "-DMKL_Complex16=std::complex<double>")
  else()
    target_link_libraries(${src}.app PRIVATE lapacke)
  endif()
endforeach ()

