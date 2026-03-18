cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("examples" LANGUAGES CXX)

find_package(BLAS REQUIRED)
enable_testing()

if (USE_NCCL)
find_package(MPI REQUIRED)
set(TEST_SOURCES xlra_example.cpp xsvd_example.cpp xsvd_1dcol_example.cpp xsvd_1drow_example.cpp xsvd_2d_example.cpp)
else()
set(TEST_SOURCES xlra_example.cpp xsvd_example.cpp)
set_source_files_properties(TEST_SOURCES PROPERTIES COMPILE_DEFINITIONS "NO_NCCL")
endif()

set_source_files_properties(${TEST_SOURCES} PROPERTIES COMPILE_OPTIONS 
  "-Wall;-Wextra;-fexceptions")

foreach (test ${TEST_SOURCES})
  get_filename_component (src ${test} NAME_WE)
  add_executable(${src}.app ${test})
  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${src}.app PRIVATE hyacin BLAS::BLAS CUDA::cudart CUDA::cublas CUDA::cusolver)

  if(USE_NCCL)
    target_include_directories(${src}.app PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(${src}.app PRIVATE MPI::MPI_CXX ${NCCL_LIBRARIES})
    add_test(NAME ${src}.app COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${src}.app> ${MPIEXEC_POSTFLAGS})
  else()
    add_test(NAME ${src}.app COMMAND $<TARGET_FILE:${src}.app>)
  endif()
endforeach ()

