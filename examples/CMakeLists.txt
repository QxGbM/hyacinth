cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

project ("examples" LANGUAGES CXX)
option(USE_SLURM "Enable NCCL backend with SLURM" ON)
option(USE_MPI "Enable NCCL backend with MPI" ON)

find_package(BLAS REQUIRED)

if (USE_NCCL)
  set(cpp_sources xlra_example.cpp xsvd_example.cpp xsvd_1dcol_example.cpp xsvd_1drow_example.cpp xsvd_2d_example.cpp)
  if (USE_SLURM)
    message(STATUS "BOOTSTRAP: SLURM-POSIX")
  endif()

  if (USE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "BOOTSTRAP: MPI")
  endif()
else()
  set(USE_SLURM OFF)
  set(USE_MPI OFF)
  set(cpp_sources xlra_example.cpp xsvd_example.cpp)
endif()

set_source_files_properties(${cpp_sources} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-fexceptions")

foreach (exe ${cpp_sources})
  get_filename_component(src ${exe} NAME_WE)
  add_executable(${src}.app ${exe})
  target_include_directories(${src}.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${src}.app PRIVATE hyacin BLAS::BLAS CUDA::cudart CUDA::cublas CUDA::cusolver)

  if(USE_NCCL)
    target_include_directories(${src}.app PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(${src}.app PRIVATE ${NCCL_LIBRARIES})

    if (USE_SLURM)
    else()
      target_compile_definitions(${src}.app PRIVATE "NO_POSIX")
    endif()

    if (USE_MPI)
      target_link_libraries(${src}.app PRIVATE MPI::MPI_CXX)
    else()
      target_compile_definitions(${src}.app PRIVATE "NO_MPI")
    endif()
  else()
    target_compile_definitions(${src}.app PRIVATE "NO_NCCL" "NO_POSIX" "NO_MPI")
  endif()
endforeach ()

