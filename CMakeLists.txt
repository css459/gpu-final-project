project(gpu-final-project C)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_STANDARD 11)

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -ccbin /usr/bin/g++-6
)

# Specify include directories
include_directories(include)
include_directories(src/kernels)

# Specify binary name and source file to build it from
cuda_add_executable(gpu-final-project src/main.cu)