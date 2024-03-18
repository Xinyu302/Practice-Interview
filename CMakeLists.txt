cmake_minimum_required(VERSION 3.10)
project(PRACTICE_INTERVIEW LANGUAGES CXX CUDA)

# 设置项目根目录
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# C++和CUDA的编译参数，可选。
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70 -std=c++11")

# 添加每个子目录
add_subdirectory(cuda_kernel)