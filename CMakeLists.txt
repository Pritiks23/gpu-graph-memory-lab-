
---

# 2. `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)

project(
    gpu_graph_memory_lab
    LANGUAGES CXX CUDA
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


find_package(CUDAToolkit REQUIRED)


add_executable(
    graph_benchmark

    src/main.cu
    src/pagerank.cu
)


target_include_directories(
    graph_benchmark
    PRIVATE
    src
)


target_link_libraries(
    graph_benchmark
    PRIVATE
    CUDA::cudart
)


set_target_properties(
    graph_benchmark
    PROPERTIES

    CUDA_SEPARABLE_COMPILATION ON
)
