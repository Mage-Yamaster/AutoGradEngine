cmake_minimum_required(VERSION 3.18)
project(AutoGrad LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES all-major)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE LIB_HEADERS "include/*.hpp" "include/*.cuh")

add_library(LibAutoGrad STATIC ${LIB_SOURCES} ${LIB_HEADERS})

target_include_directories(LibAutoGrad PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OpenBLAS_INCLUDE_DIRS}
)

target_link_libraries(LibAutoGrad
        PUBLIC
        OpenBLAS::OpenBLAS
        OpenMP::OpenMP_CXX
        CUDA::cudart
        CUDA::cublas
)

add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE LibAutoGrad)

target_compile_options(LibAutoGrad PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
