cmake_minimum_required(VERSION 3.12)

project(Jaccard-ML VERSION 0.1 LANGUAGES CUDA CXX)

enable_language(CUDA)
find_package(CUDA REQUIRED)

option(_CPU "Use CPU kernels" ON)
option(_GPU "Use GPU kernels" ON)
option(_SIMPLE_GPU "Simple GPU kernel" OFF)
option(_SIMPLE_GPU_EDGE "Edge-based GPU kernel" OFF)
option(_CUGRAPH "use cuGraph" OFF)
option(_INHOUSE_CUGRAPH "use inhouse cuGraph" OFF)
option(_DONGARRA "use dongarra" OFF)
option(_DIRECTED "Treat graphs as directed" OFF)
option(_BINNING "Run binning experiment" ON)

if (NOT ${_GPU})
set(_SIMPLE_GPU OFF)
set(_SIMPLE_GPU_EDGE OFF)
set(_CUGRAPH OFF)
set(_INHOUSE_CUGRAPH OFF)
set(_DONGARRA OFF)
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/src/config.h.in"
    "${PROJECT_BINARY_DIR}/src/config.h"
)

add_executable(jaccard src/main.cu src/edge_metrics.cu src/gpu_utils.cu)

find_package(OpenMP REQUIRED)

target_include_directories(jaccard PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(jaccard OpenMP::OpenMP_CXX)