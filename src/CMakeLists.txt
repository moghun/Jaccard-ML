add_executable(jaccard main.cu edge_metrics.cu gpu_utils.cu)

find_package(OpenMP REQUIRED)

target_include_directories(jaccard PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(jaccard OpenMP::OpenMP_CXX)