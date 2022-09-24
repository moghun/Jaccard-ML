cmake_minimum_required(VERSION 3.12)
project(json_package VERSION 0.1)


set(CMAKE_CXX_STANDARD 11)

set(VALS "int; float; double; long; long long; std::string; long long; unsigned long long; unsigned int; std::vector<unsigned int>; std::vector<int>" CACHE STRING "types for setters and getters")
set(SETS "char*; char const*" CACHE STRING "Types for setters only")

find_package (Python COMPONENTS Interpreter)
list(JOIN VALS "," VALS_JOINED)
list(JOIN SETS "," SETS_JOINED)
execute_process(COMMAND ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_explicit_instantiations.py
    --vals ${VALS_JOINED} 
    --sets ${SETS_JOINED} 
    --output-folder ${CMAKE_CURRENT_BINARY_DIR}/init)
message(STATUS "Generated explicit instantiations")
add_library(json_package json.cc)
target_include_directories(json_package PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/init>
)
