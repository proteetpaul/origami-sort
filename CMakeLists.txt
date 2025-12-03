cmake_minimum_required(VERSION 3.12)
project(Origami VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use GCC compiler
set(CMAKE_CXX_COMPILER "g++")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler-specific options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format -Wno-register -Wno-unused-variable -Wno-unused-parameter \
    -Wno-ignored-attributes")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found - parallel features may not work")
endif()

# Source directory
set(ORIGAMI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Origami)

# Collect all header files
file(GLOB HEADER_FILES
    "${ORIGAMI_SOURCE_DIR}/*.h"
)

# Common source files (shared headers)
set(COMMON_SOURCES
    ${HEADER_FILES}
)

# Preprocessor definitions
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-D_CONSOLE)

# Detect CPU features and set appropriate flags
include(CheckCXXCompilerFlag)

# Check for AVX-512 support
check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)

# Function to create an executable with appropriate flags
function(add_origami_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${COMMON_SOURCES})
    
    # Set include directories
    target_include_directories(${target_name} PRIVATE ${ORIGAMI_SOURCE_DIR})
    
    # Link OpenMP if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Set compiler flags based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Release build: Use AVX2 by default (can be overridden)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(${target_name} PRIVATE -mavx2)
        endif()

        if(COMPILER_SUPPORTS_AVX512)
            target_compile_options(${target_name} PRIVATE -mavx512f)
        endif()
        
        if(COMPILER_SUPPORTS_AVX512VL)
            target_compile_options(${target_name} PRIVATE -mavx512vl)
        endif()
        # Enable OpenMP in Release
        if(OpenMP_CXX_FOUND)
            target_compile_options(${target_name} PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
    else()
        # Debug build: Use AVX512 for debugging (as per VS project)
        if(COMPILER_SUPPORTS_AVX512)
            target_compile_options(${target_name} PRIVATE -mavx512f)
        endif()
    endif()
    
    # Additional optimization flags for Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${target_name} PRIVATE
            -funroll-loops
            -ffast-math
        )
    endif()
endfunction()

# Create executables for each benchmark
# bench_sort.cpp - main benchmark (included in Release|x64, Debug|x64)
add_origami_executable(bench_sort "${ORIGAMI_SOURCE_DIR}/bench_sort.cpp")

# bench_mtree.cpp - excluded from Release|x64, Debug|x64 in VS project
# But we'll include it anyway as it's useful
add_origami_executable(bench_mtree "${ORIGAMI_SOURCE_DIR}/bench_mtree.cpp")

# bench_bmerge.cpp - excluded from Release|x64, Debug|x64, Debug|Win32
# But we'll include it for completeness
add_origami_executable(bench_bmerge "${ORIGAMI_SOURCE_DIR}/bench_bmerge.cpp")

# bench_sort_phases.cpp - excluded from Release|x64, Debug|x64
# But we'll include it for completeness
add_origami_executable(bench_sort_phases "${ORIGAMI_SOURCE_DIR}/bench_sort_phases.cpp")

# Print configuration summary
message(STATUS "")
message(STATUS "=== Origami Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
if(COMPILER_SUPPORTS_AVX512)
    message(STATUS "AVX-512: Supported")
endif()
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2: Supported")
endif()
if(COMPILER_SUPPORTS_SSE42)
    message(STATUS "SSE4.2: Supported")
endif()
message(STATUS "===================================")
message(STATUS "")

