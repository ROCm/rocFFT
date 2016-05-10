# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( transpose_outplace_1024 transpose_outplace_1024.cpp )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

if(PLATFORM_NAME STREQUAL "NVIDIA")
    set(hip_include_dir ${HIP_ROOT}/include)
    add_definitions(-D__NVCC__)
    find_package(CUDA REQUIRED)
    
endif( )

target_include_directories( transpose_outplace_1024
  PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  PRIVATE $<BUILD_INTERFACE:${hip_include_dir}>
  PRIVATE $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
target_link_libraries( transpose_outplace_1024 rocfft ${FFTW_LIBRARIES} )
