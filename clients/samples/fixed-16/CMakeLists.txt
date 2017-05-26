################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

add_executable( fixed-16-float  fixed-16-float.cpp )
add_executable( fixed-16-double fixed-16-double.cpp )

if(PLATFORM_NAME STREQUAL "NVIDIA")
    set(hip_include_dir ${HIP_SRC_ROOT}/src/HIP/include)
    add_definitions(-D__NVCC__)
    find_package(CUDA REQUIRED)
    
endif( )

target_include_directories( fixed-16-float
  PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  PRIVATE $<BUILD_INTERFACE:${hip_include_dir}>
  PRIVATE $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
target_include_directories( fixed-16-double
  PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  PRIVATE $<BUILD_INTERFACE:${hip_include_dir}>
  PRIVATE $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
target_link_libraries( fixed-16-float rocfft ${FFTW_LIBRARIES} )
target_link_libraries( fixed-16-double rocfft ${FFTW_LIBRARIES} )
