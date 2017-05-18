################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

add_executable( fixed-large-float fixed-large-float.cpp )
add_executable( fixed-large-double fixed-large-double.cpp )

if(PLATFORM_NAME STREQUAL "NVIDIA")
    set(hip_include_dir ${HIP_SRC_ROOT}/src/HIP/include)
    add_definitions(-D__NVCC__)
    find_package(CUDA REQUIRED)
    
endif( )

target_include_directories( fixed-large-float
  PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  PRIVATE $<BUILD_INTERFACE:${hip_include_dir}>
  PRIVATE $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
target_include_directories( fixed-large-double
  PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  PRIVATE $<BUILD_INTERFACE:${hip_include_dir}>
  PRIVATE $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
target_link_libraries( fixed-large-float rocfft ${FFTW_LIBRARIES} )
target_link_libraries( fixed-large-double rocfft ${FFTW_LIBRARIES} )
