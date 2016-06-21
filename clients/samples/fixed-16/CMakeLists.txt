# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( fixed-16 fixed-16.cpp )

if(PLATFORM_NAME STREQUAL "NVIDIA")
    set(hip_include_dir ${HIP_SRC_ROOT}/src/HIP/include)
    add_definitions(-D__NVCC__)
    find_package(CUDA REQUIRED)
    
endif( )

target_include_directories( fixed-16
  PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  PRIVATE $<BUILD_INTERFACE:${hip_include_dir}>
  PRIVATE $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
target_link_libraries( fixed-16 rocfft ${FFTW_LIBRARIES} )
