################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################



cmake_minimum_required( VERSION 2.8.12 )

project( rocfft-kernel-generator CXX C )
 
add_executable(rocfft-kernel-generator generator.main.cpp)

target_compile_options(rocfft-kernel-generator PUBLIC -std=c++11)

target_include_directories(rocfft-kernel-generator 
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set(KERNELS_INTERNAL_USE "all" )
# set(KERNELS_INTERNAL_USE "all" CACHE STRING "Supported FFT sizes.Default all.Choose all if mix radices are wanted")
# set_property(CACHE KERNELS_INTERNAL_USE PROPERTY STRINGS pow2 pow3 pow5 pow2,3 pow2,5 pow3,5 all)


add_custom_command(
    TARGET rocfft-kernel-generator
    COMMAND ./rocfft-kernel-generator ${KERNELS_INTERNAL_USE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "run the generator to generated reqired kernels before building rocfft-device"
)

