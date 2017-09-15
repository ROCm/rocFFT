################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set( kernel_path "${CMAKE_CURRENT_BINARY_DIR}" )

add_executable( rocfft-kernel-generator generator.main.cpp generator.file.cpp )

# Test for specific compiler features if cmake version is recent enough
target_compile_features( rocfft-kernel-generator PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

set_target_properties( rocfft-kernel-generator PROPERTIES VERSION ${rocfft_VERSION} CXX_EXTENSIONS NO )
set_target_properties( rocfft-kernel-generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft-kernel-generator PROPERTIES DEBUG_POSTFIX "-d" )

target_include_directories(rocfft-kernel-generator
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

set( KERNELS_INTERNAL_USE "all" CACHE STRING "FFT kernels to generate" )
set_property( CACHE KERNELS_INTERNAL_USE PROPERTY STRINGS pow2 pow3 pow5 "pow2,3" "pow2,5" "pow3,5" all )

add_custom_command(
    TARGET rocfft-kernel-generator
    POST_BUILD
    COMMAND ${PROJECT_BINARY_DIR}/staging/rocfft-kernel-generator ${KERNELS_INTERNAL_USE}
    BYPRODUCTS kernel_launch_generator.h
    WORKING_DIRECTORY ${kernel_path}
    COMMENT "Generator producing device kernels for rocfft-device"
)

