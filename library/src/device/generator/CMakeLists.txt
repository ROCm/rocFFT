################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# set( staging_path "${PROJECT_BINARY_DIR}/staging" )

# set( kernel_path "${staging_path}/generated-kernels" )
# file( MAKE_DIRECTORY ${kernel_path} )

add_executable( rocfft-kernel-generator generator.main.cpp generator.file.cpp )

# Test for specific compiler features if cmake version is recent enough
target_compile_features( rocfft-kernel-generator PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

set_target_properties( rocfft-kernel-generator PROPERTIES VERSION ${rocfft_VERSION} CXX_EXTENSIONS NO )
# set_target_properties( rocfft-kernel-generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${staging_path}" )
set_target_properties( rocfft-kernel-generator PROPERTIES DEBUG_POSTFIX "-d" )

target_include_directories( rocfft-kernel-generator
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)

# add_custom_command(
#     TARGET rocfft-kernel-generator
#     POST_BUILD
#     COMMAND ${staging_path}/rocfft-kernel-generator ${generator_pattern}
#     BYPRODUCTS ${kernels_launch} ${kernels_1xxx} ${kernels_2xxx} ${kernels_3xxx_4xxx} ${kernels_5xxx_6xxx_7xxx_8xxx_9xxx}
#     WORKING_DIRECTORY ${kernel_path}
#     COMMENT "Generator producing device kernels for rocfft-device"
# )
