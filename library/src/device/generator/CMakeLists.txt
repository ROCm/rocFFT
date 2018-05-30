################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

add_executable( rocfft-kernel-generator generator.main.cpp generator.file.cpp )
target_compile_features( rocfft-kernel-generator PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

set_target_properties( rocfft-kernel-generator PROPERTIES VERSION ${rocfft_VERSION} CXX_EXTENSIONS NO )
set_target_properties( rocfft-kernel-generator PROPERTIES DEBUG_POSTFIX "-d" )

target_include_directories( rocfft-kernel-generator
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
