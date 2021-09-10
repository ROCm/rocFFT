################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

add_executable( rocfft-kernel-generator generator.main.cpp generator.file.cpp )

target_compile_options( rocfft-kernel-generator PRIVATE ${WARNING_FLAGS} )

if( NOT USE_CUDA )
  target_link_libraries( rocfft-kernel-generator PRIVATE hip::device )
endif( )

set_target_properties( rocfft-kernel-generator PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocfft-kernel-generator PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
)

target_include_directories( rocfft-kernel-generator
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)
