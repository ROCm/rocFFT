# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# Author: Kent Knox

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################

# This builds the generator executable
add_subdirectory( generator )

# This includes an explicit list (all possible) of files the generator currently outputs
include( generator/generated-kernels.cmake )

set( generator_pattern all CACHE STRING "FFT kernels to generate" )
set_property( CACHE generator_pattern PROPERTY STRINGS pow2 pow3 pow5 "pow2,3" "pow2,5" "pow3,5" all )

set( gen_headers ${kernels_launch} ${kernels_all} )
if( generator_pattern STREQUAL "pow2" )
    set( gen_headers ${kernels_launch} ${kernels_pow2} )
elseif( generator_pattern STREQUAL "pow3" )
    set( gen_headers ${kernels_launch} ${kernels_pow3} )
elseif( generator_pattern STREQUAL "pow5" )
    set( gen_headers ${kernels_launch} ${kernels_pow5} )
endif()
# add_custom_command will create source output files, that are used in a target below
# In order for the dependencies to work in cmake, add_custom_command must be in the same
# CMakeLists.txt file as the target [rocfft-device]
add_custom_command(
  OUTPUT ${gen_headers} 
  COMMAND rocfft-kernel-generator ${generator_pattern}
  DEPENDS rocfft-kernel-generator
#  WORKING_DIRECTORY ${kernel_path}
  COMMENT "Generator producing device kernels for rocfft-device"
)

# The following is a list of implementation files defining the library
set( rocfft_device_source
  transpose.cpp
  real2complex.cpp
  complex2real.cpp
  function_pool.cpp
  kernel_launch_single.cpp
  kernel_launch_double.cpp
  kernel_launch_single_large.cpp
  kernel_launch_double_large.cpp
  transpose.cpp
)

prepend_path( "../.." rocfft_headers_public relative_rocfft_device_headers_public )

add_library( rocfft-device
  ${rocfft_device_source}
  ${relative_rocfft_device_headers_public}
  ${gen_headers}
  )
add_library( roc::rocfft-device ALIAS rocfft-device )
target_compile_features( rocfft-device PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

target_link_libraries( rocfft-device PRIVATE hip::hip_hcc hip::hip_device hcc::hccshared )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocfft-device PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( rocfft-device PRIVATE --amdgpu-target=${target} )
  endforeach( )
endif( )

target_include_directories( rocfft-device
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generator>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

set_target_properties( rocfft-device PROPERTIES VERSION ${rocfft_VERSION} SOVERSION ${rocfft_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( rocfft-device PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft-device PROPERTIES DEBUG_POSTFIX "-d" )

# Package that helps me set visibility for function names exported from shared library
# set_target_properties( rocfft-device PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
# generate_export_header( rocfft-device EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocfft-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocfft-device PROPERTIES PREFIX "lib" )
endif( )

rocm_install_targets(
  TARGETS rocfft-device
  PREFIX rocfft
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
