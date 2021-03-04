# #############################################################################
# Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################


# A helper function to prefix a source list of files with a common
# path into a new list (non-destructive)
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

option(ROCFFT_DEVICE_FORCE_RELEASE "Force the rocfft-device library to Release build type" OFF)
if(ROCFFT_DEVICE_FORCE_RELEASE)
  set (CMAKE_BUILD_TYPE Release)
endif()

# This builds the generator executable
add_subdirectory( generator )

# This includes an explicit list (all possible) of files the generator
# currently outputs
include( generator/generated-kernels.cmake )


# list of {"pow2" "pow3" "pow5" "pow7" "small" "large" "2D" "all"}, set to "none" if build only manual size
# ex: "pow2" "pow5" "pow7" "2D" will generate pow2,5 + radix7,11,13 + 2D
if( NOT GENERATOR_PATTERN )
  set( GENERATOR_PATTERN "all" )
endif()

# list of {"single" "double" "all"}
if( NOT GENERATOR_PRECISION )
  set( GENERATOR_PRECISION "all" )
endif()

# list of any supported small size, "" means empty
# ex: 1024 4096 336 56 will generat 4 kernels only
if( NOT GENERATOR_MANUAL_SMALL_SIZE )
  set( GENERATOR_MANUAL_SMALL_SIZE "" )
endif()

# list of any supported large size, "" means empty
# supported 64, 81, 100, 128, 200, 256
if( NOT GENERATOR_MANUAL_LARGE_SIZE )
  set( GENERATOR_MANUAL_LARGE_SIZE "" )
endif()

# default:
# not specifying any properties generate all size, with all precision with 150 small kernel groups (cpp)

# example 1:
# Adding the following cmd lines to generate only [small-4096], [large-100-sbcc/sbrc] with single precision
# And GENERATOR_GROUP_NUM=8 generates 8 small kernel groups (cpp)
# "-DGENERATOR_PATTERN=none"
# "-DGENERATOR_PRECISION=single"
# "-DGENERATOR_MANUAL_SMALL_SIZE=4096"
# "-DGENERATOR_MANUAL_LARGE_SIZE=100"
# "-DGENERATOR_GROUP_NUM=8"

# example 2:
# Adding the following cmd lines to generate all [2D], [pow2], and [small-336, 56] with double precision
# And GENERATOR_GROUP_NUM=3 generates 3 small kernel groups (cpp)
# "-DGENERATOR_PATTERN=2D;pow2"
# "-DGENERATOR_PRECISION=double"
# "-DGENERATOR_MANUAL_SMALL_SIZE=56;336"
# "-DGENERATOR_GROUP_NUM=3"


list( LENGTH GENERATOR_MANUAL_SMALL_SIZE num_manual_small )
list( LENGTH GENERATOR_MANUAL_LARGE_SIZE num_manual_large )

set( gen_headers ${kernels_launch_basic} )
if( "all" IN_LIST GENERATOR_PATTERN )
    if( "all" IN_LIST GENERATOR_PRECISION )
        list(APPEND gen_headers ${kernels_launch_all} )
    else()
        if( "single" IN_LIST GENERATOR_PRECISION )
            list(APPEND gen_headers ${kernels_launch_single} )
        endif()
        if( "double" IN_LIST GENERATOR_PRECISION )
            list(APPEND gen_headers ${kernels_launch_double} )
        endif()
    endif()
else()
    if( "large" IN_LIST GENERATOR_PATTERN OR num_manual_large GREATER 0)
        if( "all" IN_LIST GENERATOR_PRECISION )
            list(APPEND gen_headers ${kernels_launch_large_all} )
        else()
            if( "single" IN_LIST GENERATOR_PRECISION )
                list(APPEND gen_headers ${kernels_launch_large_single} )
            endif()
            if( "double" IN_LIST GENERATOR_PRECISION )
                list(APPEND gen_headers ${kernels_launch_large_double} )
            endif()
        endif()
    endif()

    if( "2D" IN_LIST GENERATOR_PATTERN )
        if( "all" IN_LIST GENERATOR_PRECISION )
            list(APPEND gen_headers ${kernels_launch_2D_all} )
        else()
            if( "single" IN_LIST GENERATOR_PRECISION )
                list(APPEND gen_headers ${kernels_launch_2D_single} )
            endif()
            if( "double" IN_LIST GENERATOR_PRECISION )
                list(APPEND gen_headers ${kernels_launch_2D_double} )
            endif()
        endif()
    endif()

    if( "small" IN_LIST GENERATOR_PATTERN OR "pow2" IN_LIST GENERATOR_PATTERN OR
        "pow3" IN_LIST GENERATOR_PATTERN OR "pow5" IN_LIST GENERATOR_PATTERN OR
        "pow7" IN_LIST GENERATOR_PATTERN OR num_manual_small GREATER 0)
        if( "all" IN_LIST GENERATOR_PRECISION )
            list(APPEND gen_headers ${kernels_launch_small_all} )
        else()
            if( "single" IN_LIST GENERATOR_PRECISION )
                list(APPEND gen_headers ${kernels_launch_small_single} )
            endif()
            if( "double" IN_LIST GENERATOR_PRECISION )
                list(APPEND gen_headers ${kernels_launch_small_double} )
            endif()
        endif()
    endif()

endif()

# add_custom_command will create source output files, that are used in
# a target below
# In order for the dependencies to work in cmake, add_custom_command
# must be in the same CMakeLists.txt file as the target
# [rocfft-device]

string(REPLACE ";" "," PATTERN_ARG "${GENERATOR_PATTERN}")
string(REPLACE ";" "," PRECISION_ARG "${GENERATOR_PRECISION}")
string(REPLACE ";" "," MANUAL_SMALL_ARG "${GENERATOR_MANUAL_SMALL_SIZE}")
string(REPLACE ";" "," MANUAL_LARGE_ARG "${GENERATOR_MANUAL_LARGE_SIZE}")

set( manual_size_small_arg "" )
if( num_manual_small GREATER 0 )
  set( manual_size_small_arg "--manual-small" "${MANUAL_SMALL_ARG}" )
endif()

set( manual_size_large_arg "" )
if( num_manual_large GREATER 0 )
  set( manual_size_large_arg "--manual-large" "${MANUAL_LARGE_ARG}" )
endif()

add_custom_command(
  OUTPUT ${gen_headers}
  COMMAND rocfft-kernel-generator
  ${manual_size_small_arg}
  ${manual_size_large_arg}
  "-t" "${PATTERN_ARG}"
  "-p" "${PRECISION_ARG}"
  "-g" ${GENERATOR_GROUP_NUM}
  DEPENDS rocfft-kernel-generator
  COMMENT "Generator producing device kernels for rocfft-device"
)

# The following is a list of implementation files defining the library
set( rocfft_device_source
  transpose.cpp
  bluestein.cpp
  real2complex_embed.cpp
  complex2real_embed.cpp
  realcomplex_even.cpp
  realcomplex_pair.cpp
  realcomplex_even_transpose.cpp
)

prepend_path( "../.."
  rocfft_headers_public relative_rocfft_device_headers_public )

if(NOT SINGLELIB)
  add_library( rocfft-device
    ${rocfft_device_source}
    ${relative_rocfft_device_headers_public}
    ${gen_headers} )
else()
  # Compile the device lib as a static library, which is then linked
  # into librocfft.so Useful for testing purposes.
  add_library( rocfft-device STATIC
    ${rocfft_device_source}
    ${relative_rocfft_device_headers_public}
    ${gen_headers}
    )
  set_property(TARGET rocfft-device PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

add_library( roc::rocfft-device ALIAS rocfft-device )
target_compile_features( rocfft-device PRIVATE
  cxx_static_assert cxx_nullptr cxx_auto_type )

if( NOT BUILD_SHARED_LIBS )
  target_link_libraries( rocfft-device INTERFACE hip::host )
endif()

if(HIP_PLATFORM STREQUAL "nvcc")
  target_compile_options( rocfft-device PRIVATE
    "-gencode arch=compute_75,code=sm_75"
    "-gencode arch=compute_70,code=sm_70"
    "-gencode arch=compute_60,code=sm_60" )
    target_compile_options( rocfft-device PRIVATE -Xcompiler -fPIC)
else()
  target_compile_options( rocfft-device PRIVATE -fno-gpu-rdc )

  # Set AMD GPU architecture options

  # Enable compilation of desired architectures
  foreach( target ${AMDGPU_TARGETS} )
    target_compile_options( rocfft-device PRIVATE --amdgpu-target=${target} )
  endforeach( )

endif()

target_include_directories( rocfft-device
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generator>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

rocm_set_soversion( rocfft-device ${rocfft_SOVERSION} )
set_target_properties( rocfft-device PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocfft-device PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft-device PROPERTIES CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON )

# Following Boost conventions of prefixing 'lib' on static built
# libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocfft-device PROPERTIES PREFIX "lib" )
endif( )

rocm_install_targets(
  TARGETS
  rocfft-device
  PREFIX
  rocfft )

rocm_install_symlink_subdir( rocfft )
