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

# Generated kernels

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
# ex: 50, 64, 81, 100, 128, 200, 256
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
# "-DGENERATOR_PATTERN=2D,pow2"
# "-DGENERATOR_PRECISION=double"
# "-DGENERATOR_MANUAL_SMALL_SIZE=56,336"
# "-DGENERATOR_GROUP_NUM=3"

if( NOT DEFINED PYTHON3_EXE )
  set(PYTHON3_EXE python3)
endif()

# Make it possible to let install.sh control this ?
if( NOT GENERATOR_GROUP_NUM )
  set( GENERATOR_GROUP_NUM 150 )
endif()
set( kgen ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py )
set( kgendeps ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py
              ${CMAKE_SOURCE_DIR}/library/src/device/generator.py
              ${CMAKE_SOURCE_DIR}/library/src/device/rtccache.py
              ${CMAKE_SOURCE_DIR}/library/src/device/stockham.py )
execute_process(COMMAND ${PYTHON3_EXE} ${kgen}
  --pattern=${GENERATOR_PATTERN}
  --precision=${GENERATOR_PRECISION}
  --groups=${GENERATOR_GROUP_NUM}
  --manual-small=${GENERATOR_MANUAL_SMALL_SIZE}
  --manual-large=${GENERATOR_MANUAL_LARGE_SIZE}
  --runtime-compile=${ROCFFT_RUNTIME_COMPILE}
  list
  OUTPUT_VARIABLE gen_headers
  RESULT_VARIABLE STATUS)
if( STATUS AND NOT STATUS EQUAL 0 )
  message( FATAL_ERROR "Kernel generator failed (list): ${STATUS}")
endif()

# XXX eventually add dependency to kgen here...
add_custom_command(
  OUTPUT ${gen_headers}
  COMMAND ${PYTHON3_EXE} ${kgen}
  --pattern=${GENERATOR_PATTERN}
  --precision=${GENERATOR_PRECISION}
  --groups=${GENERATOR_GROUP_NUM}
  --manual-small=${GENERATOR_MANUAL_SMALL_SIZE}
  --manual-large=${GENERATOR_MANUAL_LARGE_SIZE}
  --runtime-compile=${ROCFFT_RUNTIME_COMPILE}
  generate $<TARGET_FILE:rocfft-kernel-generator>
  DEPENDS rocfft-kernel-generator ${kgendeps}
  COMMENT "Generator producing device kernels for rocfft-device"
)

# embed the generator itself into c++ files
set( kgen_embed_command ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed-cpp.py )
set( kgen_embed_h ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed.h )
set( kgen_embed_files
     ${kgendeps}
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/common.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/callback.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/butterfly_constant.h
     ${CMAKE_SOURCE_DIR}/library/src/include/real2complex.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rocfft_butterfly_template.h
   )
add_custom_command(
  OUTPUT ${kgen_embed_h}
  COMMAND ${PYTHON3_EXE} ${kgen_embed_command}
  ${kgen_embed_files} ${kgen_embed_h}
  DEPENDS ${kgen_embed_files}
)

# The following is a list of implementation files defining the library
set( rocfft_device_source
  apply_callback.cpp
  transpose.cpp
  bluestein.cpp
  real2complex_embed.cpp
  complex2real_embed.cpp
  realcomplex_even.cpp
  realcomplex_even_transpose.cpp
)

set_property(
  SOURCE ${rocfft_device_source}
  PROPERTY COMPILE_OPTIONS ${WARNING_FLAGS}
  )

prepend_path( "../.."
  rocfft_headers_public relative_rocfft_device_headers_public )

option(ROCFFT_CALLBACKS_ENABLED "Enable user-defined callbacks for load/stores from global memory" ON)
# break down generated headers into "single", "double", and "misc",
# to build 3 libraries
set( gen_headers_single ${gen_headers} )
list( FILTER gen_headers_single INCLUDE REGEX single )
set( gen_headers_double ${gen_headers} )
list( FILTER gen_headers_double INCLUDE REGEX double )

set( gen_headers_misc ${gen_headers} )
list( FILTER gen_headers_misc EXCLUDE REGEX single|double )
# add non-generated stuff to misc
list( APPEND gen_headers_misc ${rocfft_device_source} ${relative_rocfft_device_headers_public} ${kgen_embed_h} )

# we haven't known if there are any old-kernels generated. It could be empty if all are new-kernels
set( BUILD_SINGLE_PRECISION OFF PARENT_SCOPE )
set( BUILD_DOUBLE_PRECISION OFF PARENT_SCOPE )

set( sub_libraries misc )
if(gen_headers_single)
  list( APPEND sub_libraries single )
  set( BUILD_SINGLE_PRECISION ON PARENT_SCOPE )
endif()

if(gen_headers_double)
  list( APPEND sub_libraries double )
  set( BUILD_DOUBLE_PRECISION ON PARENT_SCOPE )
endif()

foreach( sub ${sub_libraries} )
  set( gen_headers_var gen_headers_${sub} )
  if(NOT SINGLELIB)
    add_library( rocfft-device-${sub}
      ${${gen_headers_var}} )
  else()
    # Compile the device lib as a static library, which is then linked
    # into librocfft.so Useful for testing purposes.
    add_library( rocfft-device-${sub} STATIC
      ${${gen_headers_var}} )

    # if we're building singlelib, we don't want to export any of the
    # device library symbols to the main library
    if(NOT HIP_PLATFORM STREQUAL "nvcc")
      set_target_properties( rocfft-device-${sub} PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
    endif()
  endif()

  set_property(TARGET rocfft-device-${sub} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if(ROCFFT_CALLBACKS_ENABLED)
    target_compile_definitions( rocfft-device-${sub} PRIVATE ROCFFT_CALLBACKS_ENABLED )
  endif()

  add_library( roc::rocfft-device-${sub} ALIAS rocfft-device-${sub} )
  target_compile_features( rocfft-device-${sub} PRIVATE
    cxx_static_assert cxx_nullptr cxx_auto_type )

  if( NOT BUILD_SHARED_LIBS )
    target_link_libraries( rocfft-device-${sub} INTERFACE hip::host )
  endif()
  if( WIN32 )
    target_link_libraries( rocfft-device-${sub} PRIVATE hip::device hip::host )
  endif()

  if(HIP_PLATFORM STREQUAL "nvcc")
    target_compile_options( rocfft-device-${sub} PRIVATE
      "-gencode arch=compute_75,code=sm_75"
      "-gencode arch=compute_70,code=sm_70"
      "-gencode arch=compute_60,code=sm_60" )
      target_compile_options( rocfft-device-${sub} PRIVATE -Xcompiler -fPIC)
  else()
    target_compile_options( rocfft-device-${sub} PRIVATE -fno-gpu-rdc )

    # the device libraries are all HIP code
    target_compile_options( rocfft-device-${sub} PRIVATE -x hip )

    # Set AMD GPU architecture options

    # Enable compilation of desired architectures
    foreach( target ${AMDGPU_TARGETS} )
      target_compile_options( rocfft-device-${sub} PRIVATE --offload-arch=${target} )
    endforeach( )

  endif()

  target_include_directories( rocfft-device-${sub}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generator>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  )

  rocm_set_soversion( rocfft-device-${sub} ${rocfft_SOVERSION} )
  set_target_properties( rocfft-device-${sub} PROPERTIES CXX_EXTENSIONS NO )
  set_target_properties( rocfft-device-${sub} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${PROJECT_BINARY_DIR}/staging" )
  set_target_properties( rocfft-device-${sub} PROPERTIES CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON )

  # Following Boost conventions of prefixing 'lib' on static built
  # libraries, across all platforms
  if( NOT BUILD_SHARED_LIBS )
    set_target_properties( rocfft-device-${sub} PROPERTIES PREFIX "lib" )
  endif( )

  rocm_install_targets(
    TARGETS
    rocfft-device-${sub}
    PREFIX
    rocfft )
endforeach()

# misc lib has the function pool which depends on the single/double
# libs
if(gen_headers_single)
  target_link_libraries( rocfft-device-misc PRIVATE rocfft-device-single )
endif()
if(gen_headers_double)
  target_link_libraries( rocfft-device-misc PRIVATE rocfft-device-double )
endif()
