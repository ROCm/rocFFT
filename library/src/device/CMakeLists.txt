# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
  if(WIN32)
    message(
      FATAL_ERROR
      "ROCFFT_DEVICE_FORCE_RELEASE cannot be used on Windows, as the debug and normal runtimes "
      "are ABI-incompatible.  The core rocFFT lib and device libs must both use the same runtime."
    )
  endif()
  set (CMAKE_BUILD_TYPE Release)
endif()

# This builds the generator executable
add_subdirectory( generator )

if(USE_CUDA)
  add_compile_options(--cuda-path=${CUDA_PREFIX} --cuda-gpu-arch=${CUDA_ARCH} -xcuda)
endif()

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
# not specifying any properties generate all size, with all precision

# example 1:
# Adding the following cmd lines to generate only [small-4096], [large-100-sbcc/sbrc] with single precision
# "-DGENERATOR_PATTERN=none"
# "-DGENERATOR_PRECISION=single"
# "-DGENERATOR_MANUAL_SMALL_SIZE=4096"
# "-DGENERATOR_MANUAL_LARGE_SIZE=100"

# example 2:
# Adding the following cmd lines to generate all [2D], [pow2], and [small-336, 56] with double precision
# "-DGENERATOR_PATTERN=2D,pow2"
# "-DGENERATOR_PRECISION=double"
# "-DGENERATOR_MANUAL_SMALL_SIZE=56,336"

# Make it possible to let install.sh control this ?
set( kgen ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py )
set( kgendeps ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py
              ${CMAKE_SOURCE_DIR}/library/src/device/generator.py )
execute_process(COMMAND ${PYTHON3_EXE} ${kgen}
  --pattern=${GENERATOR_PATTERN}
  --precision=${GENERATOR_PRECISION}
  --manual-small=${GENERATOR_MANUAL_SMALL_SIZE}
  --manual-large=${GENERATOR_MANUAL_LARGE_SIZE}
  --runtime-compile-default=${ROCFFT_RUNTIME_COMPILE_DEFAULT}
  list
  OUTPUT_VARIABLE gen_headers
  RESULT_VARIABLE STATUS)
if( STATUS AND NOT STATUS EQUAL 0 )
  message( FATAL_ERROR "Kernel generator failed (list): ${STATUS}")
endif()

# stockham_aot will be having relative RUNPATH with respect to package install directory
# Set LD_LIBRARY_PATH for running the executable from build directory
add_custom_command(OUTPUT ${gen_headers}
  COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}" ${PYTHON3_EXE} ${kgen}
  --pattern=${GENERATOR_PATTERN}
  --precision=${GENERATOR_PRECISION}
  --manual-small=${GENERATOR_MANUAL_SMALL_SIZE}
  --manual-large=${GENERATOR_MANUAL_LARGE_SIZE}
  --runtime-compile-default=${ROCFFT_RUNTIME_COMPILE_DEFAULT}
  generate $<TARGET_FILE:stockham_aot>
  DEPENDS stockham_aot ${kgendeps}
  COMMENT "Generator producing device kernels for rocfft-device"
)

# add virtual build target for generated kernels
add_custom_target(gen_headers_target
  DEPENDS ${gen_headers}
  VERBATIM
)

# device library starts empty and is built from generated files
set( rocfft_device_source
)

set_property(
  SOURCE ${rocfft_device_source}
  PROPERTY COMPILE_OPTIONS ${WARNING_FLAGS}
  )

prepend_path( "../.."
  rocfft_headers_public relative_rocfft_device_headers_public )

option(ROCFFT_CALLBACKS_ENABLED "Enable user-defined callbacks for load/stores from global memory" ON)

# add generated files to the source list - do this after setting
# warning flags so that they apply only to manually-maintained files
list( APPEND rocfft_device_source ${gen_headers} )

# split device lib into 4 so that no single library gets too large to
# link
#
# sort the list to keep the split library contents relatively stable
# over time
list( SORT rocfft_device_source )

# function pool is a generated file, but put it in its own
# library so it's easier to link to.
list( FILTER rocfft_device_source EXCLUDE REGEX function_pool.cpp )
add_library( rocfft-function-pool OBJECT
  function_pool.cpp
)

foreach( pool rocfft-function-pool )
  target_include_directories( ${pool}
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/device>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  )
  set_target_properties( ${pool} PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )
  add_dependencies(${pool} gen_headers_target)
endforeach()

