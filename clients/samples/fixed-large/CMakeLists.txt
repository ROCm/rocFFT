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

# ########################################################################
# The following check are hacks to get compiles working for CUDA backend
# These should be removed in the future.

# NVCC can not compile fftw.3 header file appropriately before v3.3.5
# https://github.com/FFTW/fftw3/issues/18
# v3.3.5 is installed by default on Ubuntu 16, workaround is to trick fftw3
# into thinking nvcc is an Intel compiler as desribed in above issue
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  add_definitions( -D__INTEL_COMPILER )
endif( )
# ########################################################################

set( sample_list fixed-large-float fixed-large-double )

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cpp )

  target_include_directories( ${sample}
    PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )

  target_link_libraries( ${sample} PRIVATE roc::rocfft ${FFTW_LIBRARIES} )

  target_compile_features( ${sample} PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

  set_target_properties( ${sample} PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
  set_target_properties( ${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

  if( CUDA_FOUND )
    target_include_directories( ${sample}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
      )
    target_compile_definitions( ${sample} PRIVATE __HIP_PLATFORM_NVCC__ )
    target_link_libraries( ${sample} PRIVATE ${CUDA_LIBRARIES} )
  else( )
    target_link_libraries( ${sample} PRIVATE hip::hip_hcc )
  endif( )


if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  target_compile_options( ${sample} PRIVATE -Wno-unused-command-line-argument -hc )
endif( )

if(HIP_PLATFORM STREQUAL "hip-clang")
  target_compile_options( ${sample} PRIVATE -Wno-unused-command-line-argument )
endif( )

endforeach( )
