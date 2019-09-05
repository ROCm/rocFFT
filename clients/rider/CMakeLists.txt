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

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.65.1 1.65 )
set( Boost_USE_STATIC_LIBS OFF )

# ########################################################################
# The following check are hacks to get compiles working for CUDA backend
# These should be removed in the future.

# If hipcc is set as compiler, we are compiling for CUDA backend and
# nvcc complains of illegal -dumpversion flag because cmake thinks nvcc is a GNU compiler
# A workaround is setting boost compiler explicitely

# NVCC can not compile fftw.3 header file appropriately before v3.3.5
# https://github.com/FFTW/fftw3/issues/18
# v3.3.5 is installed by default on Ubuntu 16, workaround is to trick fftw3
# into thinking nvcc is an Intel compiler as desribed in above issue
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  set( Boost_COMPILER "" )
  set( BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu/" )  # Don't know why we need this yet
endif( )
# ########################################################################

find_package( Boost COMPONENTS program_options )

if( NOT Boost_FOUND )
  message( STATUS "Dynamic boost libraries not found. Attempting to find static libraries " )
  set( Boost_USE_STATIC_LIBS ON )
  find_package( Boost COMPONENTS program_options )

  if( NOT Boost_FOUND )
    message( FATAL_ERROR "boost is a required dependency and is not found;  try adding boost path to CMAKE_PREFIX_PATH" )
  endif( )
endif( )

set( rocfft_rider_source rider.cpp misc.cpp )
set( rocfft_rider_includes rider.h misc.h )

add_executable( rocfft-rider ${rocfft_rider_source} ${rocfft_rider_includes} )

target_compile_features( rocfft-rider PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )

target_include_directories( rocfft-rider
PRIVATE
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
)

target_link_libraries( rocfft-rider PRIVATE roc::rocfft ${Boost_LIBRARIES} )

if( CUDA_FOUND )
  target_include_directories( rocfft-rider
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-rider PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocfft-rider PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_link_libraries( rocfft-rider PRIVATE hip::hip_hcc )
endif( )

if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  target_compile_options( rocfft-rider PRIVATE -Wno-unused-command-line-argument -hc)
endif( )

if(HIP_PLATFORM STREQUAL "hip-clang")
  target_compile_options( rocfft-rider PRIVATE -Wno-unused-command-line-argument)
endif( )


set_target_properties( rocfft-rider PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocfft-rider PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
