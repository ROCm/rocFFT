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
# when using hipcc as comiler.  These should be removed in the future.

# If hipcc is set as compiler, we are compiling for CUDA backend and
# nvcc complains of illegal -dumpversion flag because cmake thinks nvcc is a GNU compiler
# A workaround is setting boost compiler explicitely

if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  set( Boost_COMPILER "" )
  set( BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu/" )  # Don't know why we need to specify this
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

find_package( GTest REQUIRED )

set( rocfft_selftest_source test_complex.cpp )

set( rocfft_selftest_includes basis_vector.h )

add_executable( rocfft-selftest ${rocfft_selftest_source} ${rocfft_selftest_includes} )

target_compile_features( rocfft-selftest PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )

target_include_directories( rocfft-selftest
  PRIVATE
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
)

target_link_libraries( rocfft-selftest PRIVATE roc::rocfft ${GTEST_LIBRARIES} ${Boost_LIBRARIES} )

if( CUDA_FOUND )
  target_include_directories( rocfft-selftest
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-selftest PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocfft-selftest PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_link_libraries( rocfft-selftest PRIVATE hip::hip_hcc )
endif( )

if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  target_compile_options( rocfft-selftest PRIVATE -Wno-unused-command-line-argument -hc )
endif( )

if(HIP_PLATFORM STREQUAL "hip-clang")
  target_compile_options( rocfft-selftest PRIVATE -Wno-unused-command-line-argument )
endif( )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( rocfft-selftest PRIVATE Threads::Threads )

set_target_properties( rocfft-selftest PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocfft-selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
