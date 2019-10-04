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

find_package( Boost COMPONENTS program_options REQUIRED)
set( Boost_DEBUG ON )
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

if(NOT Boost_LIBRARIES)
  # Fixes a bug in Boost's CMAKE where Boost_LIBRARIES is not set.
  set(Boost_LIBRARIES Boost::program_options)
endif()


find_package( GTest REQUIRED )
find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT DOUBLE )

set(rocfft_test_source
  gtest_main.cpp
  accuracy_test_1D.cpp
  accuracy_test_2D.cpp
  accuracy_test_3D.cpp
  unit_test.cpp
  hipfft_test.cpp
  misc/source/test_exception.cpp
  )

set(rocfft_test_includes
  fftw_transform.h
  rocfft_against_fftw.h
  misc/include/test_exception.h
  )

add_executable( rocfft-test ${rocfft_test_source} ${rocfft_test_includes} )

target_compile_features( rocfft-test
  PRIVATE
  cxx_static_assert
  cxx_nullptr
  cxx_auto_type )

target_include_directories( rocfft-test
  PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${FFTW_INCLUDES}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
)

target_link_libraries( rocfft-test
  PRIVATE
  roc::rocfft
  ${GTEST_LIBRARIES}
  ${FFTW_LIBRARIES}
  ${Boost_LIBRARIES} )

if( CUDA_FOUND )
  target_include_directories( rocfft-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-test PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocfft-test PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_link_libraries( rocfft-test PRIVATE hip::hip_hcc )
endif( )

if(CMAKE_CXX_COMPILER MATCHES ".*/hcc$")
  target_compile_options( rocfft-test PRIVATE -Wno-unused-command-line-argument -hc )
endif( )

if(HIP_PLATFORM STREQUAL "hip-clang")
  target_compile_options( rocfft-test PRIVATE -Wno-unused-command-line-argument )
endif( )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( rocfft-test PRIVATE Threads::Threads)

set_target_properties( rocfft-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocfft-test
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
