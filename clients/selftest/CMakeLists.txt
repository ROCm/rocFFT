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

cmake_minimum_required( VERSION 3.16 )

# This should appear before the project command, because it does not
# use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
endif( )



# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( rocfft-clients-self-tests LANGUAGES CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( NOT TARGET rocfft )
  find_package( rocfft REQUIRED CONFIG PATHS )
endif( )

if( NOT HIP_FOUND )
  find_package( HIP REQUIRED )
endif()

if( NOT ROCM_FOUND )
  find_package( ROCM 0.7.3 REQUIRED )
endif()

include( ROCMInstallTargets )

set( rocfft_selftest_source test_complex.cpp )

set( rocfft_selftest_includes basis_vector.h )

add_executable( rocfft-selftest ${rocfft_selftest_source} ${rocfft_selftest_includes} )

find_package( Boost COMPONENTS program_options REQUIRED)
set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.65.1 1.65 )
set( Boost_USE_STATIC_LIBS OFF )

if( NOT Boost_LIBRARIES )
  # Fixes a bug in Boost's CMAKE where Boost_LIBRARIES is not set.
  set( Boost_LIBRARIES Boost::program_options )
endif()

set( rocfft-selftest_include_dirs
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
  )

set( rocfft-selftest_link_libs 
  ${Boost_LIBRARIES}
  )

include( ../cmake/build-gtest.cmake )

if( BUILD_GTEST OR NOT GTEST_FOUND )
  add_dependencies( rocfft-selftest gtest )
  list( APPEND rocfft-selftest_include_dirs ${GTEST_INCLUDE_DIRS} )
  list( APPEND rocfft-selftest_link_libs ${GTEST_LIBRARIES} )
else()
  list( APPEND rocfft-selftest_include_dirs $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}> )
  list( APPEND rocfft-selftest_link_libs ${GTEST_LIBRARIES} )  
endif()

target_compile_options( rocfft-selftest PRIVATE ${WARNING_FLAGS} )

target_include_directories( rocfft-selftest
  PRIVATE
  ${rocfft-selftest_include_dirs}
  )

target_link_libraries( rocfft-selftest
  PRIVATE
  roc::rocfft 
  ${rocfft-selftest_link_libs}
  )

if( USE_CUDA )
  target_include_directories( rocfft-selftest
    PRIVATE
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-selftest PRIVATE __HIP_PLATFORM_NVCC__ )
endif( )
target_link_libraries( rocfft-selftest PRIVATE ${ROCFFT_CLIENTS_HOST_LINK_LIBS} )

set_target_properties( rocfft-selftest PROPERTIES
  DEBUG_POSTFIX "-d"
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
)

if( ROCFFT_BUILD_SCOPE )
  set( SELFTEST_OUT_DIR "/../staging" )
elseif( ROCFFT_CLIENTS_BUILD_SCOPE )
  set( SELFTEST_OUT_DIR "/../bin" )
else()
  set( SELFTEST_OUT_DIR "/bin" )
endif()
string( CONCAT SELFTEST_OUT_DIR "${PROJECT_BINARY_DIR}" ${SELFTEST_OUT_DIR} )

set_target_properties(rocfft-selftest 
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY
                      ${SELFTEST_OUT_DIR})  

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( rocfft-selftest PRIVATE Threads::Threads )

rocm_install(TARGETS rocfft-selftest COMPONENT tests)
