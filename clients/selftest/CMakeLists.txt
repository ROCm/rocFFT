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

if( GTEST_FOUND)
  list( APPEND rocfft-selftest_include_dirs $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}> )
  list( APPEND rocfft-selftest_link_libs ${GTEST_LIBRARIES} )
else()
  add_dependencies( rocfft-selftest gtest )
  list( APPEND rocfft-selftest_include_dirs ${GTEST_INCLUDE_DIRS} )
  list( APPEND rocfft-selftest_link_libs ${GTEST_LIBRARIES} )
endif()

target_compile_features( rocfft-selftest
  PRIVATE
  cxx_static_assert
  cxx_nullptr cxx_lambdas
  cxx_auto_type
  )
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
  target_link_libraries( rocfft-selftest PRIVATE ${CUDA_LIBRARIES} )
endif( )

set_target_properties( rocfft-selftest PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocfft-selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/staging"
  )
