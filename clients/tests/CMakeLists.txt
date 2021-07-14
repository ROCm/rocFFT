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


set( rocfft-test_source
  gtest_main.cpp
  accuracy_test.cpp
  accuracy_test_1D.cpp
  accuracy_test_2D.cpp
  accuracy_test_3D.cpp
  accuracy_test_adhoc.cpp
  accuracy_test_callback.cpp
  multithread_test.cpp
  unit_test.cpp
  misc/source/test_exception.cpp
  )

set( rocfft-test_includes
  fftw_transform.h
  rocfft_against_fftw.h
  misc/include/test_exception.h
  )

add_executable( rocfft-test ${rocfft-test_source} ${rocfft-test_includes} )

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

find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT DOUBLE )

set( rocfft-test_include_dirs
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${FFTW_INCLUDES}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
  )

set( rocfft-test_link_libs
  ${FFTW_LIBRARIES}
  ${Boost_LIBRARIES}
  )

if( GTEST_FOUND)
  list( APPEND rocfft-test_include_dirs $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}> )
  list( APPEND rocfft-test_link_libs ${GTEST_LIBRARIES} )
else()
  add_dependencies( rocfft-test gtest )
  list( APPEND rocfft-test_include_dirs ${GTEST_INCLUDE_DIRS} )
  list( APPEND rocfft-test_link_libs ${GTEST_LIBRARIES} )
endif()

target_compile_features( rocfft-test
  PRIVATE
  cxx_static_assert
  cxx_nullptr
  cxx_auto_type
  )
target_compile_options( rocfft-test PRIVATE ${WARNING_FLAGS} )

if( ROCFFT_RUNTIME_COMPILE )
  target_compile_options( rocfft-test PRIVATE -DROCFFT_RUNTIME_COMPILE )
endif()

target_include_directories( rocfft-test
  PRIVATE
  ${rocfft-test_include_dirs}
  )


if( NOT BUILD_SHARED_LIBS )
  #target_link_libraries(rocfft-test INTERFACE hip::host)
  list(APPEND rocfft-test_link_libs hip::host)
endif()

target_link_libraries( rocfft-test
  PRIVATE
  roc::rocfft
  ${rocfft-test_link_libs}
  )

if( USE_CUDA )
  target_include_directories( rocfft-test
    PRIVATE
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-test PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocfft-test PRIVATE ${CUDA_LIBRARIES} )
else()
  target_link_libraries( rocfft-test PRIVATE hip::device )
endif( )

option( BUILD_CLIENTS_TESTS_OPENMP "Build tests with OpenMP" ON )

if( BUILD_CLIENTS_TESTS_OPENMP )
  target_compile_options( rocfft-test PRIVATE -fopenmp -DBUILD_CLIENTS_TESTS_OPENMP )
  target_link_libraries( rocfft-test PRIVATE -fopenmp -L${HIP_CLANG_ROOT}/lib -Wl,-rpath=${HIP_CLANG_ROOT}/lib )
  target_include_directories( rocfft-test PRIVATE ${HIP_CLANG_ROOT}/include )
endif()

set_target_properties( rocfft-test
  PROPERTIES DEBUG_POSTFIX "-d"
  CXX_EXTENSIONS NO
  CXX_STANDARD 14
  )

set_target_properties( rocfft-test
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

if (WIN32)

  # Ensure tests run with HIP DLLs and not anything the driver owns
  # in system32.  Libraries like amdhip64.dll are also in the HIP
  # runtime, and we need run with those.  But the only way to make a
  # same-named DLL override something in system32 is to have it next
  # to the executable.  So copy them in.
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET rocfft-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging )
  endforeach( file_i )
endif()
