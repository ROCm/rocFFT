################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################
# Author: Kent Knox

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

find_package( GTest REQUIRED )
find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT DOUBLE )

set(rocfft_test_source
  gtest_main.cpp
  accuracy_test_1D.cpp
  accuracy_test_2D.cpp
  unit_test.cpp
  template.cpp
  hipfft_test.cpp
  misc/source/test_exception.cpp
  )

set(rocfft_test_includes
  accuracy_test_common.h
  buffer_memory.h
  buffer.h
  rocfft_transform.h
  fftw_transform.h
  rocfft_against_fftw.h
  misc/include/test_exception.h
  )

add_executable( rocfft-test ${rocfft_test_source} ${rocfft_test_includes} )

target_compile_features( rocfft-test PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )
target_include_directories( rocfft-test
  PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${FFTW_INCLUDES}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/include>
)

target_link_libraries( rocfft-test PRIVATE roc::rocfft ${GTEST_LIBRARIES} ${FFTW_LIBRARIES} ${Boost_LIBRARIES} )

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

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocfft-test PRIVATE -Wno-unused-command-line-argument )

  # foreach( target ${AMDGPU_TARGETS} )
  #   target_link_libraries( rocfft-test PRIVATE --amdgpu-target=${target} )
  # endforeach( )
endif( )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( rocfft-test PRIVATE Threads::Threads )

set_target_properties( rocfft-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocfft-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
