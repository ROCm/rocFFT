################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

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

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocfft-selftest PRIVATE -Wno-unused-command-line-argument )

  # foreach( target ${AMDGPU_TARGETS} )
  #   target_link_libraries( rocfft-selftest PRIVATE --amdgpu-target=${target} )
  # endforeach( )
endif( )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( rocfft-selftest PRIVATE Threads::Threads )

set_target_properties( rocfft-selftest PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( rocfft-selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
