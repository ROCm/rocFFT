################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# Natively available on including Ubuntu 14.04, OpenSUSE 13.2, CentOS 6.6
cmake_minimum_required( VERSION 2.8.12 )

set (CMAKE_CXX_STANDARD 11)

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/library-package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake /opt/rocm/hip/cmake )

#list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# The superbuild does not build anything itself, all compiling is done in external projects
project( rocfft-superbuild NONE )

# Everything is initially off, so that cache is not initialized until user elects to build
option( BUILD_LIBRARY "Build rocFFT library" OFF )
option( BUILD_CLIENTS "Build rocFFT clients" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocFFT as a shared library" OFF )

set( BOOST_ROOT /opt/boost  CACHE PATH "Specify boost installation dir")

set( HOST_TOOLCHAIN_NAME "clang" CACHE STRING "Compiler toolchain: (clang gcc)" )
set_property( CACHE HOST_TOOLCHAIN_NAME PROPERTY STRINGS clang gcc )

set( DEVICE_TOOLCHAIN_NAME "hipcc" CACHE STRING "Compiler toolchain for library: (hcc hipcc)" )
set_property( CACHE DEVICE_TOOLCHAIN_NAME PROPERTY STRINGS hcc hipcc )

set( PLATFORM_NAME "AMD" CACHE STRING "Device platform: (AMD NVIDIA)")
set_property( CACHE PLATFORM_NAME PROPERTY STRINGS AMD NVIDIA)

message( STATUS "PLATFORM NAME is set to be ${PLATFORM_NAME}")

# Default behavior is to NOT install library, but clients may overload
set( rocfft_INSTALL_COMMAND INSTALL_COMMAND ${CMAKE_COMMAND} -E echo_append )

# Clients are programs provided in this repository, that make use of the library as a library client.  This can include
# but is not limited to benchmarks, tests and samples.
if( BUILD_CLIENTS )
  # Clients need to find and link rocfft; we install it locally instead of globally
  unset( rocfft_INSTALL_COMMAND )
endif()

# standard cmake module to manage dependencies
include( ExternalProject )

# This captures all of the dependencies cmake builds itself
set( rocfft_dependencies )

set( DEVICE_TOOLCHAIN_FILE "" )
if( DEVICE_TOOLCHAIN_NAME STREQUAL "hipcc" )
  if( NOT DEFINED HIP_ROOT )
    include( cmake/external-hip.cmake )
    list( APPEND rocfft_dependencies HIP )
  endif( )

  # configure toolchain file to find the hip compiler with the hip we just downloaded
  configure_file( "${PROJECT_SOURCE_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake.in"
      "${PROJECT_BINARY_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake" )

  set( DEVICE_TOOLCHAIN_FILE "${PROJECT_BINARY_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake" )
else( )
  set( DEVICE_TOOLCHAIN_FILE "${PROJECT_SOURCE_DIR}/cmake/${DEVICE_TOOLCHAIN_NAME}-toolchain.cmake" )
endif( )

# Default arguments that get passed down into all external projects
set( BASE_CMAKE_ARGS
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
 )

# CMAKE_BUILD_TYPE only applies to single configuration build systems
if( NOT CMAKE_CONFIGURATION_TYPES )
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} )
endif( )

if( DEFINED CMAKE_CXX_CLANG_TIDY )
  message( STATUS "clang-tidy enabled: ${CMAKE_CXX_CLANG_TIDY}")
  list( APPEND BASE_CMAKE_ARGS -DCMAKE_CXX_CLANG_TIDY=${CMAKE_CXX_CLANG_TIDY} )
endif( )

#setup kernel generator as an external project
set(KERNELS_INTERNAL_USE "all" CACHE STRING "Supported FFT sizes.Default all.Choose all if mix radices are wanted")
set_property(CACHE KERNELS_INTERNAL_USE PROPERTY STRINGS pow2 pow3 pow5 pow2,3 pow2,5 pow3,5 all)

ExternalProject_Add(
    rocfft-kernel-generator
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/library/src/generator
    BINARY_DIR ${PROJECT_SOURCE_DIR}/library/src/generator
    INSTALL_COMMAND ""
    )

add_custom_command(
    TARGET rocfft-kernel-generator
    COMMAND ./rocfft-kernel-generator ${KERNELS_INTERNAL_USE}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/library/src/generator
    COMMENT "run the generator to generated reqired kernels in ${PROJECT_SOURCE_DIR}/library/src/generator"
)



#set up rocfft dependency
if( BUILD_LIBRARY )

  list(APPEND rocfft_dependencies)
  set( LIBRARY_CMAKE_ARGS
    ${BASE_CMAKE_ARGS}
    -DHOST_TOOLCHAIN_FILE=${HOST_TOOLCHAIN_FILE}
    -DDEVICE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
    -DPLATFORM_NAME=${PLATFORM_NAME}
    -DHIP_ROOT=${HIP_ROOT}
    -DHIP_SRC_ROOT=${HIP_SRC_ROOT}#needed for NV device
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
  )

  # Build the library as an external project
  ExternalProject_Add( rocfft
    DEPENDS ${rocfft_dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/library
    BINARY_DIR library-build
    INSTALL_DIR library-package
    CMAKE_ARGS ${LIBRARY_CMAKE_ARGS}
    ${rocfft_INSTALL_COMMAND}
  )
endif()

add_dependencies(rocfft rocfft-kernel-generator)

# Build clients of the library
if( BUILD_CLIENTS )
  include( clients/cmake/build-options.cmake )

  # Default arguments that get passed down into all external projects
  set( CLIENTS_CMAKE_ARGS
    ${BASE_CMAKE_ARGS}
    -DBUILD_CLIENTS_SAMPLES=${BUILD_CLIENTS_SAMPLES}
    -DBUILD_CLIENTS_RIDER=${BUILD_CLIENTS_RIDER}
    -DBUILD_CLIENTS_SELFTEST=${BUILD_CLIENTS_SELFTEST}
    -DBUILD_CLIENTS_BENCHMARKS=${BUILD_CLIENTS_BENCHMARKS}
    -DBUILD_CLIENTS_TESTS=${BUILD_CLIENTS_TESTS}
    -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/cmake/${HOST_TOOLCHAIN_NAME}-toolchain.cmake
    -DDEVICE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
    -DPLATFORM_NAME=${PLATFORM_NAME}
    -DHIP_ROOT=${HIP_ROOT}
    -DHIP_SRC_ROOT=${HIP_SRC_ROOT}#needed for NV device
    )

    if( DEFINED BOOST_ROOT )
      list( APPEND CLIENTS_CMAKE_ARGS -DBOOST_ROOT=${BOOST_ROOT} )
    endif( )

    if( DEFINED FFTW_ROOT )
      list( APPEND CLIENTS_CMAKE_ARGS -DFFTW_ROOT=${FFTW_ROOT} )
    endif( )

    if( BUILD_LIBRARY )
      ExternalProject_Get_Property( rocfft install_dir )
      list( APPEND CLIENTS_CMAKE_ARGS -DCMAKE_PREFIX_PATH=${install_dir} )
    endif( )

  # Clients are set up as an external project to take advantage of specifying toolchain files.
  # We want cmake to go through it's usual discovery process
  ExternalProject_Add( rocfft-clients
    DEPENDS ${rocfft_dependencies} rocfft
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/clients
    BINARY_DIR clients-build
    INSTALL_DIR clients-package
    CMAKE_ARGS ${CLIENTS_CMAKE_ARGS}
    INSTALL_COMMAND ""
  )
endif( )
