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

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has
# cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# This should appear before the project command, because it does not
# use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
  set( CPACK_PACKAGING_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
  set( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# This project may compile dependencies for clients
project( rocfft-clients LANGUAGES CXX )


list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )


# This option only works for make/nmake and the ninja generators, but
# no reason it shouldn't be on all the time.
# This tells cmake to create a compile_commands.json file that can be
# used with clang tooling or vim.
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# If rocfft is not a target, then we know clients are built separately
# from the library and we must search for the rocfft package.
if( NOT TARGET rocfft )
  find_package( rocfft REQUIRED CONFIG PATHS )
endif( )

# Hip headers required of all clients; clients use hip to allocate
# device memory.
# NB: 2020-04-17: Using hip-clang compiled from source requires that
# this not be added twice.
if( NOT HIP_FOUND)
  find_package( HIP REQUIRED )
endif()

if( BUILD_CLIENTS_TESTS  OR BUILD_CLIENTS_SELFTEST )
  find_package( GTest 1.10.0 )
  include( ExternalProject )
  if( NOT GTEST_FOUND)
    set( GTEST_INCLUDE_DIRS
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest/googletest/include
      )
    set( GTEST_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
      )

    set(GTEST_SRC_URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz CACHE STRING "Location of GTest source code")
    set(GTEST_SRC_SHA256 9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb CACHE STRING "SHA256 hash of GTest source code")

    ExternalProject_Add( gtest
      URL ${GTEST_SRC_URL}
      URL_HASH SHA256=${GTEST_SRC_SHA256}
      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${GTEST_LIBRARIES}
      )
    ExternalProject_Get_Property( gtest source_dir binary_dir )
 endif()
endif()

option( BUILD_FFTW "Download and build FFTW" OFF )

if( BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SAMPLES )
 # look for installed FFTW if we weren't asked to build it
 if( NOT BUILD_FFTW )
   find_package( FFTW 3.0 MODULE COMPONENTS FLOAT DOUBLE )
 endif()

 # also try to build FFTW if FFTW isn't present
 if( BUILD_FFTW OR NOT FFTW_FOUND )
   set( FFTW_LIBRARIES_DOUBLE
     ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_double-build/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3_threads${CMAKE_STATIC_LIBRARY_SUFFIX}
     ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_double-build/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3${CMAKE_STATIC_LIBRARY_SUFFIX}
   )
   set( FFTW_LIBRARIES_SINGLE
     ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_single-build/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3f_threads${CMAKE_STATIC_LIBRARY_SUFFIX}
     ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_single-build/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3f${CMAKE_STATIC_LIBRARY_SUFFIX}
   )

   set( FFTW_CMAKE_ARGS_COMMON
     -DDISABLE_FORTRAN=ON
     -DENABLE_AVX2=ON
     -DENABLE_THREADS=ON
     -DBUILD_SHARED_LIBS=OFF
     -DBUILD_TESTS=OFF
     -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
   )

   set(FFTW_SRC_URL http://www.fftw.org/fftw-3.3.9.tar.gz CACHE STRING "Location of FFTW source code")
   set(FFTW_SRC_SHA256 bf2c7ce40b04ae811af714deb512510cc2c17b9ab9d6ddcf49fe4487eea7af3d CACHE STRING "SHA256 hash of FFTW source code")

   # build double-precision FFTW
   ExternalProject_Add( fftw_double
     URL ${FFTW_SRC_URL}
     URL_HASH SHA256=${FFTW_SRC_SHA256}
     SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fftw
     PREFIX ${CMAKE_CURRENT_BINARY_DIR}
     CMAKE_ARGS ${FFTW_CMAKE_ARGS_COMMON}
     INSTALL_COMMAND ""
     BUILD_BYPRODUCTS ${FFTW_LIBRARIES_DOUBLE}
     )
   ExternalProject_Get_Property( fftw_double source_dir binary_dir )

   # also build single-precision fftw from the same source dir
   ExternalProject_Add( fftw_single
     DOWNLOAD_COMMAND ""
     SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fftw
     PREFIX ${CMAKE_CURRENT_BINARY_DIR}
     CMAKE_ARGS ${FFTW_CMAKE_ARGS_COMMON} -DENABLE_FLOAT=ON
     INSTALL_COMMAND ""
     BUILD_BYPRODUCTS ${FFTW_LIBRARIES_SINGLE}
     DEPENDS fftw_double
     )
   ExternalProject_Get_Property( fftw_single source_dir binary_dir )

   set( FFTW_INCLUDES
     ${CMAKE_CURRENT_BINARY_DIR}/src/fftw/api
     )
   set( FFTW_LIBRARIES
     ${FFTW_LIBRARIES_DOUBLE}
     ${FFTW_LIBRARIES_SINGLE}
     )

   # FFTW we build is always threaded
   set( FFTW_MULTITHREAD TRUE )
 endif()
endif()

# each backend requires different libraries for host and device code
if( USE_CUDA )
  if( NOT DEFINED CUDA_PREFIX )
    message( FATAL_ERROR "CUDA_PREFIX variable is required." )
  endif()
  if( NOT DEFINED CUDA_ARCH )
    message( FATAL_ERROR "CUDA_ARCH variable is required." )
  endif()
  add_compile_options(-I${HIP_ROOT_DIR}/include -I${CUDA_PREFIX}/include -D__HIP_PLATFORM_NVIDIA__)
  add_link_options(-L${CUDA_PREFIX}/lib64 -pthread)
  add_compile_options(--cuda-path=${CUDA_PREFIX} --cuda-gpu-arch=${CUDA_ARCH} -xcuda)
  set( ROCFFT_CLIENTS_HOST_LINK_LIBS -lcudart -ldl -lrt )
else()
  set( ROCFFT_CLIENTS_HOST_LINK_LIBS hip::host )
  set( ROCFFT_CLIENTS_DEVICE_LINK_LIBS hip::device )
endif()

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_TESTS )
  add_subdirectory( tests )
endif( )

if( BUILD_CLIENTS_SELFTEST )
  add_subdirectory( selftest )
endif( )

if( BUILD_CLIENTS_RIDER )
  add_subdirectory( rider )
endif( )

# This is a simple and temporary way to package all clients with dpkg
# on ubuntu.
# We need a better solution to share with other projects in rocm-cmake.
include(CMakeParseArguments)

function(rocm_create_package_clients)
  set(options)
  set(oneValueArgs LIB_NAME DESCRIPTION SECTION MAINTAINER VERSION)
  set(multiValueArgs DEPENDS)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  string(CONCAT PACKAGE_NAME ${PARSE_LIB_NAME} "-clients-" ${PARSE_VERSION}
    "-Linux.deb")
  string(CONCAT DEB_CONTROL_FILE_CONTENT
    "Package: " ${PARSE_LIB_NAME} "-clients"
    "\nVersion: " ${PARSE_VERSION}
    "\nSection: " ${PARSE_SECTION}
    "\nPriority: optional"
    "\nArchitecture: amd64"
    "\nMaintainer: " ${PARSE_MAINTAINER}
    "\nDescription: " ${PARSE_DESCRIPTION}
    "\nDepends: " ${PARSE_LIB_NAME} "(>=" ${PARSE_VERSION} ")\n\n")

  if(EXISTS "${PROJECT_BINARY_DIR}/package")
    file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/package")
  endif()
  file(MAKE_DIRECTORY
    "${PROJECT_BINARY_DIR}/package/${CPACK_PACKAGING_INSTALL_PREFIX}/${PARSE_LIB_NAME}/clients")
  file(WRITE
    "${PROJECT_BINARY_DIR}/package/DEBIAN/control"
    ${DEB_CONTROL_FILE_CONTENT})

  add_custom_target(package_clients
    COMMAND ${CMAKE_COMMAND} -E remove
    -f "${PROJECT_BINARY_DIR}/package/${CPACK_PACKAGING_INSTALL_PREFIX}/${PARSE_LIB_NAME}/clients/*"
    COMMAND ${CMAKE_COMMAND} -E copy
    "${PROJECT_BINARY_DIR}/staging/*"
    "${PROJECT_BINARY_DIR}/package/${CPACK_PACKAGING_INSTALL_PREFIX}/${PARSE_LIB_NAME}/clients"
    COMMAND dpkg -b "${PROJECT_BINARY_DIR}/package/"  ${PACKAGE_NAME})
endfunction(rocm_create_package_clients)


if (BUILD_CLIENTS_SAMPLES OR
    BUILD_CLIENTS_TESTS OR
    BUILD_CLIENTS_SELFTEST OR
    BUILD_CLIENTS_RIDER)
  rocm_create_package_clients(LIB_NAME rocfft
    DESCRIPTION "ROCm FFT library client programs"
    MAINTAINER "rocfft-maintainer@amd.com"
    SECTION "dev"
    VERSION ${rocfft_VERSION})
endif()
