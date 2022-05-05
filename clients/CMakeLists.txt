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

set( ROCFFT_CLIENTS_BUILD_SCOPE ON )

# This project may compile dependencies for clients
project( rocfft-clients LANGUAGES CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )


# This option only works for make/nmake and the ninja generators, but
# no reason it shouldn't be on all the time.
# This tells cmake to create a compile_commands.json file that can be
# used with clang tooling or vim.
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if(NOT ROCFFT_BUILD_SCOPE AND 
   NOT BUILD_CLIENTS_SAMPLES AND 
   NOT BUILD_CLIENTS_TESTS AND 
   NOT BUILD_CLIENTS_SELFTEST AND 
   NOT BUILD_CLIENTS_RIDER)
  set( BUILD_CLIENTS_SAMPLES ON )
  set( BUILD_CLIENTS_TESTS ON )
  set( BUILD_CLIENTS_SELFTEST ON )
  set( BUILD_CLIENTS_RIDER ON )
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

if (BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SELFTEST)
  include( cmake/build-gtest.cmake )
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
    "${PROJECT_BINARY_DIR}/package/${CPACK_PACKAGING_INSTALL_PREFIX}/share/${PARSE_LIB_NAME}/clients")
  file(WRITE
    "${PROJECT_BINARY_DIR}/package/DEBIAN/control"
    ${DEB_CONTROL_FILE_CONTENT})

  add_custom_target(package_clients
    COMMAND ${CMAKE_COMMAND} -E remove
    -f "${PROJECT_BINARY_DIR}/package/${CPACK_PACKAGING_INSTALL_PREFIX}/share/${PARSE_LIB_NAME}/clients/*"
    COMMAND ${CMAKE_COMMAND} -E copy
    "${PROJECT_BINARY_DIR}/staging/*"
    "${PROJECT_BINARY_DIR}/package/${CPACK_PACKAGING_INSTALL_PREFIX}/share/${PARSE_LIB_NAME}/clients"
    COMMAND dpkg -b "${PROJECT_BINARY_DIR}/package/"  ${PACKAGE_NAME})
endfunction(rocm_create_package_clients)


if ( (BUILD_CLIENTS_SAMPLES OR
      BUILD_CLIENTS_TESTS OR
      BUILD_CLIENTS_SELFTEST OR
      BUILD_CLIENTS_RIDER) 
      AND ROCFFT_BUILD_SCOPE )
  rocm_create_package_clients(LIB_NAME rocfft
    DESCRIPTION "ROCm FFT library client programs"
    MAINTAINER "rocfft-maintainer@amd.com"
    SECTION "dev"
    VERSION ${rocfft_VERSION})
endif()
