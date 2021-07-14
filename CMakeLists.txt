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

# Workaround for missing llvm cmake files in /opt/rocm/lib/cmake. Remove when this is fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )


# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()


project( rocfft LANGUAGES CXX )


# Control hip-clang use:
set( USE_HIP_CLANG OFF CACHE BOOL "Use hip-clang to build for amdgpu" )
if( USE_HIP_CLANG )
  message( STATUS "Use hip-clang to build for amdgpu backend" )
  set( HIP_PLATFORM "hip-clang" )
  set( HIP_COMPILER "clang" )
else()
  set( HIP_PLATFORM "hcc" )
endif()

# The nvcc backend can be used to compile for CUDA devices.
if( USE_CUDA )
  # NB: CUDA has first-class since cmake 2.8; find_package(CUDA) is deprecated.
  # See, eg, https://cmake.org/cmake/help/v3.12/module/FindCUDA.html
  set( ENV{HIP_PLATFORM} "nvcc" )
  set( HIP_PLATFORM "nvcc" )
  set( CMAKE_C_COMPILE_OPTIONS_PIC
    -Xcompiler ${CMAKE_C_COMPILE_OPTIONS_PIC} )
  set( CMAKE_CXX_COMPILE_OPTIONS_PIC
    -Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_PIC} )
  set( CMAKE_SHARED_LIBRARY_C_FLAGS
    -Xlinker ${CMAKE_SHARED_LIBRARY_C_FLAGS} )
  set( CMAKE_SHARED_LIBRARY_CXX_FLAGS "-Xlinker -fPIC" )
  set( CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
endif( )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help
# setup and install
find_package( ROCM CONFIG )
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds OPTIONAL )

# Using standardized versioning from rocm-cmake
set ( VERSION_STRING "1.0.13" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# Append our library helper cmake path and the cmake path for hip (for
# convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# Enable verbose output
option( BUILD_VERBOSE "Output additional build information" OFF )

# Build the reference C++ kernel implementations.
option( BUILD_CPUREF "Build CPU reference debug code" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option
# such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocFFT as a shared library" ON )

option( WERROR "Treat warnings as errors" OFF )

option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)

option(ROCFFT_RUNTIME_COMPILE "Enable runtime compilation of kernels" OFF)

if(BUILD_ADDRESS_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()


if( WERROR )
  set( WARNING_FLAGS -Wall -Wno-unused-function -Werror )
else( )
  set( WARNING_FLAGS -Wall -Wno-unused-function )
endif( )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL
  "Installation directory for libraries" FORCE)

# Set the AMDGPU_TARGETS with backward compatiblity
# Use target ID syntax if supported for AMDGPU_TARGETS
if(COMMAND rocm_check_target_ids)
  rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
    TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
  )
else()
  # Detect compiler support for target ID
  # This section is deprecated. Please use rocm_check_target_ids for future use.
  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
      OUTPUT_VARIABLE CXX_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
  endif()
  if(TARGET_ID_SUPPORT)
    set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx1030")
  else()
    set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908")
  endif()
endif()
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")

# Find HCC/HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
  message( STATUS "Building with ROCm tools" )
  find_package( hip REQUIRED CONFIG PATHS )
endif( )

# Hip headers required of all clients; clients use hip to allocate
# device memory
find_package( hip REQUIRED CONFIG PATHS )

add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# Build clients of the library
if( BUILD_CLIENTS )
  set( BUILD_CLIENTS_RIDER ON )
  set( BUILD_CLIENTS_SAMPLES ON )
  set( BUILD_CLIENTS_SELFTEST ON )
  set( BUILD_CLIENTS_TESTS ON )
endif( )

if( BUILD_CLIENTS_SAMPLES
    OR BUILD_CLIENTS_TESTS
    OR BUILD_CLIENTS_SELFTEST
    OR BUILD_CLIENTS_RIDER )
  add_subdirectory( clients )
endif( )

