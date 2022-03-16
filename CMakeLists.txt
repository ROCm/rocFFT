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



# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()


project( rocfft LANGUAGES CXX C )


# Control hip-clang use:
set( USE_HIP_CLANG OFF CACHE BOOL "Use hip-clang to build for amdgpu" )
if( USE_HIP_CLANG )
  message( STATUS "Use hip-clang to build for amdgpu backend" )
  set( HIP_PLATFORM "hip-clang" )
  set( HIP_COMPILER "clang" )
else()
  set( HIP_PLATFORM "hcc" )
endif()

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
      ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  message(STATUS "downloading... done")

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzvf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif( )
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds )
include( ROCMClients )

# Using standardized versioning from rocm-cmake
set ( VERSION_STRING "1.0.16" )
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

option(ROCFFT_RUNTIME_COMPILE "Enable runtime compilation of kernels" ON)

if(BUILD_ADDRESS_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()


set( WARNING_FLAGS -Wall -Wno-unused-function -Wimplicit-fallthrough -Wunreachable-code )
if( WERROR )
  set( WARNING_FLAGS ${WARNING_FLAGS} -Werror )
endif( )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL
  "Installation directory for libraries" FORCE)

# Use target ID syntax if supported for AMDGPU_TARGETS
rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
  TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030")
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")

# HIP is required - library and clients use HIP to access the device
find_package( HIP REQUIRED )

# The nvidia backend can be used to compile for CUDA devices.
# Specify the CUDA prefix in the CUDA_PREFIX variable.
# CUDA_ARCH (e.g. sm_75) is also required.
if( USE_CUDA )
  if( NOT DEFINED CUDA_PREFIX )
    message( FATAL_ERROR "CUDA_PREFIX variable is required (e.g. /usr/local/cuda-11.4)" )
  endif()
  if( NOT DEFINED CUDA_ARCH )
    message( FATAL_ERROR "CUDA_ARCH variable is required. (e.g. sm_75)" )
  endif()
  add_compile_options(-I${HIP_ROOT_DIR}/include -I${CUDA_PREFIX}/include -D__HIP_PLATFORM_NVIDIA__)
  add_link_options(-L${CUDA_PREFIX}/lib64 -pthread)
endif( )


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
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SELFTEST)
    rocm_package_setup_client_component(tests DEPENDS COMPONENT clients-common)
  endif()
  if(BUILD_CLIENTS_RIDER)
    rocm_package_setup_client_component(benchmarks DEPENDS COMPONENT clients-common)
    rocm_install(
      DIRECTORY scripts/perf
      DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT benchmarks
    )
  endif()
  add_subdirectory( clients )
endif( )

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  set(INSTALL_PREFIX "C:/hipSDK")
  set(CPACK_SET_DESTDIR OFF)
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
  set(CPACK_PACKAGING_INSTALL_PREFIX "")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
endif()

# Package specific CPACK vars
string( TOLOWER "${HIP_RUNTIME}" HIP_RUNTIME_LOWER )
if( HIP_RUNTIME_LOWER STREQUAL "rocclr" )
  rocm_package_add_dependencies("hip-rocclr >= 3.5.0")
endif( )
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT" )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib/cmake" )

set( ROCFFT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

set( package_name rocfft )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm FFT library"
    MAINTAINER "rocfft-maintainer@amd.com"
    LDCONFIG
    LDCONFIG_DIR ${ROCFFT_CONFIG_DIR}
)
