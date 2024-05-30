# #############################################################################
# Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

# We use C++17 features, this will add compile option: -std=c++17
set( CMAKE_CXX_STANDARD 17 )

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

set( ROCFFT_BUILD_SCOPE ON )

project( rocfft LANGUAGES CXX C )

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
include( ROCMHeaderWrapper )

# Using standardized versioning from rocm-cmake
set ( VERSION_STRING "1.0.28" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# Append our library helper cmake path and the cmake path for hip (for
# convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# Enable verbose output
option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option
# such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocFFT as a shared library" ON )

option( WERROR "Treat warnings as errors" OFF )

option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)

option(ROCFFT_RUNTIME_COMPILE_DEFAULT "Compile kernels at runtime by default" OFF)

# Using -DROCFFT_BUILD_OFFLINE_TUNER=ON to compile an executable,
# Set default to OFF since users are not likely to tune
option(ROCFFT_BUILD_OFFLINE_TUNER "Build with offline tuner executable rocfft_offline_tuner" OFF)

if(BUILD_ADDRESS_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

set( WARNING_FLAGS -Wall -Wno-unused-function -Wimplicit-fallthrough -Wunreachable-code -Wsign-compare )
if( WERROR )
  set( WARNING_FLAGS ${WARNING_FLAGS} -Werror )
endif( )

# Use target ID syntax if supported for AMDGPU_TARGETS
rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
  TARGETS "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102")
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
list(LENGTH AMDGPU_TARGETS AMDGPU_TARGETS_LENGTH)

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

# hipcc automatically provides HIP include dirs and HIP platform,
# but plain clang needs to be told
if( NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  include_directories( ${HIP_INCLUDE_DIRS} )
  if( USE_CUDA )
    add_compile_definitions( __HIP_PLATFORM_NVIDIA__ )
  else()
    add_compile_definitions( __HIP_PLATFORM_AMD__ )
  endif()
endif()

add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# Build clients of the library
if( BUILD_CLIENTS )
  set( BUILD_CLIENTS_BENCH ON )
  set( BUILD_CLIENTS_SAMPLES ON )
  set( BUILD_CLIENTS_TESTS ON )
endif( )

# old name for BUILD_CLIENTS_BENCH
if( BUILD_CLIENTS_RIDER )
  set( BUILD_CLIENTS_BENCH ${BUILD_CLIENTS_RIDER} )
endif()

if( BUILD_CLIENTS_SAMPLES
    OR BUILD_CLIENTS_TESTS
    OR BUILD_CLIENTS_BENCH )
  if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SELFTEST OR BUILD_CLIENTS_BENCH)
    find_package( Boost COMPONENTS program_options REQUIRED)
    set(BOOST_DEB "libboost-program-options${Boost_VERSION_MAJOR}.${Boost_VERSION_MINOR}.${Boost_VERSION_PATCH}")
    set(BOOST_RPM "boost-program-options = ${Boost_VERSION_MAJOR}.${Boost_VERSION_MINOR}.${Boost_VERSION_PATCH}")
  endif()
  if( NOT CLIENTS_OS )
    rocm_set_os_id( CLIENTS_OS )
  endif()
  if(BUILD_CLIENTS_TESTS AND (NOT DEFINED BUILD_CLIENTS_TESTS_OPENMP OR BUILD_CLIENTS_TESTS_OPENMP))
    set(OPENMP_DEB "libgomp1")
    set(FFTW_DEB "libfftw3-bin")
    if(CLIENTS_OS STREQUAL "sles")
      set(OPENMP_RPM "libgomp1")
      set(FFTW_RPM "libfftw3-3")
    else()
      set(OPENMP_RPM "libgomp")
      set(FFTW_RPM "fftw-libs")
    endif()
  endif()
  if(CLIENTS_OS STREQUAL "sles")
    set(BOOST_RPM RPM "libboost_program_options${Boost_VERSION_MAJOR}_${Boost_VERSION_MINOR}_${Boost_VERSION_PATCH}")
  endif()
  rocm_package_setup_component(clients)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        DEB ${BOOST_DEB} ${OPENMP_DEB} ${FFTW_DEB} rocrand
        RPM ${BOOST_RPM} ${OPENMP_RPM} ${FFTW_RPM} rocrand
    )
  endif()
  if(BUILD_CLIENTS_BENCH)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        DEB ${BOOST_DEB} rocrand
        RPM ${BOOST_RPM} rocrand
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
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
  endif()
  rocm_package_add_dependencies("${DEPENDS_HIP_RUNTIME} >= 4.5.0")
endif( )
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT" )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

set( ROCFFT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

set( package_name rocfft )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm FFT library"
    MAINTAINER "rocfft-maintainer@amd.com"
    LDCONFIG
    LDCONFIG_DIR ${ROCFFT_CONFIG_DIR}
)
