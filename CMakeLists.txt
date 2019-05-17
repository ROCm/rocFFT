# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################
# Author: Kent Knox

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( rocfft LANGUAGES CXX )

# ########################################################################
# NOTE:  CUDA compiling path
# ########################################################################
# I have tried compiling rocFFT library source with multiple methods,
# and ended up using the approach where we set the CXX compiler to hipcc.
# I didn't like using the HIP_ADD_LIBRARY or CUDA_ADD_LIBRARY approaches,
# for the reasons I list here.
# 1.  Adding header include directories is through HIP_INCLUDE_DIRECTORIES(), which
# is global to a directory and affects all targets
# 2.  You must add HIP_SOURCE_PROPERTY_FORMAT OBJ properties to .cpp files
# to get HIP_ADD_LIBRARY to recognize the file
# 3.  HIP_ADD_LIBRARY invokes a call to add_custom_command() to compile files,
# and rocFFT does the same.  The order in which custom commands execute is
# undefined, and sometimes a file is attempted to be compiled before it has
# been generated.  The fix for this is to create 'PHONY' targets, which I
# don't desire.

# Using hipcc allows us to avoid the above problems, with two primary costs:
# 1.  The cmake logic to detect compiler features fails with nvcc backend
# 2.  Upfront cost to figure out all the strange compiler/linker flags I define
# below.

# Hopefully, cost #2 is already paid.  All in all, I want to get rid of the
# need for hipcc, and hope that at some point of time in the future we
# can use the export config files from hip & hcc for both ROCm & nvcc backends.
# ########################################################################

# ########################################################################
# Main
# ########################################################################
set( USE_HIP_CLANG OFF CACHE BOOL "Use hip-clang to build for amdgpu" )
if( USE_HIP_CLANG )
  message( STATUS "Use hip-clang to build for amdgpu backend" )
  set( HIP_PLATFORM "hip-clang" )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  # For now, we assume hipcc compiler means to compile for CUDA backend
  message( STATUS "HIPCC compiler detected; CUDA backend selected" )

  set( HIP_PLATFORM "nvcc" )
  set( CMAKE_C_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_SHARED_LIBRARY_C_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_CXX_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}" )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}" )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  message( STATUS "HCC compiler set; ROCm backend selected [ CXX=/opt/rocm/bin/hcc cmake ... ]" )
endif( )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )


set ( VERSION_STRING "0.9.3." )
# Check if BUILD_NUMBER is defined in a Jenkins envirnment
if ($ENV{BUILD_NUMBER} )
  string(CONCAT BUILD_VERSION ${VERSION_STRING} $ENV{BUILD_NUMBER}) 
else ()
  string(CONCAT BUILD_VERSION ${VERSION_STRING} "0") 
endif ()
rocm_setup_version( VERSION ${BUILD_VERSION} NO_GIT_TAG_VERSION )

# Get the git tag version
include ( cmake/version.cmake )
rocm_get_git_commit_id ( ${PROJECT_NAME}_VERSION_COMMIT_ID )



# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# NOTE:  workaround until hcc & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip )

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocFFT as a shared library" ON )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Find HCC/HIP dependencies
if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
  message( STATUS "Building with ROCm tools" )
  find_package( hcc REQUIRED CONFIG PATHS /opt/rocm )
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
find_package( hip REQUIRED CONFIG PATHS /opt/rocm )

# Quietly look for CUDA, but if not found it's not an error
# The presense of hip is not sufficient to determine if we want a rocm or cuda backend
find_package( CUDA QUIET )

# CMake list of machine targets
set( AMDGPU_TARGETS gfx803;gfx900;gfx906 CACHE STRING "List of specific machine types for library to target" )

add_subdirectory( library )

include( clients/cmake/build-options.cmake )


# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SELFTEST OR BUILD_CLIENTS_SAMPLES )
  add_subdirectory( clients )
endif( )

