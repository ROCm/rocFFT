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

# This project may compile dependencies for clients
project( rocfft-clients LANGUAGES CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# if rocfft is not a target, then we know clients are built separately from the library and we must search
# for the rocfft package
if( NOT TARGET rocfft )
  find_package( rocfft REQUIRED CONFIG PATHS /opt/rocm/rocfft )
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
find_package( hip REQUIRED CONFIG PATHS /opt/rocm )

# Quietly look for CUDA, but if not found it's not an error
# The presense of hip is not sufficient to determine if we want a rocm or cuda backend
find_package( CUDA QUIET )

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

# This is a simple and temporary way to package all clients with dpkg on ubuntu.
# We need a better solution to share with other projects in rocm-cmake.
include(CMakeParseArguments)

function(rocm_create_package_clients)
    set(options)
    set(oneValueArgs LIB_NAME DESCRIPTION SECTION MAINTAINER VERSION)
    set(multiValueArgs DEPENDS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(CONCAT PACKAGE_NAME ${PARSE_LIB_NAME} "-clients-" ${PARSE_VERSION} "-Linux.deb")
    string(CONCAT DEB_CONTROL_FILE_CONTENT "Package: " ${PARSE_LIB_NAME} "-clients"
                                           "\nVersion: " ${PARSE_VERSION}
                                           "\nSection: " ${PARSE_SECTION}
                                           "\nPriority: optional"
                                           "\nArchitecture: amd64"
                                           "\nMaintainer: " ${PARSE_MAINTAINER}
                                           "\nDescription: " ${PARSE_DESCRIPTION}
                                           "\nDepends: " ${PARSE_LIB_NAME} "(=" ${PARSE_VERSION} ")\n\n")

    if(EXISTS "${PROJECT_BINARY_DIR}/package")
        file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/package")
    endif()
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/package/opt/rocm/${PARSE_LIB_NAME}/clients")
    file(WRITE "${PROJECT_BINARY_DIR}/package/DEBIAN/control" ${DEB_CONTROL_FILE_CONTENT})

    add_custom_target(package_clients
        COMMAND ${CMAKE_COMMAND} -E remove -f "${PROJECT_BINARY_DIR}/package/opt/rocm/${PARSE_LIB_NAME}/clients/*"
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/*" "${PROJECT_BINARY_DIR}/package/opt/rocm/${PARSE_LIB_NAME}/clients"
        COMMAND dpkg -b "${PROJECT_BINARY_DIR}/package/"  ${PACKAGE_NAME})
endfunction(rocm_create_package_clients)


if (BUILD_CLIENTS_SAMPLES OR  BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SELFTEST OR BUILD_CLIENTS_RIDER)

rocm_create_package_clients(LIB_NAME rocfft
                            DESCRIPTION "Radeon Open Compute FFT library"
                            MAINTAINER "Kent Knox <kent.knox@amd.com>"
                            SECTION "dev"
                            VERSION ${rocfft_VERSION})
endif()
