################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# We require 2.8.12 for linking libraries we specify PRIVATE, INTERFACE and
# PUBLIC keywords
cmake_minimum_required( VERSION 2.8.12 )

set (CMAKE_CXX_STANDARD 11)

if( CMAKE_GENERATOR MATCHES "NMake" )
	option( NMAKE_COMPILE_VERBOSE "Print VERBOSE compile/link msgs to the console" OFF )
	if( NMAKE_COMPILE_VERBOSE )
		set( CMAKE_START_TEMP_FILE "" )
		set( CMAKE_END_TEMP_FILE "" )
		set( CMAKE_VERBOSE_MAKEFILE 1 )
	endif( )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT CMAKE_CONFIGURATION_TYPES )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

#set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -DTMP_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

if( DEFINED HOST_TOOLCHAIN_FILE )
endif( )

if( DEFINED DEVICE_TOOLCHAIN_FILE )
	set( CMAKE_TOOLCHAIN_FILE "${DEVICE_TOOLCHAIN_FILE}" )
endif( )

# Check if cmake supports the new VERSION tag for project() commands
# rocfft becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( rocfft VERSION 0.0.2.0 LANGUAGES CXX C )
else( )
	project( rocfft CXX C )
	# Define a version for the code
	if( NOT DEFINED rocfft_VERSION_MAJOR )
		set( rocfft_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED rocfft_VERSION_MINOR )
		set( rocfft_VERSION_MINOR 0 )
	endif( )

	if( NOT DEFINED rocfft_VERSION_PATCH )
		set( rocfft_VERSION_PATCH 2 )
	endif( )

	if( NOT DEFINED rocfft_VERSION_TWEAK )
		set( rocfft_VERSION_TWEAK 0 )
	endif( )

	set( rocfft_VERSION "${rocfft_VERSION_MAJOR}.${rocfft_VERSION_MINOR}.${rocfft_VERSION_PATCH}.${rocfft_VERSION_TWEAK}")
endif( )

if(PLATFORM_NAME STREQUAL "NVIDIA")
    find_package(CUDA REQUIRED)
endif( )

# MACOSX_RPATH is enabled by default policy
if( POLICY CMP0042 )
	cmake_policy( SET CMP0042 NEW )
endif( )

message( STATUS "rocfft_VERSION= ${rocfft_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )

# This is incremented when the ABI to the library changes
set( rocfft_SOVERSION 1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
# Need to check out CMAKE_RUNTIME_OUTPUT_DIRECTORY variable, and see if that eliminates the need to modify install path
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# Set common compile and link options
if( MSVC )
	# Following options for nMake
	message( STATUS "Detected MSVS Ver: " ${MSVC_VERSION} )

	# CMake uses huge stack frames for windows, we would like to remove.
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" )
endif( )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")

include ( build-bitness )

# Print out compiler flags for viewing/debug
message( STATUS "CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
message( STATUS "CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
message( STATUS "CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
message( STATUS "CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
message( STATUS "CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )

include( GNUInstallDirs )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${PROJECT_SOURCE_DIR}/include/rocfft-version.h.in" "${PROJECT_BINARY_DIR}/include/rocfft-version.h" )

set( rocfft_headers_public
  include/rocfft.h
  include/rocfft_transpose.h
  include/hipfft.h
  ${PROJECT_BINARY_DIR}/include/rocfft-version.h
)

source_group( "Header Files\\Public" FILES ${rocfft_headers_public} )

# Build into subdirectories
add_subdirectory( src )

install( FILES ${rocfft_headers_public}
	DESTINATION
		${CMAKE_INSTALL_INCLUDEDIR} )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
	set( CPACK_SOURCE_GENERATOR "ZIP" )
	set( CPACK_GENERATOR "ZIP" )
else( )
	set( CPACK_SOURCE_GENERATOR "TGZ" )

    set( CPACK_GENERATOR "DEB;RPM" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
	# set( CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON )
endif( )

set( CPACK_PACKAGE_CONTACT "Kent Knox <kent.knox@amd.com>" )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute FFT library")
set( CPACK_PACKAGE_VERSION_MAJOR ${rocfft_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${rocfft_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${rocfft_VERSION_PATCH} )
set( CPACK_PACKAGE_VERSION_TWEAK ${rocfft_VERSION_TWEAK} )
set( CPACK_PACKAGE_VERSION ${rocfft_VERSION} )
set( CPACK_PACKAGE_VENDOR "AMD")
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/../LICENSE.md" )
set( CPACK_SOURCE_IGNORE_FILES "/\\\\.git/" )

# Package specific CPACK vars
set( CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.0.17174)" )
set( CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.0.17174" )

if( "${CPACK_PACKAGING_INSTALL_PREFIX}" MATCHES "^/opt/rocm.*$")
    include( package-functions )
    write_rocm_package_script_files( ${PROJECT_BINARY_DIR} "rocfft" "librocfft-hcc.so" )

    set( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${PROJECT_BINARY_DIR}/postinst;${PROJECT_BINARY_DIR}/prerm")
    set( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/postinst" )
    set( CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/prerm" )
endif( )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )
