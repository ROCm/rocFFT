################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# We require 2.8.12 for linking libraries we specify PRIVATE, INTERFACE and
# PUBLIC keywords
cmake_minimum_required( VERSION 2.8.12 )

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
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Check if cmake supports the new VERSION tag for project() commands
# rocfft becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( rocfft-test-correctness VERSION 0.0.0.1 LANGUAGES C CXX )
else( )
	project( rocfft-test-correctness CXX C )
	# Define a version for the code
	if( NOT DEFINED rocfft_VERSION_MAJOR )
		set( rocfft-test_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED rocfft_VERSION_MINOR )
		set( rocfft-test_VERSION_MINOR 0 )
	endif( )

	if( NOT DEFINED rocfft_VERSION_PATCH )
		set( rocfft-test_VERSION_PATCH 0 )
	endif( )

	if( NOT DEFINED rocfft_VERSION_TWEAK )
		set( rocfft-test_VERSION_TWEAK 1 )
	endif( )

	set( rocfft-test_VERSION "${rocfft-test_VERSION_MAJOR}.${rocfft-test_VERSION_MINOR}.${rocfft-test_VERSION_PATCH}.${rocfft-test_VERSION_TWEAK}")
endif( )

include ( build-bitness )

# Modify the global find property to help us find libraries like Boost in the correct paths for 64-bit
# Essentially, find_library calls will look for /lib64 instead of /lib; works for windows and linux
if( BUILD_64 )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
	message( STATUS "FIND_LIBRARY_USE_LIB64_PATHS{TRUE} - 64-bit builds" )
else( )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
	message( STATUS "FIND_LIBRARY_USE_LIB64_PATHS{FALSE} - 32-bit builds" )
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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.60.0 1.60 )
set( Boost_USE_STATIC_LIBS ON )
find_package( Boost REQUIRED COMPONENTS program_options )

if( NOT Boost_FOUND )
	set( Boost_USE_STATIC_LIBS OFF )
	find_package( Boost REQUIRED COMPONENTS program_options )
endif( )

find_package( GTest REQUIRED )
find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT )
find_package( rocfft REQUIRED CONFIG )

if(PLATFORM_NAME STREQUAL "NVIDIA")
    set(hip_include_dir ${HIP_SRC_ROOT}/src/HIP/include)
    add_definitions(-D__NVCC__)
    find_package(CUDA REQUIRED)
endif( )

set(rocfft_test_source
    gtest_main.cpp
    accuracy_test_transpose.cpp
    accuracy_test_pow2.cpp
    misc/source/test_exception.cpp)

set(rocfft_test_includes
    accuracy_test_common.h
    buffer_memory.h
    buffer.h 
    rocfft_transform.h
    fftw_transform.h
    rocfft_against_fftw.h
    misc/include/test_exception.h)

add_executable( rocfft-test-correctness ${rocfft_test_source} ${rocfft_test_includes} )

# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
  target_compile_features( rocfft-test-correctness PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
		# ::gets removed from std++14, but g++4.8 still uses it
		# http://clang-developers.42468.n3.nabble.com/libc-gets-removed-from-C11-td4032945.html
    # target_compile_options( rocfft-test PRIVATE -std=c++14 )
		target_compile_options( rocfft-test-correctness PRIVATE -std=c++11 -pthread )
  endif( )
endif( )

set(TEST_MISC_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/misc/include)
target_compile_definitions( rocfft-test-correctness PRIVATE GTEST_USE_OWN_TR1_TUPLE=1 )
target_include_directories( rocfft-test-correctness
  PRIVATE
#		$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${FFTW_INCLUDES}>
    $<BUILD_INTERFACE:${hip_include_dir}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${TEST_MISC_INCLUDE_DIRS}>
)

target_link_libraries( rocfft-test-correctness rocfft ${GTEST_LIBRARIES} ${FFTW_LIBRARIES} ) # ${Boost_LIBRARIES}

# Ubuntu systems need to explicitely link to pthreads lib because of --as-needed
# https://github.com/google/googletest/issues/391#issuecomment-125645879
# No easy way to filter to Ubuntu
if( UNIX )
	target_link_libraries( rocfft-test-correctness pthread )
endif( )

set_target_properties( rocfft-test-correctness PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocfft-test-correctness PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
