################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

#if( FFTW_FIND_VERSION VERSION_LESS "3" )
#    message( FFTW_FIND_VERION is ${FFTW_FIND_VERSION})
#    message( FATAL_ERROR "FindFFTW can not configure versions less than FFTW 3.0.0" )
#endif( )

find_path(FFTW_INCLUDE_DIRS
    NAMES fftw3.h
    HINTS
        ${FFTW_ROOT}/include
        $ENV{FFTW_ROOT}/include
    PATHS
        /usr/include
        /usr/local/include
)
mark_as_advanced( FFTW_INCLUDE_DIRS )

# message( STATUS "FFTW_FIND_COMPONENTS: ${FFTW_FIND_COMPONENTS}" )
# message( STATUS "FFTW_FIND_REQUIRED_FLOAT: ${FFTW_FIND_REQUIRED_FLOAT}" )

# Print out the bit-ness search mode cmake is set too, to aid in debugging
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )
if( LIB64 )
  message( STATUS "FindFFTW is searching for 64-bit libraries" )
else( )
  message( STATUS "FindFFTW is searching for 32-bit libraries" )
endif( )

set( FFTW_LIBRARIES "" )
set( FFTW_FIND_REQUIRED_FLOAT TRUE)
if( FFTW_FIND_REQUIRED_FLOAT OR FFTW_FIND_REQUIRED_SINGLE )
  find_library( FFTW_LIBRARIES_SINGLE
      NAMES fftw3f fftw3f-3
      HINTS
          ${FFTW_ROOT}/lib
          $ENV{FFTW_ROOT}/lib
      PATHS
          /usr/lib
          /usr/local/lib
          /usr/lib/x86_64-linux-gnu
      PATH_SUFFIXES
          x86_64-linux-gnu
      DOC "FFTW dynamic library"
  )
  mark_as_advanced( FFTW_LIBRARIES_SINGLE )
  list( APPEND FFTW_LIBRARIES ${FFTW_LIBRARIES_SINGLE} )
endif( )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( FFTW
    REQUIRED_VARS FFTW_INCLUDE_DIRS FFTW_LIBRARIES )

if( NOT FFTW_FOUND )
    message( STATUS "FindFFTW could not find all of the following fftw libraries" )
    message( STATUS "${FFTW_FIND_COMPONENTS}" )
else( )
    message(STATUS "FindFFTW configured variables:" )
    message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}" )
    message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}" )
endif()
