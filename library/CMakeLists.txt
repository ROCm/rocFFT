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


# target_link_libraries() override
function( target_link_libraries target_name )
  # Remove explicit mentions of hip library so cmake doesn't complain
  # on nvcc path.
  if( USE_CUDA )
    foreach( link_library ${ARGN} )
      if( (link_library MATCHES "^hip::|^hcc::") )
      else( )
        if( TARGET ${link_library} )
          list( APPEND new_list -Xlinker ${link_library} )
        else( )
          list( APPEND new_list ${link_library} )
        endif( )
      endif( )
    endforeach( )
    _target_link_libraries( ${target_name} ${new_list} )
  else( )
    _target_link_libraries( ${target_name} ${ARGN} )
  endif( )
endfunction( )



# This is incremented when the ABI to the library changes
set( rocfft_SOVERSION 0.1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but
# no reason it shouldn't be on all the time.
# This tells cmake to create a compile_commands.json file that can be
# used with clang tooling or vim.
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
if (BUILD_CPUREF)
  set(CMAKE_CXX_FLAGS_DEBUG "-DREF_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
  
# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "rocfft_VERSION: ${rocfft_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_CXX_COMPILER: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS_RELEASE: " ${CMAKE_EXE_LINKER_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: " ${CMAKE_SHARED_LINKER_FLAGS_RELEASE} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_C_FLAGS: ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_CXX_FLAGS: ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_DEBUG: ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
endif( )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/rocfft-version.h.in" "${PROJECT_BINARY_DIR}/include/rocfft-version.h" )

set( rocfft_headers_public
  include/rocfft.h
  ${PROJECT_BINARY_DIR}/include/rocfft-version.h
)

source_group( "Header Files\\Public" FILES ${rocfft_headers_public} )

include( GNUInstallDirs )

set( BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )

if( ROCFFT_RUNTIME_COMPILE )
  # look for python development files - we also require an interpreter
  # but the version of it needn't match the development version.
  find_package (Python3 3.6 COMPONENTS Development REQUIRED)
endif()

# Build into subdirectories
add_subdirectory( src )

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
  set( CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)" )
  set( CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0" )
elseif( HIP_RUNTIME STREQUAL "cuda" )
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-nvcc (>= 3.5.0)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip-nvcc >= 3.5.0")
endif( )
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md" )
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
