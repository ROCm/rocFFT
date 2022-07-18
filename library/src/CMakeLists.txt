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

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

add_executable( rocfft_rtc_helper rocfft_rtc_helper.cpp rtc_compile.cpp )

if( WIN32 )
  set( ROCFFT_HIPRTC_LIB "${HIP_PATH}/lib/hiprtc.lib" )
endif()

# each backend requires different libraries for host and device code
if( USE_CUDA )
  set( ROCFFT_HOST_LINK_LIBS -lnvrtc -lnvrtc-builtins -lnvptxcompiler_static -lcuda )
else()
  set( ROCFFT_HOST_LINK_LIBS hip::host )
  set( ROCFFT_DEVICE_LINK_LIBS hip::device )
endif()

set( package_targets rocfft rocfft_rtc_helper )
target_include_directories( rocfft_rtc_helper
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
)
target_link_libraries( rocfft_rtc_helper PRIVATE ${ROCFFT_HOST_LINK_LIBS} ${ROCFFT_HIPRTC_LIB} )
set_target_properties( rocfft_rtc_helper PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON )

# Package that helps me set visibility for function names exported
# from shared library
include( GenerateExportHeader )

if( NOT DEFINED PYTHON3_EXE )
  set(PYTHON3_EXE python3)
endif()

add_subdirectory( device )

#
# embed the generator itself into c++ files
#

# script that produces an include file
set( kgen_embed_command ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed-cpp.py )

# location of the include file
set( kgen_embed_h ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed.h )

# files that need to be embedded into the library, to be able to generate code
set( kgen_embed_files
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/common.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/callback.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/butterfly_constant.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/real2complex_device.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rocfft_butterfly_template.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_workarounds.h
   )

# files that contribute to the logic of how code gets generated -
# embedded files obviously already contribute.  these are checksummed
# to serve as a "version" for the code generator.
set( kgen_logic_files

     # python code that does the embedding
     ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed-cpp.py

     # python code that decides kernel parameters
     ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py
     ${CMAKE_SOURCE_DIR}/library/src/device/generator.py

     # stockham generator code
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/generator.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_aot.cpp
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen.cpp
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_2d.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_base.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_cc.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_cr.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_rc.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_rr.h

     # c++ code that does rtc-specific adjustments
     ${CMAKE_SOURCE_DIR}/library/src/rtc.cpp
   )

add_custom_command(
  OUTPUT ${kgen_embed_h}
  COMMAND ${PYTHON3_EXE} ${kgen_embed_command}
  --embed ${kgen_embed_files} --logic ${kgen_logic_files} --output ${kgen_embed_h}
  DEPENDS ${kgen_embed_command} ${kgen_embed_files} ${kgen_logic_files}
)

# The following is a list of implementation files defining the library
set( rocfft_source
  auxiliary.cpp
  plan.cpp
  transform.cpp
  repo.cpp
  powX.cpp
  twiddles.cpp
  kargs.cpp
  tree_node.cpp
  tree_node_1D.cpp
  tree_node_2D.cpp
  tree_node_3D.cpp
  tree_node_bluestein.cpp
  tree_node_real.cpp
  fuse_shim.cpp
  assignment_policy.cpp
  node_factory.cpp
  rtc_exports.cpp
  )

# SQLite 3.36.0 enabled the backup API by default, which we need
# for cache serialization.  We also want to use a static SQLite,
# and distro static libraries aren't typically built
# position-independent.
option( SQLITE_USE_SYSTEM_PACKAGE "Use SQLite3 from find_package" OFF )
if( SQLITE_USE_SYSTEM_PACKAGE )
  find_package(SQLite3 3.36 REQUIRED)
  set(ROCFFT_SQLITE_LIB SQLite::SQLite3)
else()
  include( FetchContent )

  if(DEFINED ENV{SQLITE_SRC_URL})
    set(SQLITE_SRC_URL_INIT $ENV{SQLITE_SRC_URL})
  else()
    set(SQLITE_SRC_URL_INIT https://sqlite.org/2021/sqlite-amalgamation-3360000.zip)
  endif()
  set(SQLITE_SRC_URL ${SQLITE_SRC_URL_INIT} CACHE STRING "Location of SQLite source code")
  set(SQLITE_SRC_SHA256 999826fe4c871f18919fdb8ed7ec9dd8217180854dd1fe21eea96aed36186729 CACHE STRING "SHA256 hash of SQLite source code")

  # embed SQLite
  FetchContent_Declare(sqlite_local
    URL ${SQLITE_SRC_URL}
    URL_HASH SHA256=${SQLITE_SRC_SHA256}
  )
  FetchContent_MakeAvailable(sqlite_local)
  add_library( sqlite3 OBJECT ${sqlite_local_SOURCE_DIR}/sqlite3.c )
  set_target_properties( sqlite3 PROPERTIES
    C_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    )

  # we don't need extensions, and omitting them from SQLite removes the
  # need for dlopen/dlclose from within rocFFT
  target_compile_options(
    sqlite3
    PRIVATE -DSQLITE_OMIT_LOAD_EXTENSION
  )
  set(ROCFFT_SQLITE_LIB sqlite3)
endif()

# files that depend on kernel-generator-embed.h
foreach( file rtc.cpp rtc_stockham.cpp )
  set_property(
    SOURCE ${file}
    PROPERTY OBJECT_DEPENDS ${kgen_embed_h}
  )
endforeach()

# RTC stuff that's used by both core library and helpers
add_library( rocfft-rtc-common OBJECT
  compute_scheme.cpp
  rocfft_ostream.cpp
  rtc.cpp
  rtc_cache.cpp
  rtc_compile.cpp
  rtc_subprocess.cpp
  rtc_stockham.cpp
)
target_include_directories( rocfft-rtc-common
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  ${sqlite_local_SOURCE_DIR}
)
set_target_properties( rocfft-rtc-common PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  VISIBILITY_INLINES_HIDDEN ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries( rocfft-rtc-common PRIVATE ${ROCFFT_HOST_LINK_LIBS} ${ROCFFT_HIPRTC_LIB} )

prepend_path( ".." rocfft_headers_public relative_rocfft_headers_public )

add_library( rocfft
  ${rocfft_source}
  ${relative_rocfft_headers_public}
  )
add_library( roc::rocfft ALIAS rocfft )
target_link_libraries( rocfft PRIVATE ${ROCFFT_SQLITE_LIB} )

if( WIN32 )
  target_link_libraries( rocfft PRIVATE ${ROCFFT_HIPRTC_LIB} )
else()
  target_link_libraries( rocfft PRIVATE -lstdc++fs )
endif()
if( ROCFFT_RUNTIME_COMPILE )
  target_compile_options( rocfft PRIVATE -DROCFFT_RUNTIME_COMPILE )
  target_compile_options( rocfft-rtc-common PRIVATE -DROCFFT_RUNTIME_COMPILE )
endif()

target_compile_options( rocfft PRIVATE ${WARNING_FLAGS} )
if( ${CMAKE_BUILD_TYPE} STREQUAL Debug )
  target_compile_options(rocfft PRIVATE -Og)
endif()

if( NOT BUILD_SHARED_LIBS )
  target_link_libraries( rocfft INTERFACE ${ROCFFT_HOST_LINK_LIBS} )
endif()

target_link_libraries( rocfft PRIVATE ${ROCFFT_DEVICE_LINK_LIBS} )

target_link_libraries( rocfft PRIVATE rocfft-rtc-common )
target_link_libraries( rocfft PRIVATE rocfft-device-0 )
target_link_libraries( rocfft PRIVATE rocfft-device-1 )
target_link_libraries( rocfft PRIVATE rocfft-device-2 )
target_link_libraries( rocfft PRIVATE rocfft-device-3 )
# rocfft uses dladdr to find the RTC helper program
if( NOT WIN32 )
  target_link_libraries( rocfft PUBLIC -ldl pthread )
endif()

target_include_directories( rocfft
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/src/device>
  ${sqlite_local_SOURCE_DIR}
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocfft>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

rocm_set_soversion( rocfft ${rocfft_SOVERSION} )
set_target_properties( rocfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocfft PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties( rocfft PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )

generate_export_header( rocfft EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocfft/rocfft-export.h )

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_file(
    rocfft-version.h rocfft-export.h
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR} rocfft/${CMAKE_INSTALL_INCLUDEDIR}
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/rocfft/rocfft-version.h
  )
endif( )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocfft PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
  ${CMAKE_SOURCE_DIR}/library/include
  ${CMAKE_BINARY_DIR}/include
  )

#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS roc::rocfft
  DEPENDS PACKAGE hip
  NAMESPACE roc::
  )

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/rocfft"
        DESTINATION "." )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()
 
