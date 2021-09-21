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

set( package_targets rocfft )

# Package that helps me set visibility for function names exported
# from shared library
include( GenerateExportHeader )

if( NOT DEFINED PYTHON3_EXE )
  set(PYTHON3_EXE python3)
endif()

add_subdirectory( device )

# embed the generator itself into c++ files
set( kgen_embed_command ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed-cpp.py )
set( kgen_embed_h ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed.h )
set( kgen_embed_files
     ${kgendeps}
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/common.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/callback.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/butterfly_constant.h
     ${CMAKE_SOURCE_DIR}/library/src/include/real2complex.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rocfft_butterfly_template.h
   )
add_custom_command(
  OUTPUT ${kgen_embed_h}
  COMMAND ${PYTHON3_EXE} ${kgen_embed_command}
  ${kgen_embed_files} ${kgen_embed_h}
  DEPENDS ${kgen_embed_files}
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
  rocfft_ostream.cpp
  tree_node.cpp
  tree_node_1D.cpp
  tree_node_2D.cpp
  tree_node_3D.cpp
  tree_node_bluestein.cpp
  tree_node_real.cpp
  fuse_shim.cpp
  assignment_policy.cpp
  node_factory.cpp
  hipfft.cpp
  rtc.cpp
  )

set_property(
  SOURCE rtc.cpp
  PROPERTY OBJECT_DEPENDS ${kgen_embed_h}
)

prepend_path( ".." rocfft_headers_public relative_rocfft_headers_public )

add_library( rocfft ${rocfft_source} ${relative_rocfft_headers_public} )
add_library( roc::rocfft ALIAS rocfft )

# add warning flags on everything but hipfft.cpp, which has a known
# deprecation warning in it
set( rocfft_warn_source ${rocfft_source})
list( REMOVE_ITEM rocfft_warn_source hipfft.cpp )
set_property(
  SOURCE ${rocfft_warn_source}
  PROPERTY COMPILE_OPTIONS ${WARNING_FLAGS}
  )
# when hipfft.cpp is removed, flip the whole target to use the same
# flags
#target_compile_options( rocfft PRIVATE ${WARNING_FLAGS} )

if( NOT BUILD_SHARED_LIBS )
  target_link_libraries( rocfft INTERFACE hip::host )
endif()

if( NOT USE_CUDA )
  target_link_libraries( rocfft PRIVATE hip::device )
endif()

target_link_libraries( rocfft PRIVATE rocfft-device-0 )
target_link_libraries( rocfft PRIVATE rocfft-device-1 )
target_link_libraries( rocfft PRIVATE rocfft-device-2 )
target_link_libraries( rocfft PRIVATE rocfft-device-3 )

if( ROCFFT_RUNTIME_COMPILE )
  target_link_libraries( rocfft PRIVATE ${Python3_LIBRARIES} )
  target_include_directories( rocfft PRIVATE ${Python3_INCLUDE_DIRS} )
  target_compile_options( rocfft PRIVATE -DROCFFT_RUNTIME_COMPILE )
endif()

target_include_directories( rocfft
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/src/device>
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

rocm_set_soversion( rocfft ${rocfft_SOVERSION} )
set_target_properties( rocfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocfft PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

#TODO:
# hipcc(with nvcc backend) build has problem for share library visibility,
# need to figure out the reason and enable visibility "hidden" for nvcc eventually.
if(NOT HIP_PLATFORM STREQUAL "nvcc")
  set_target_properties( rocfft PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
endif()

generate_export_header( rocfft EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocfft-export.h )

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
  PREFIX rocfft
  )
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS roc::rocfft
  PREFIX rocfft
  DEPENDS PACKAGE hip
  NAMESPACE roc::
  )

rocm_install_symlink_subdir( rocfft )

# During transition to standalone hipFFT repository, don't install
# symlinks to hipFFT headers.  Once the hipFFT headers are removed
# from rocFFT, we can delete this..
if(NOT CMAKE_HOST_WIN32)
  rocm_install(CODE "
      file(REMOVE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/include/hipfft.h\")
    ")
endif()
