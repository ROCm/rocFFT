# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

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

function( cuda_find_samples_inc_dir header_dir )
  find_path( CUDA_SAMPLES_INCLUDE_DIR
  NAMES helper_math.h
  HINTS
      ${NVCUDASAMPLES_ROOT}/common/inc
      $ENV{NVCUDASAMPLES_ROOT}/common/inc
  PATHS
      /usr/local/cuda/samples/common/inc
  )
  set( ${header_dir} ${CUDA_SAMPLES_INCLUDE_DIR} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################
# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )

add_subdirectory( device )

# The following is a list of implementation files defining the library
set( rocfft_source
      auxiliary.cpp
      plan.cpp
      transform.cpp
      hipfft.cpp
      repo.cpp
      powX.cpp
      get_radix.cpp
      twiddles.cpp
      kargs.cpp
    )

prepend_path( ".." rocfft_headers_public relative_rocfft_headers_public )

add_library( rocfft ${rocfft_source} ${relative_rocfft_headers_public} )
add_library( roc::rocfft ALIAS rocfft )
target_compile_features( rocfft PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

target_link_libraries( rocfft PRIVATE hip::hip_hcc hip::hip_device hcc::hccshared rocfft-device )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocfft PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( rocfft PRIVATE --amdgpu-target=${target} )
  endforeach( )
else( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  cuda_find_samples_inc_dir( cuda_samples_inc_dir )
endif( )

target_include_directories( rocfft
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/src/device>
          $<BUILD_INTERFACE:${cuda_samples_inc_dir}>
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include/rocfft>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocfft>
          $<INSTALL_INTERFACE:include>
)

set_target_properties( rocfft PROPERTIES VERSION ${rocfft_VERSION} SOVERSION ${rocfft_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( rocfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft PROPERTIES DEBUG_POSTFIX "-d" )

set_target_properties( rocfft PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( rocfft EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocfft-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocfft PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS rocfft
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX rocfft
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS rocfft-targets
  PREFIX rocfft
  DEPENDS PACKAGE hip
  NAMESPACE roc::
 )
