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

# ########################################################################
# Main
# ########################################################################

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

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  add_library( rocfft ${rocfft_source} ${relative_rocfft_headers_public} )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
# For CUDA platforms, use HIP to compile source
  list( APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake )

  list( APPEND HIP_NVCC_FLAGS "-arch=sm_30 -std=c++11" ) #at least Kepler and above
  find_package( HIP REQUIRED )

  # message("CUDA_root = ${CUDA_TOOLKIT_ROOT_DIR}")
  # cuda_rename_source( rocfft_source rocfft_source )
  # set( CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} )
  # set(hip_include_dir ${HIP_SRC_ROOT}/src/HIP/include)
  # et(extra_cuda_include_dir ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
  # HIP_INCLUDE_DIRECTORIES( ${hip_include_dir}
  #     ${extra_cuda_include_dir}
  #     ${CMAKE_SOURCE_DIR}/src/include
  #     ${CMAKE_SOURCE_DIR}/src/device/generator
  #     ${CMAKE_SOURCE_DIR}/include
  #     ${PROJECT_BINARY_DIR}/src/device/generator
  #     ${PROJECT_BINARY_DIR}/include
  #     ${pwd_include_dir}
  #     )

  HIP_ADD_LIBRARY( rocfft ${rocfft_source} ${relative_rocfft_headers_public} )
endif( )

add_subdirectory( device )

target_link_libraries( rocfft PUBLIC hip::hip_hcc PRIVATE hip::hip_device hcc::hccshared rocfft-device )

# Test for specific compiler features if cmake version is recent enough
target_compile_features( rocfft PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocfft PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( rocfft PRIVATE --amdgpu-target=${target} )
  endforeach( )
endif( )

target_include_directories( rocfft
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/src/device>
          $<INSTALL_INTERFACE:include>
)

set_target_properties( rocfft PROPERTIES VERSION ${rocfft_VERSION} SOVERSION ${rocfft_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( rocfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft PROPERTIES DEBUG_POSTFIX "-d" )

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
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

rocm_install_symlink_subdir( rocfft )