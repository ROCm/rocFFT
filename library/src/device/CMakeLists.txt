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
set( rocfft_device_source
  transpose.cpp
  real2complex.cpp
  function_pool.cpp
  kernel_launch_single.cpp
  kernel_launch_double.cpp
  kernel_launch_pow2_large_single.cpp
  kernel_launch_pow2_large_double.cpp
  transpose.cpp
  kernel_launch_large_transpose.cpp
)

prepend_path( "../.." rocfft_headers_public relative_rocfft_device_headers_public )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  add_library( rocfft-device ${rocfft_device_source} ${relative_rocfft_device_headers_public} )
elseif( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )

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

  HIP_ADD_LIBRARY( rocfft-device ${rocfft_device_source} ${relative_rocfft_device_headers_public} )
endif( )

# NOTE: 2 issues below
# 1.  rocfft header 'rocfft-types.h' exposes the hip header hip/hip_vector_types.h, requiring exposing hip::hip_hcc as PUBLIC
# 2.  hcc requires -Wl,-Bsymbolic for rocm v1.6.1, should be able to remove in future
target_link_libraries( rocfft-device PUBLIC hip::hip_hcc PRIVATE hip::hip_device hcc::hccshared )

# Test for specific compiler features if cmake version is recent enough
target_compile_features( rocfft-device PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocfft-device PRIVATE -Wno-unused-command-line-argument )

  foreach( target ${AMDGPU_TARGETS} )
    target_link_libraries( rocfft-device PRIVATE --amdgpu-target=${target} )
  endforeach( )
endif( )

target_include_directories( rocfft-device
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generator>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generator>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
)

set_target_properties( rocfft-device PROPERTIES VERSION ${rocfft_VERSION} SOVERSION ${rocfft_SOVERSION} CXX_EXTENSIONS NO )
set_target_properties( rocfft-device PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( rocfft-device PROPERTIES DEBUG_POSTFIX "-d" )

# Package that helps me set visibility for function names exported from shared library
# set_target_properties( rocfft-device PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
# include( GenerateExportHeader )
# generate_export_header( rocfft-device EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocfft-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocfft-device PROPERTIES PREFIX "lib" )
endif( )

# Building rocfft-device invokes the kernel generator first as part of the build
add_subdirectory( generator )
add_dependencies( rocfft-device rocfft-kernel-generator )

rocm_install_targets(
  TARGETS rocfft-device
  PREFIX rocfft
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
