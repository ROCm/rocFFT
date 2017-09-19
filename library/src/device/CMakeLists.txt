################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

cmake_minimum_required( VERSION 2.8.12 )


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
# A helper function to convert an hcc .cpp file into the equivalent cuda .cu file
# ########################################################################
function( cuda_rename_source source_list_of_files return_list_of_files )

  # This loop copies each file in source_list_of_files to CMAKE_CURRENT_BINARY_DIR
  # Then it replaces .cpp extentions with .cu
  foreach( cpp_file ${${source_list_of_files}} )
    string(REGEX REPLACE "\\.cpp$" ".cu" cu_file ${cpp_file} )
    configure_file( "${cpp_file}" "${cu_file}" COPYONLY )
    list( APPEND cuda_file_list ${cu_file} )
  endforeach( )

  # This prepends the CMAKE_CURRENT_BINARY_DIR path to the list of generated files
  prepend_path( "${CMAKE_CURRENT_BINARY_DIR}" cuda_file_list cuda_file_list )
  set( ${return_list_of_files} ${cuda_file_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30 -std=c++11") #at least Kepler and above


project( rocfft-device CXX C )

add_subdirectory( generator )

set( rocfft_device_source
     transpose.cpp
     real2complex.cpp
     function_pool.cpp
     kernel_launch_single.cpp
     kernel_launch_double.cpp
     kernel_launch_pow2_large_single.cpp
     kernel_launch_pow2_large_double.cpp
     transpose.cpp
     kernel_launch_large.cpp
)

if(PLATFORM_NAME STREQUAL "AMD")
  add_library( rocfft-device
     ${rocfft_device_source}
  )
  set_target_properties( rocfft-device PROPERTIES OUTPUT_NAME "rocfft-device-hcc" )
elseif(PLATFORM_NAME STREQUAL "NVIDIA")
  message("CUDA_root = ${CUDA_TOOLKIT_ROOT_DIR}")
  cuda_rename_source( rocfft_device_source rocfft_device_source )
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(hip_include_dir ${HIP_SRC_ROOT}/src/HIP/include)
  set(extra_cuda_include_dir ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
  CUDA_INCLUDE_DIRECTORIES(${hip_include_dir}
                           ${extra_cuda_include_dir}
                           ${PROJECT_SOURCE_DIR}
                           ${PROJECT_SOURCE_DIR}/generator
                           ${PROJECT_BINARY_DIR}/generator
                           ${PROJECT_SOURCE_DIR}/../include
                           ${PROJECT_SOURCE_DIR}/../../include
                          )
  CUDA_ADD_LIBRARY( rocfft-device
    ${rocfft_device_source}
  )
  set_target_properties( rocfft-device PROPERTIES OUTPUT_NAME "rocfft-device-cuda" )
endif( )


add_dependencies( rocfft-device rocfft-kernel-generator )


target_include_directories( rocfft-device
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/generator>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generator>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../include>
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../../include>
)

install( TARGETS rocfft-device
  EXPORT rocfft-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

