# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
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
include( GenerateExportHeader )
add_compiler_export_flags( )

if(PLATFORM_NAME STREQUAL "AMD")
    set( rocfft_source
         rocfft.cpp
         rocfft_transpose.cpp
    )
elseif(PLATFORM_NAME STREQUAL "NVIDIA")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/rocfft.cpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/src/rocfft.cpp ${CMAKE_CURRENT_BINARY_DIR}/src/rocfft.cu)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/rocfft_transpose.cpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/src/rocfft_transpose.cpp ${CMAKE_CURRENT_BINARY_DIR}/src/rocfft_transpose.cu)
    set( rocfft_source
         ${CMAKE_CURRENT_BINARY_DIR}/src/rocfft.cu
         ${CMAKE_CURRENT_BINARY_DIR}/src/rocfft_transpose.cu
       )
endif( )
prepend_path( ".." rocfft_headers_public relative_rocfft_headers_public )

set(internal_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/internal_include)
set(internal_include_src ${internal_include_dirs}/rocfft_transpose_kernel.h)
message("hiproot = ${HIP_ROOT}")
message("BUILD_LIBRARY_TYPE = ${BUILD_LIBRARY_TYPE}")
message("relative_rocfft_headers_public = ${relative_rocfft_headers_public}")
if(PLATFORM_NAME STREQUAL "AMD")
    add_library( rocfft ${BUILD_LIBRARY_TYPE}
      ${rocfft_source}
      ${relative_rocfft_headers_public}
    )
elseif(PLATFORM_NAME STREQUAL "NVIDIA")
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(hip_include_dir ${HIP_ROOT}/include)
    set(extra_cuda_include_dir ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
    CUDA_INCLUDE_DIRECTORIES(${hip_include_dir}
                             ${extra_cuda_include_dir}                            
                             ${CMAKE_SOURCE_DIR}/include
                             ${internal_include_dirs}
                            )
    CUDA_ADD_LIBRARY( rocfft ${BUILD_LIBRARY_TYPE}
      ${rocfft_source}
      ${relative_rocfft_headers_public}
      ${internal_include_src}
    )
endif( )
MESSAGE("CUDA_root = ${CUDA_TOOLKIT_ROOT_DIR}")

# Try to test for specific compiler features if cmake version is recent enough
if( CMAKE_VERSION VERSION_GREATER "3.0" )
  target_compile_features( rocfft PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
else( )
  # Otherwise, just try to compile the library with a standards flag
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    target_compile_options( rocfft PRIVATE -std=c++11 )
  endif( )
endif( )

target_include_directories( rocfft
  PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<INSTALL_INTERFACE:include>
)

set_target_properties( rocfft PROPERTIES VERSION ${rocfft_VERSION} SOVERSION ${rocfft_SOVERSION} )
set_target_properties( rocfft PROPERTIES DEBUG_POSTFIX "-d" )
set_target_properties( rocfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( ${BUILD_LIBRARY_TYPE} STREQUAL "STATIC" )
  set_target_properties( rocfft PROPERTIES PREFIX "lib" )
endif( )

# Package that helps me set visibility for function names exported from shared library
GENERATE_EXPORT_HEADER( rocfft EXPORT_FILE_NAME rocfft-export.h )

# The following is cmake code to generate a config file package for rocfft
# Documentation for how to use the following package:
# https://cmake.org/cmake/help/v3.0/module/CMakePackageConfigHelpers.html
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
set( config_package_location ${LIB_INSTALL_DIR}/cmake/rocfft )

configure_package_config_file(
  rocfft-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/rocfft-config.cmake
  INSTALL_DESTINATION ${config_package_location}
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rocfft-config-version.cmake
  VERSION ${rocfft_VERSION}
  COMPATIBILITY SameMajorVersion
)

install( TARGETS rocfft
  EXPORT rocfft-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install( EXPORT rocfft-targets
  DESTINATION
    ${config_package_location}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/rocfft-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/rocfft-config-version.cmake
  DESTINATION
    ${config_package_location} )

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/rocfft-export.h
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}
)
