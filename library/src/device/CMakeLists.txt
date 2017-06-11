
cmake_minimum_required( VERSION 2.8.12 )

project( rocfft-device CXX C )

add_subdirectory( generator )

set( rocfft_device_source
     function_pool.cpp
     kernel_launch_single.cpp
     kernel_launch_double.cpp
     kernel_launch_pow2_large_single.cpp
     kernel_launch_pow2_large_double.cpp
)

add_library( rocfft-device
     ${rocfft_device_source}
)

add_dependencies( rocfft-device rocfft-kernel-generator )

set_target_properties( rocfft-device PROPERTIES OUTPUT_NAME "rocfft-device-hcc" )

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

