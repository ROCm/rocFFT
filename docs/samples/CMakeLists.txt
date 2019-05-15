CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# project name
PROJECT(rocfft_samples CXX)

find_package(hip)
find_package(rocfft)

set (samples complex_1d real2complex_1d)

foreach( exe ${samples} )
  add_executable(${exe} "${exe}.cpp")

  target_link_libraries( ${exe} PRIVATE roc::rocfft hip::hip_hcc )
  target_include_directories( ${exe} PRIVATE ${rocfft_INCLUDE_DIR} )
  set_target_properties( ${exe} PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties( ${exe} PROPERTIES CXX_STANDARD 14)
  
  # prevent issue where __float128 is not supported
  set_target_properties( ${exe} PROPERTIES CXX_EXTENSIONS OFF)
endforeach()

# Build release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
