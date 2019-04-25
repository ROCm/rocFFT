CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# project name
PROJECT(rocfft_samples CXX)

find_package(hip)
find_package(rocfft)

add_executable(complex_1d complex_1d.cpp)
target_link_libraries( complex_1d PRIVATE roc::rocfft hip::hip_hcc )
target_include_directories( complex_1d PRIVATE ${rocfft_INCLUDE_DIR} )
set_target_properties( complex_1d PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties( complex_1d PROPERTIES CXX_STANDARD 14)

# prevent issue where __float128 is not supported
set_target_properties( complex_1d PROPERTIES CXX_EXTENSIONS OFF)

# Build release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
