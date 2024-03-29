# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

include( ExternalProject )

option( BUILD_GTEST "Download and build GoogleTest" OFF )

if( NOT BUILD_GTEST )
    find_package( GTest 1.11.0 )
endif()

if( (BUILD_GTEST OR NOT GTEST_FOUND) AND (NOT TARGET gtest) )
  set(GTEST_INCLUDE_DIRS
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest/googletest/include)
  set(GTEST_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})
  
  set(GTEST_SRC_URL https://github.com/google/googletest/archive/release-1.11.0.tar.gz CACHE STRING "Location of GTest source code")
  set(GTEST_SRC_SHA256 b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5 CACHE STRING "SHA256 hash of GTest source code")
  
  ExternalProject_Add(gtest
                      URL ${GTEST_SRC_URL}
                      URL_HASH SHA256=${GTEST_SRC_SHA256}
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
                      CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                      INSTALL_COMMAND ""
                      BUILD_BYPRODUCTS ${GTEST_LIBRARIES})
  ExternalProject_Get_Property( gtest source_dir binary_dir )  
endif()

