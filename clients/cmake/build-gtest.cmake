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

