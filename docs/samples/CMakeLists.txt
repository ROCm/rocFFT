CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# project name
PROJECT(rocfft_samples CXX)

find_package(hip)
find_package(rocfft)

add_executable(complex_forward_1d complex_forward_1d.cpp)
target_compile_features( complex_forward_1d PRIVATE cxx_static_assert cxx_nullptr cxx_lambdas cxx_auto_type )
target_link_libraries( complex_forward_1d PRIVATE roc::rocfft hip::hip_hcc )
target_include_directories( complex_forward_1d PRIVATE ${rocfft_INCLUDE_DIR} )
message("!! ${rocfft_INCLUDE_DIR}")

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( complex_forward_1d PRIVATE -Wno-unused-command-line-argument )

  # foreach( target ${AMDGPU_TARGETS} )
  #   target_link_libraries( rocfft-selftest PRIVATE --amdgpu-target=${target} )
  # endforeach( )
endif( )
