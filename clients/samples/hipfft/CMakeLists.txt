################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

find_package(hip)
find_package(rocfft)

set( sample_list hipfft_planmany_2d_z2z hipfft_planmany_2d_r2c hipfft_setworkarea)

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cpp )

  target_include_directories( ${sample}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )

  target_link_libraries( ${sample} PRIVATE roc::rocfft)

  target_compile_features( ${sample} PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

  set_target_properties( ${sample} PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
  set_target_properties( ${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

  if( CUDA_FOUND )
    target_include_directories( ${sample}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
      )
    target_compile_definitions( ${sample} PRIVATE __HIP_PLATFORM_NVCC__ )
    target_link_libraries( ${sample} PRIVATE ${CUDA_LIBRARIES} )
  else( )
    target_link_libraries( ${sample} PRIVATE hip::hip_hcc )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
    # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
    target_compile_options( ${sample} PRIVATE -Wno-unused-command-line-argument )

    # foreach( target ${AMDGPU_TARGETS} )
    #   target_link_libraries( ${sample} PRIVATE --amdgpu-target=${target} )
    # endforeach( )
  endif( )

endforeach( )
