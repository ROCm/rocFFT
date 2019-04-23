################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

# ########################################################################
# The following check are hacks to get compiles working for CUDA backend
# These should be removed in the future.

# NVCC can not compile fftw.3 header file appropriately before v3.3.5
# https://github.com/FFTW/fftw3/issues/18
# v3.3.5 is installed by default on Ubuntu 16, workaround is to trick fftw3
# into thinking nvcc is an Intel compiler as desribed in above issue
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  add_definitions( -D__INTEL_COMPILER )
endif( )
# ########################################################################

set( sample_list hipfft_planmany_2d_r2c hipfft_planmany_2d_z2z)

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cpp )

  target_include_directories( ${sample}
    PRIVATE $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )

  target_link_libraries( ${sample} PRIVATE roc::rocfft ${FFTW_LIBRARIES} )

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
