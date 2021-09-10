# #############################################################################
# Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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
# #############################################################################

set( sample_list fixed-large-float fixed-large-double )

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cpp )

  target_include_directories( ${sample}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )

  target_link_libraries( ${sample} PRIVATE roc::rocfft )

  target_compile_options( ${sample} PRIVATE ${WARNING_FLAGS} )

  set_target_properties( ${sample} PROPERTIES
    DEBUG_POSTFIX "-d"
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
  )
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
    target_link_libraries( ${sample} PRIVATE hip::device )
  endif( )

endforeach( )
