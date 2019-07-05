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

function( get_subdir_list current_dir subdir_list )
  file( GLOB files LIST_DIRECTORIES true RELATIVE ${current_dir} * )
  foreach( file ${files} )
    if( IS_DIRECTORY ${current_dir}/${file} )
      list( APPEND temp_subdir_list ${file} )
    endif( )
  endforeach( )
  set( ${subdir_list} ${temp_subdir_list} PARENT_SCOPE )
endfunction( )

find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT DOUBLE )

get_subdir_list( ${CMAKE_CURRENT_SOURCE_DIR} samples_subdirs )

foreach( client ${samples_subdirs} )
  add_subdirectory( ${client} )
endforeach( )
