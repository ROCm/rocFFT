# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

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
