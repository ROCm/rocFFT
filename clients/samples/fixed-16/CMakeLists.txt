# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( fixed-16 fixed-16.cpp )

target_include_directories( fixed-16
  PRIVATE
		$<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
)
target_link_libraries( fixed-16 rocfft ${FFTW_LIBRARIES} )
