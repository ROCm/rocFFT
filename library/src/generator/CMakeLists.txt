################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################



cmake_minimum_required( VERSION 2.8.12 )


#project( rocfft-kernel-generator)
 
add_executable(rocfft-kernel-generator generator.main.cpp)

target_compile_options(rocfft-kernel-generator PUBLIC -std=c++11)


