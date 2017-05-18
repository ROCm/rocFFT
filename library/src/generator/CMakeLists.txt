################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################



cmake_minimum_required( VERSION 2.8.12 )


project( rocfft_kernel_generator)
 
add_executable(kernel_generator generator.main.cpp)

target_compile_options(kernel_generator PUBLIC -std=c++11)


