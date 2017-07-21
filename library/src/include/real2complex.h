/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#ifndef REAL_TO_COMPLEX_H
#define REAL_TO_COMPLEX_H

#include <unordered_map>
#include "tree_node.h"


void real2complex(size_t input_size, 
                  void* input_buffer, void* output_buffer, 
                  rocfft_precision precision);

void complex2hermitian(size_t length, 
                       void* input_buffer, size_t input_distance, 
                       void* output_buffer, size_t output_distance, 
                       size_t batch, rocfft_precision precision);

#endif // REAL_TO_COMPLEX_H

