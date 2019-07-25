/******************************************************************************
* Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "rocfft.h"

int main()
{
    // The problem size
    const size_t Nx = 4;
    const size_t Ny = 4;
    const size_t Nz = 2;

    std::cout << "Complex 3d in-place FFT example\n";

    // Initialize data on the host
    std::vector<float2> cx(Nx * Ny * Nz);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                cx[pos].x        = i + j + k;
                cx[pos].y        = 0;
            }
        }
    }

    std::cout << "Input:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                std::cout << "( " << cx[pos].x << "," << cx[pos].y << ") ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    rocfft_setup();

    // Create HIP device object.
    float2* x;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));

    //  Copy data to device
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    const size_t lengths[3] = {Nx, Ny, Nz};

    rocfft_status status;

    // Create plans
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                rocfft_placement_inplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_single,
                                3, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    rocfft_execution_info forward_info;
    status = rocfft_execution_info_create(&forward_info);
    assert(status == rocfft_status_success);
    size_t fbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
    assert(status == rocfft_status_success);
    void* fbuffer;
    hipMalloc(&fbuffer, fbuffersize);
    status = rocfft_execution_info_set_work_buffer(forward_info, fbuffer, fbuffersize);
    assert(status == rocfft_status_success);

    // Create plans
    rocfft_plan backward = NULL;
    status               = rocfft_plan_create(&backward,
                                rocfft_placement_inplace,
                                rocfft_transform_type_complex_inverse,
                                rocfft_precision_single,
                                3, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    rocfft_execution_info backward_info;
    status = rocfft_execution_info_create(&backward_info);
    assert(status == rocfft_status_success);
    size_t bbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);
    assert(status == rocfft_status_success);
    void* bbuffer;
    hipMalloc(&bbuffer, bbuffersize);
    status = rocfft_execution_info_set_work_buffer(backward_info, bbuffer, bbuffersize);
    assert(status == rocfft_status_success);

    // Execute the forward transform
    status = rocfft_execute(forward,
                            (void**)&x, // in_buffer
                            NULL, // out_buffer
                            forward_info); // execution info
    assert(status == rocfft_status_success);

    // Copy result back to host
    std::vector<float2> cy(cx.size());
    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);

    std::cout << "Transformed:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {

            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                std::cout << "( " << cy[pos].x << "," << cy[pos].y << ") ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Execute the backward transform
    status = rocfft_execute(backward,
                            (void**)&x, // in_buffer
                            NULL, // out_buffer
                            backward_info); // execution info
    assert(status == rocfft_status_success);

    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    std::cout << "Transformed back:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                std::cout << "( " << cy[pos].x << "," << cy[pos].y << ") ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    const float overN = 1.0f / cx.size();
    float       error = 0.0f;
    for(size_t i = 0; i < cx.size(); i++)
    {
        float diff
            = std::max(std::abs(cx[i].x - cy[i].x * overN), std::abs(cx[i].y - cy[i].y * overN));
        if(diff > error)
        {
            error = diff;
        }
    }
    std::cout << "Maximum error: " << error << "\n";

    hipFree(x);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();

    return 0;
}
