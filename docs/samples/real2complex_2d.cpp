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
    const size_t Nx = 8;
    const size_t Ny = 8;

    const size_t Nycomplex = Ny / 2 + 1;

    std::cout << "Complex 1d in-place FFT example\n";

    // Initialize data on the host
    std::vector<float> cx(Nx * Ny);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            cx[i * Ny + j] = i + j;
        }
    }

    std::cout << "Input:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            std::cout << cx[i * Ny + j] << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Output buffer
    std::vector<float2> cy(Nx * Nycomplex);

    rocfft_setup();

    // Create HIP device objects:
    float* x;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    float2* y;
    hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));

    //  Copy data to device
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    const size_t lengths[2] = {Nx, Ny};

    // Create plans
    rocfft_plan forward = NULL;
    rocfft_plan_create(&forward,
                       rocfft_placement_notinplace,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_single,
                       2, // Dimensions
                       lengths, // lengths
                       1, // Number of transforms
                       NULL); // Description

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info fftinfo;
    rocfft_execution_info_create(&fftinfo);
    size_t wbuffersize = 0;
    rocfft_plan_get_work_buffer_size(forward, &wbuffersize);
    void* wbuffer;
    hipMalloc(&wbuffer, wbuffersize);
    rocfft_execution_info_set_work_buffer(fftinfo, wbuffer, wbuffersize);

    // Execute the forward transform
    rocfft_execute(forward, // plan
                   (void**)&x, // in_buffer
                   (void**)&y, // out_buffer
                   fftinfo); // execution info

    std::cout << "Transformed:\n";

    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; ++j)
        {
            const size_t pos = i * Nycomplex + j;
            std::cout << "( " << cy[pos].x << "," << cy[pos].y << ") ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Create plans
    rocfft_plan backward = NULL;
    rocfft_plan_create(&backward,
                       rocfft_placement_notinplace,
                       rocfft_transform_type_real_inverse,
                       rocfft_precision_single,
                       2, // Dimensions
                       lengths, // lengths
                       1, // Number of transforms
                       NULL); // Description

    // Execute the backward transform
    rocfft_execute(backward, // plan
                   (void**)&y, // in_buffer
                   (void**)&x, // out_buffer
                   fftinfo); // execution info

    std::cout << "Transformed back:\n";
    std::vector<float> backx(cx.size());
    hipMemcpy(
        backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; ++j)
        {
            const size_t pos = i * Ny + j;
            std::cout << backx[pos] << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    const float overN = 1.0f / cx.size();
    float       error = 0.0f;
    for(size_t i = 0; i < cx.size(); i++)
    {
        //std::cout << "i: " << i << "\t" << cx[i] << "\t" << backx[i] << "\n";
        float diff = std::abs(backx[i] * overN - cx[i]);
        if(diff > error)
        {
            error = diff;
        }
    }
    std::cout << "Maximum error: " << error << "\n";

    hipFree(x);
    hipFree(y);
    hipFree(wbuffer);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();

    return 0;
}
