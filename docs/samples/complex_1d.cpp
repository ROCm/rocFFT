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
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "rocfft.h"

int main()
{
    std::cout << "Complex 1d in-place FFT example\n";

    // The problem size
    const size_t N = 11;

    // Initialize data on the host
    std::cout << "Input:\n";
    std::vector<std::complex<float>> cx(N);
    for(size_t i = 0; i < N; i++)
    {
        cx[i] = std::complex<float>(i, 0);
    }
    for(size_t i = 0; i < N; i++)
    {
        std::cout << cx[i] << " ";
    }
    std::cout << std::endl;

    rocfft_setup();

    // Create HIP device object and copy data:
    float2* x;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    // Create forward plan
    rocfft_plan forward = NULL;
    rocfft_plan_create(&forward,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
                       1, // Number of transforms
                       NULL); // Description

    // We may need work memory, which is passed via rocfft_execution_info
    rocfft_execution_info forwardinfo;
    rocfft_execution_info_create(&forwardinfo);
    size_t fbuffersize = 0;
    rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
    void* fbuffer;
    hipMalloc(&fbuffer, fbuffersize);
    rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);

    // Create backward plan
    rocfft_plan backward = NULL;
    rocfft_plan_create(&backward,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_inverse,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
                       1, // Number of transforms
                       NULL); // Description

    rocfft_execution_info backwardinfo;
    rocfft_execution_info_create(&backwardinfo);
    size_t bbuffersize = 0;
    rocfft_plan_get_work_buffer_size(backward, &bbuffersize);
    void* bbuffer;
    hipMalloc(&bbuffer, bbuffersize);
    rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);

    // Execute the forward transform
    rocfft_execute(forward,
                   (void**)&x, // in_buffer
                   NULL, // out_buffer
                   forwardinfo); // execution info

    // Copy result back to host
    std::vector<std::complex<float>> cy(N);
    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);

    std::cout << "Transformed:\n";
    for(size_t i = 0; i < cy.size(); i++)
    {
        std::cout << cy[i] << " ";
    }
    std::cout << std::endl;

    // Execute the backward transform
    rocfft_execute(backward,
                   (void**)&x, // in_buffer
                   NULL, // out_buffer
                   backwardinfo); // execution info

    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    std::cout << "Transformed back:\n";
    for(size_t i = 0; i < cy.size(); i++)
    {
        std::cout << cy[i] << " ";
    }
    std::cout << std::endl;

    const float overN = 1.0f / N;
    float       error = 0.0f;
    for(size_t i = 0; i < cx.size(); i++)
    {
        float diff = std::max(std::abs(cx[i].real() - cy[i].real() * overN),
                              std::abs(cx[i].imag() - cy[i].imag() * overN));
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
