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
    const size_t N = 8;

    std::cout << "Complex 1d in-place FFT example\n";

    // Initialize data on the host
    std::vector<float> cx(N);
    for(size_t i = 0; i < N; i++)
    {
        cx[i] = i;
    }

    std::cout << "Input:\n";
    for(size_t i = 0; i < N; i++)
    {
        std::cout << cx[i] << "  ";
    }
    std::cout << "\n";

    rocfft_setup();

    // Create HIP device objects:
    float* x;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    float2* y;
    hipMalloc(&y, (N / 2 + 1) * sizeof(float));

    //  Copy data to device
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    // Create plans
    rocfft_plan forward = NULL;
    rocfft_plan_create(&forward,
                       rocfft_placement_notinplace,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
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
    std::vector<float2> cy(N / 2 + 1);
    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < cy.size(); i++)
    {
        std::cout << "( " << cy[i].x << "," << cy[i].y << ") ";
    }
    std::cout << "\n";

    // Create plans
    rocfft_plan backward = NULL;
    rocfft_plan_create(&backward,
                       rocfft_placement_notinplace,
                       rocfft_transform_type_real_inverse,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
                       1, // Number of transforms
                       NULL); // Description

    // Execute the backward transform
    rocfft_execute(backward, // plan
                   (void**)&y, // in_buffer
                   (void**)&x, // out_buffer
                   fftinfo); // execution info

    std::cout << "Transformed back:\n";
    std::vector<float> backx(N);
    hipMemcpy(
        backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < backx.size(); i++)
    {
        std::cout << backx[i] << "  ";
    }
    std::cout << "\n";

    const float overN = 1.0f / N;
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
