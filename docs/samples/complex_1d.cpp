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
    std::vector<float2> cx(N);
    for(size_t i = 0; i < N; i++)
    {
        cx[i].x = i;
        cx[i].y = 0;
    }

    std::cout << "Input:\n";
    for(size_t i = 0; i < N; i++)
    {
        std::cout << "( " << cx[i].x << "," << cx[i].y << ") ";
    }
    std::cout << "\n";

    rocfft_setup();

    // Create HIP device object.
    float2* x;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));

    //  Copy data to device
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    // Create plans
    rocfft_plan forward = NULL;
    rocfft_plan_create(&forward,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
                       1, // Number of transforms
                       NULL); // Description

    // Create plans
    rocfft_plan backward = NULL;
    rocfft_plan_create(&backward,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_inverse,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
                       1, // Number of transforms
                       NULL); // Description

    // Execute the forward transform
    rocfft_execute(forward,
                   (void**)&x, // in_buffer
                   NULL, // out_buffer
                   NULL); // execution info

    // Copy result back to host
    std::vector<float2> cy(N);
    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);

    std::cout << "Transformed:\n";
    for(size_t i = 0; i < cy.size(); i++)
    {
        std::cout << "( " << cy[i].x << "," << cy[i].y << ") ";
    }
    std::cout << "\n";

    // Execute the backward transform
    rocfft_execute(backward,
                   (void**)&x, // in_buffer
                   NULL, // out_buffer
                   NULL); // execution info

    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    std::cout << "Transformed back:\n";
    for(size_t i = 0; i < cy.size(); i++)
    {
        std::cout << "( " << cy[i].x << "," << cy[i].y << ") ";
    }
    std::cout << "\n";

    const float overN = 1.0f / N;
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
