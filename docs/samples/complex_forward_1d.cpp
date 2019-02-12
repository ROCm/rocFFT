#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime_api.h>
#include "rocfft.h"


int main()
{
    // For size N <= 4096
    const size_t N = 16;

    std::vector<float2> cx(N);

    for (size_t i = 0; i < N; i++)
    {
        cx[i].x = (i%3) - (i%7);
        cx[i].y = 0;
    }

    // rocfft gpu compute
    // ========================================

    rocfft_setup();

    size_t Nbytes = N * sizeof(float2);

    // Create HIP device object.
    float2 *x;
    hipMalloc(&x, Nbytes);

    //  Copy data to device
    hipMemcpy(x, &cx[0], Nbytes, hipMemcpyHostToDevice);

    // Create plan
    rocfft_plan plan = NULL;
    size_t length = N;
    rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, &length, 1, NULL);

    // Execute plan
    rocfft_execute(plan, (void**)&x, NULL, NULL);

    // Destroy plan
    rocfft_plan_destroy(plan);

    // Copy result back to host
    std::vector<float2> y(N);
    hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost);
    hipFree(x);

    rocfft_cleanup();

    for( size_t i = 0;i < N;++i){
        if(std::abs( cx[i].x - y[i].x ) < 1e-5){
            std::cerr << "Error - unexpected matching element: observed " << y[i].x << ", expected " <<  cx[i].x << std::endl;
            return 1;
        }
    }

    std::cout << "complex forward 1d - OK!" << std::endl;

    return 0;
}
