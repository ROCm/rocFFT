#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(rocfft_UnitTest, sample_code_iterative)
{
    for(size_t j = 0; j < 10; j++)
    {
        // rocFFT gpu compute
        // ========================================

        size_t N      = 16;
        size_t Nbytes = N * sizeof(float2);

        // Create HIP device buffer
        float2* x;
        hipMalloc(&x, Nbytes);

        // Initialize data
        std::vector<float2> cx(N);
        for(size_t i = 0; i < N; i++)
        {
            cx[i].x = 1;
            cx[i].y = -1;
        }

        //  Copy data to device
        hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);

        // Create rocFFT plan
        rocfft_plan plan   = NULL;
        size_t      length = N;
        rocfft_plan_create(&plan,
                           rocfft_placement_inplace,
                           rocfft_transform_type_complex_forward,
                           rocfft_precision_single,
                           1,
                           &length,
                           1,
                           NULL);

        // Execute plan
        rocfft_execute(plan, (void**)&x, NULL, NULL);

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Destroy plan
        rocfft_plan_destroy(plan);

        // Copy result back to host
        std::vector<float2> y(N);
        hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

        // Check results
        EXPECT_EQ((float)N, y[0].x);
        EXPECT_EQ(-(float)N, y[0].y);

        for(size_t i = 1; i < N; i++)
        {
            EXPECT_EQ(0, y[i].x);
            EXPECT_EQ(0, y[i].y);
        }

        // Free device buffer
        hipFree(x);
    }
}
