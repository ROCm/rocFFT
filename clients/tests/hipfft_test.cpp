/******************************************************************************
* Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipfft.h"
#include "rocfft_against_fftw.h"
#include "test_constants.h"
#include <fftw3.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <unistd.h>
#include <vector>

TEST(hipfftTest, Create1dPlan)
{
    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t length = 1024;
    EXPECT_TRUE(hipfftPlan1d(&plan, length, HIPFFT_C2C, 1) == HIPFFT_SUCCESS);

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeC2C)
{
    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 1024;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_C2C, 1, &workSize) == HIPFFT_SUCCESS);
    // No extra work buffer for C2C
    EXPECT_TRUE(0 == workSize);

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeR2C)
{
    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_R2C, 1, &workSize) == HIPFFT_SUCCESS);
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(float));
    }

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeC2R)
{
    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_C2R, 1, &workSize) == HIPFFT_SUCCESS);
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(float));
    }

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeD2Z)
{
    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_D2Z, batch, &workSize) == HIPFFT_SUCCESS);
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(double));
    }

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeZ2D)
{
    hipfftHandle plan = NULL;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_Z2D, batch, &workSize) == HIPFFT_SUCCESS);
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(double));
    }

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, RunR2C)
{
    const size_t N = 4096;
    float        in[N];
    for(size_t i = 0; i < N; i++)
        in[i] = i + (i % 3) - (i % 7);

    hipfftReal*    d_in;
    hipfftComplex* d_out;
    hipMalloc(&d_in, N * sizeof(hipfftReal));
    hipMalloc(&d_out, (N / 2 + 1) * sizeof(hipfftComplex));

    hipMemcpy(d_in, in, N * sizeof(hipfftReal), hipMemcpyHostToDevice);

    hipfftHandle plan = NULL;
    hipfftCreate(&plan);
    size_t workSize;
    hipfftMakePlan1d(plan, N, HIPFFT_R2C, 1, &workSize);

    hipfftExecR2C(plan, d_in, d_out);

    std::vector<hipfftComplex> out(N / 2 + 1);
    hipMemcpy(&out[0], d_out, (N / 2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    hipfftDestroy(plan);
    hipFree(d_in);
    hipFree(d_out);
    if(N % 2 != 0)
    {
        EXPECT_TRUE(workSize != 0);
    }

    double ref_in[N];
    for(size_t i = 0; i < N; i++)
        ref_in[i] = in[i];

    fftw_complex* ref_out;
    fftw_plan     ref_p;

    ref_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    ref_p   = fftw_plan_dft_r2c_1d(N, ref_in, ref_out, FFTW_ESTIMATE);
    fftw_execute(ref_p);

    double maxv  = 0;
    double nrmse = 0; // normalized root mean square error
    for(size_t i = 0; i < (N / 2 + 1); i++)
    {
        // printf("element %d: FFTW result %f, %f; hipFFT result %f, %f \n", \
        //   (int)i, ref_out[i][0], ref_out[i][1], out[i].x, out[i].y);
        double dr = ref_out[i][0] - out[i].x;
        double di = ref_out[i][1] - out[i].y;
        maxv      = fabs(ref_out[i][0]) > maxv ? fabs(ref_out[i][0]) : maxv;
        maxv      = fabs(ref_out[i][1]) > maxv ? fabs(ref_out[i][1]) : maxv;

        nrmse += ((dr * dr) + (di * di));
    }
    nrmse /= (double)((N / 2 + 1));
    nrmse = sqrt(nrmse);
    nrmse /= maxv;

    EXPECT_TRUE(nrmse < rmse_tolerance);
    fftw_free(ref_out);
}
