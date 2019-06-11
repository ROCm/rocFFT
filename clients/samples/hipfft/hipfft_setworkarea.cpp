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

#include <iomanip>
#include <iostream>

#include <fftw3.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>

using namespace std;

int main()
{
    int major_version;
    hipfftGetProperty(MAJOR_VERSION, &major_version);
    std::cout << "hipFFT major_version " << major_version << std::endl;

    const size_t N = 256;

    /// FFTW reference compute

    std::vector<float2> cx(N);

    size_t complex_input_bytes = sizeof(fftwf_complex) * (N / 2 + 1);
    size_t real_output_bytes   = sizeof(float) * N;

    fftwf_complex* in  = (fftwf_complex*)fftwf_malloc(complex_input_bytes);
    float*         out = (float*)fftwf_malloc(real_output_bytes);
    fftwf_plan     p   = fftwf_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);

    for(size_t i = 0; i < (N / 2 + 1); i++)
    {
        cx[i].x = in[i][0] = i + (i % 3) - (i % 7);
        cx[i].y = in[i][1] = 0;
    }

    fftwf_execute(p);

    /// hipfft gpu compute

    // Create HIP device object.
    hipfftComplex* x;
    hipMalloc(&x, std::max(complex_input_bytes, real_output_bytes));

    //  Copy input data to device
    hipMemcpy(x, &cx[0], complex_input_bytes, hipMemcpyHostToDevice);

    // Create plan
    hipfftHandle plan = NULL;
    size_t       workSize;

    hipfftEstimate1d(N, HIPFFT_C2R, 1, &workSize);
    std::cout << "hipfftEstimate1d workSize: " << workSize << std::endl;

    hipfftCreate(&plan);
    hipfftSetAutoAllocation(plan, 0);

    hipfftMakePlan1d(plan, N, HIPFFT_C2R, 1, &workSize);

    // Set work buffer
    hipfftComplex* workBuf;
    hipMalloc(&workBuf, workSize);
    hipfftSetWorkArea(plan, workBuf);

    hipfftGetSize(plan, &workSize);
    std::cout << "hipfftGetSize workSize: " << workSize << std::endl;

    // Execute plan
    hipfftExecC2R(plan, x, (hipfftReal*)x);

    // Copy result back to host
    std::vector<float> y(N);
    hipMemcpy(&y[0], x, real_output_bytes, hipMemcpyDeviceToHost);

    // Destroy plan
    hipfftDestroy(plan);

    double error      = 0;
    size_t element_id = 0;
    for(size_t i = 0; i < N; i++)
    {
        if(i < 32)
            printf("element %d: FFTW result %f; hipFFT result %f \n", (int)i, out[i], y[i]);
        double err = fabs(out[i] - y[i]);
        if(err > error)
        {
            error      = err;
            element_id = i;
        }
    }

    printf("......\nmax error of FFTW and hipFFT is %e at element %d\n",
           error / fabs(out[element_id]),
           (int)element_id);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    hipFree(x);
    hipFree(workBuf);

    return 0;
}
