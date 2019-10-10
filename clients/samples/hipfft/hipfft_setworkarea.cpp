// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <complex>
#include <hip/hip_runtime_api.h>
#include <hipfft.h>
#include <iostream>
#include <vector>

int main()
{
    std::cout << "hipfft 1D single-precision real-to-complex transform showing work memory usage\n";

    int major_version;
    hipfftGetProperty(HIPFFT_MAJOR_VERSION, &major_version);
    std::cout << "hipFFT major_version " << major_version << std::endl;

    const size_t N        = 9;
    const size_t Ncomplex = (N / 2 + 1);

    std::vector<float>               rdata(N);
    std::vector<std::complex<float>> cdata(Ncomplex);

    size_t real_bytes    = sizeof(decltype(rdata)::value_type) * rdata.size();
    size_t complex_bytes = sizeof(decltype(cdata)::value_type) * cdata.size();

    std::cout << "input:\n";
    for(size_t i = 0; i < N; i++)
    {
        rdata[i] = i;
    }
    for(size_t i = 0; i < N; i++)
    {
        std::cout << rdata[i] << " ";
    }
    std::cout << std::endl;

    // Create HIP device object.
    hipfftReal* x;
    hipMalloc(&x, real_bytes);
    hipfftComplex* y;
    hipMalloc(&y, complex_bytes);

    // Copy input data to device
    hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);

    size_t workSize;
    hipfftEstimate1d(N, HIPFFT_R2C, 1, &workSize);
    std::cout << "hipfftEstimate 1d workSize: " << workSize << std::endl;

    hipfftHandle plan = NULL;
    hipfftCreate(&plan);
    hipfftSetAutoAllocation(plan, 0);
    hipfftMakePlan1d(plan, N, HIPFFT_R2C, 1, &workSize);

    // Set work buffer
    hipfftComplex* workBuf;
    hipMalloc(&workBuf, workSize);
    hipfftSetWorkArea(plan, workBuf);
    hipfftGetSize(plan, &workSize);
    std::cout << "hipfftGetSize workSize: " << workSize << std::endl;

    // Execute plan
    hipfftExecR2C(plan, x, (hipfftComplex*)y);

    // Copy result back to host
    hipMemcpy(cdata.data(), y, complex_bytes, hipMemcpyDeviceToHost);

    std::cout << "output:\n";
    for(size_t i = 0; i < Ncomplex; i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);
    hipFree(x);
    hipFree(workBuf);

    return 0;
}
