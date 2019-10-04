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
    std::cout << "hipfft 2D single-precision real-to-complex transform using advanced interface\n";

    int rank    = 2;
    int n[2]    = {4, 5};
    int howmany = 3; // batch size

    int n1_complex_elements      = n[1] / 2 + 1;
    int n1_padding_real_elements = n1_complex_elements * 2;

    int istride    = 1;
    int ostride    = istride;
    int inembed[2] = {istride * n[0], istride * n1_padding_real_elements};
    int onembed[2] = {ostride * n[0], ostride * n1_complex_elements};
    int idist      = inembed[0] * inembed[1];
    int odist      = onembed[0] * onembed[1];

    std::cout << "n: " << n[0] << " " << n[1] << "\n"
              << "howmany: " << howmany << "\n"
              << "istride: " << istride << "\tostride: " << ostride << "\n"
              << "inembed: " << inembed[0] << " " << inembed[1] << "\n"
              << "onembed: " << onembed[0] << " " << onembed[1] << "\n"
              << "idist: " << idist << "\todist: " << odist << "\n"
              << std::endl;

    std::vector<float> data(howmany * idist);
    const auto         total_bytes = data.size() * sizeof(decltype(data)::value_type);

    std::cout << "input:\n";
    std::fill(data.begin(), data.end(), 0.0);
    for(int ibatch = 0; ibatch < howmany; ++ibatch)
    {
        for(int i = 0; i < n[0]; i++)
        {
            for(int j = 0; j < n[1]; j++)
            {
                const auto pos = ibatch * idist + istride * (i * inembed[1] + j);
                data[pos]      = i + ibatch + j;
            }
        }
    }
    for(int ibatch = 0; ibatch < howmany; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        for(int i = 0; i < inembed[0]; i++)
        {
            for(int j = 0; j < inembed[1]; j++)
            {
                const auto pos = ibatch * idist + i * inembed[1] + j;
                std::cout << data[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftHandle hipForwardPlan;
    hipfftResult result;
    result = hipfftPlanMany(&hipForwardPlan,
                            rank,
                            n,
                            inembed,
                            istride,
                            idist,
                            onembed,
                            ostride,
                            odist,
                            HIPFFT_R2C, // Use HIPFFT_D2Z for double-precsion.
                            howmany);

    hipfftReal* gpu_data;
    hipMalloc((void**)&gpu_data, total_bytes);
    hipMemcpy(gpu_data, (void*)data.data(), total_bytes, hipMemcpyHostToDevice);

    result = hipfftExecR2C(hipForwardPlan, gpu_data, (hipfftComplex*)gpu_data);

    hipMemcpy((void*)data.data(), gpu_data, total_bytes, hipMemcpyDeviceToHost);

    std::cout << "output:\n";
    const std::complex<float>* output = (std::complex<float>*)data.data();
    for(int ibatch = 0; ibatch < howmany; ++ibatch)
    {
        std::cout << "batch: " << ibatch << "\n";
        for(int i = 0; i < onembed[0]; i++)
        {
            for(int j = 0; j < onembed[1]; j++)
            {
                const auto pos = ibatch * odist + i * onembed[1] + j;
                std::cout << output[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftDestroy(hipForwardPlan);
    hipFree(gpu_data);
    return 0;
}
