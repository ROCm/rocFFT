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
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft.h>

// Kernel for initializing the real-valued input data on the GPU.
__global__ void initdata(double* x, const int Nx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < Nx)
    {
        x[idx] = idx;
    }
}

// Helper function for determining grid dimensions
template <typename Tint1, typename Tint2>
Tint1 ceildiv(const Tint1 nominator, const Tint2 denominator)
{
    return (nominator + denominator - 1) / denominator;
}

int main()
{
    std::cout << "hipfft 1D double-precision real-to-complex transform\n";

    const size_t Nx = 8;

    const size_t Ncomplex = Nx / 2 + 1;

    std::vector<double> rdata(Nx);
    size_t              real_bytes = sizeof(decltype(rdata)::value_type) * rdata.size();
    std::vector<std::complex<decltype(rdata)::value_type>> cdata(Ncomplex);
    size_t complex_bytes = sizeof(std::complex<decltype(rdata)::value_type>) * cdata.size();

    // Create HIP device object
    double* x;
    hipMalloc(&x, complex_bytes);

    // Inititalize the data on the device
    hipError_t rt;
    const dim3 blockdim(256);
    const dim3 griddim(ceildiv(Nx, blockdim.x));
    hipLaunchKernelGGL(initdata, blockdim, griddim, 0, 0, x, Nx);
    hipDeviceSynchronize();
    rt = hipGetLastError();
    assert(rt == hipSuccess);

    std::cout << "input:\n";
    hipMemcpy(rdata.data(), x, real_bytes, hipMemcpyDefault);
    for(size_t i = 0; i < rdata.size(); i++)
    {
        std::cout << rdata[i] << " ";
    }
    std::cout << std::endl;

    // Create the plan
    hipfftHandle plan = NULL;
    hipfftResult rc   = HIPFFT_SUCCESS;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan1d(&plan, // plan handle
                      Nx, // transform length
                      HIPFFT_D2Z, // transform type (HIPFFT_R2C for single-precision)
                      1); // number of transforms (deprecated)
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan:
    // hipfftExecD2Z: double precision, hipfftExecR2C: for single-precision
    // Direction is implied by real-to-complex direction
    rc = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)x);
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "output:\n";
    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);
    hipFree(x);

    return 0;
}
