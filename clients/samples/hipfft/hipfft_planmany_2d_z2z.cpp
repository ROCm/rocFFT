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
    ///------------------------------------------------------------------------
    /// 2d Z2Z forward hipfftPlanMany explicit params

    int      rank    = 2;
    int      n[2]    = {5, 6};
    int      howmany = 3;
    int      idist   = n[0] * n[1];
    int      odist   = n[0] * n[1];
    int      istride = 1;
    int      ostride = 1; // array is contiguous in memory
    int *    inembed = n, *onembed = n;
    int      sign        = FFTW_FORWARD;
    unsigned flags       = FFTW_PATIENT;
    size_t   total_bytes = sizeof(fftw_complex) * howmany * n[0] * n[1];

    fftw_complex* fftw_in  = (fftw_complex*)fftw_malloc(total_bytes);
    fftw_complex* fftw_out = (fftw_complex*)fftw_malloc(total_bytes);

    const fftw_plan fftwPlan = fftw_plan_many_dft(rank,
                                                  n,
                                                  howmany,
                                                  fftw_in,
                                                  inembed,
                                                  istride,
                                                  idist,
                                                  fftw_out,
                                                  onembed,
                                                  ostride,
                                                  odist,
                                                  sign,
                                                  flags);

    for(int i = 0; i < n[0]; i++)
        for(int j = 0; j < n[1]; j++)
        {
            fftw_in[i * n[1] + j][0] = fftw_in[i * n[1] + j][1] = (i * n[1] + j + 1 + 30) % 12;
            fftw_out[i * n[1] + j][0] = fftw_out[i * n[1] + j][1] = 0;
        }

    fftw_execute(fftwPlan);

    std::cout << "fftw inputs:\n";
    std::cout.setf(ios::fixed);
    std::cout.setf(ios::right);
    std::cout.setf(ios::showpos);
    for(int i = 0; i < n[0]; i++)
    {
        for(int j = 0; j < n[1]; j++)
        {
            std::cout << "[" << i << ", " << j << "] : (" << setprecision(2) << setw(6)
                      << fftw_in[i * n[1] + j][0] << ", " << setprecision(2) << setw(6)
                      << fftw_in[i * n[1] + j][1] << ")  ";
        }
        std::cout << std::endl;
    }
    std::cout << "fftw outputs:\n";
    for(int i = 0; i < n[0]; i++)
    {
        for(int j = 0; j < n[1]; j++)
        {
            std::cout << "[" << i << ", " << j << "] : (" << setprecision(2) << setw(6)
                      << fftw_out[i * n[1] + j][0] << ", " << setprecision(2) << setw(6)
                      << fftw_out[i * n[1] + j][1] << ")  ";
        }
        std::cout << std::endl;
    }

    for(int i = 0; i < n[0]; i++)
        for(int j = 0; j < n[1]; j++)
        {
            fftw_out[i * n[1] + j][0] = fftw_out[i * n[1] + j][1] = 0;
        }

    hipfftHandle hipPlan;
    hipfftResult result;

    result = hipfftPlanMany(
        &hipPlan, rank, n, inembed, istride, idist, onembed, ostride, odist, HIPFFT_Z2Z, howmany);

    hipfftDoubleComplex* d_in_out;
    hipMalloc((void**)&d_in_out, total_bytes);
    hipMemcpy(d_in_out, (void*)fftw_in, total_bytes, hipMemcpyHostToDevice);

    result = hipfftExecZ2Z(hipPlan, d_in_out, d_in_out, HIPFFT_FORWARD);

    hipMemcpy((void*)fftw_out, d_in_out, total_bytes, hipMemcpyDeviceToHost);

    std::cout << "hipfft outputs:\n";
    for(int i = 0; i < n[0]; i++)
    {
        for(int j = 0; j < n[1]; j++)
        {
            std::cout << "[" << i << ", " << j << "] : (" << setprecision(2) << setw(6)
                      << fftw_out[i * n[1] + j][0] << ", " << setprecision(2) << setw(6)
                      << fftw_out[i * n[1] + j][1] << ")  ";
        }
        std::cout << std::endl;
    }

    fftw_destroy_plan(fftwPlan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);

    hipFree(d_in_out);
}
