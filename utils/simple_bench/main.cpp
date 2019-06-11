/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "kernel_launch.h"
#include "twiddles.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " length_option batch" << std::endl;
        std::cout << "length_option is either 4m or 16m; batch is number of "
                     "transforms (positive integer);"
                  << std::endl;
        std::cout << "Example: " << argv[0] << " 4m 8" << std::endl;

        return -1;
    }

    DeviceCallIn  dci[5];
    DeviceCallOut dco[5];

    size_t b = 1;
    size_t n = 1;

    std::string a1(argv[1]);
    std::string a2(argv[2]);

    if(a1.compare("16m") == 0)
        n = 4096;
    else
        n = 2048;

    b = std::stoi(a2);

    size_t nPad = n + 64;

    float* cin = (float*)malloc(b * n * n * 2 * sizeof(float));
    float* cot = (float*)malloc(b * n * n * 2 * sizeof(float));

    for(size_t i = 0; i < b * n * n * 2; i++)
        cin[i] = 1.0;

    float* in = nullptr;
    float* ot = nullptr;
    float* tp = nullptr;

    hipMalloc(&in, b * n * n * 2 * sizeof(float));
    hipMalloc(&ot, b * n * n * 2 * sizeof(float));
    hipMalloc(&tp, b * n * nPad * 2 * sizeof(float));

    hipMemcpy(in, cin, b * n * n * 2 * sizeof(float), hipMemcpyHostToDevice);

    for(size_t i = 0; i < 5; i++)
    {
        dci[i].batch = b;
        dci[i].length.push_back(n);
        dci[i].length.push_back(n);
        dci[i].inStride.push_back(1);
        dci[i].outStride.push_back(1);
        dci[i].direction    = -1;
        dci[i].precision    = rocfft_precision_single;
        dci[i].inArrayType  = rocfft_array_type_complex_interleaved;
        dci[i].outArrayType = rocfft_array_type_complex_interleaved;
        dci[i].large1D      = 0;
        dci[i].transTileDir = TTD_IP_HOR;
    }

    // Tranpose 1
    dci[0].dimension = 2;
    dci[0].inStride.push_back(n);
    dci[0].outStride.push_back(nPad);
    dci[0].iDist     = n * n;
    dci[0].oDist     = n * nPad;
    dci[0].placement = rocfft_placement_notinplace;
    dci[0].scheme    = CS_KERNEL_TRANSPOSE;

    dci[0].bufIn[0]  = in;
    dci[0].bufOut[0] = tp;

    dci[0].gridParam.tpb_x = 16;
    dci[0].gridParam.tpb_y = 16;
    dci[0].gridParam.b_x   = dci[0].length[0] / 64;
    dci[0].gridParam.b_y   = (dci[0].length[1] / 64) * dci[0].batch;

    dci[0].devFnCall = &FN_PRFX(transpose_var1_sp);

    // fft 1
    dci[1].dimension = 1;
    dci[1].inStride.push_back(nPad);
    dci[1].outStride.push_back(n);
    dci[1].iDist     = n * nPad;
    dci[1].oDist     = n * n;
    dci[1].placement = rocfft_placement_notinplace;
    dci[1].scheme    = CS_KERNEL_STOCKHAM;
    dci[1].twiddles  = twiddles_create(dci[1].length[0]);

    dci[1].bufIn[0]  = tp;
    dci[1].bufOut[0] = ot;

    dci[1].gridParam.tpb_x = 256;
    dci[1].gridParam.b_x   = dci[1].length[1] * dci[1].batch;

    if(n == 4096)
        dci[1].devFnCall = &FN_PRFX(dfn_sp_op_ci_ci_stoc_2_4096);
    else
        dci[1].devFnCall = &FN_PRFX(dfn_sp_op_ci_ci_stoc_2_2048);

    // Tranpose 2
    dci[2].dimension = 2;
    dci[2].inStride.push_back(n);
    dci[2].outStride.push_back(nPad);
    dci[2].iDist          = n * n;
    dci[2].oDist          = n * nPad;
    dci[2].placement      = rocfft_placement_notinplace;
    dci[2].scheme         = CS_KERNEL_TRANSPOSE;
    dci[2].large1D        = n * n;
    dci[2].twiddles_large = twiddles_create(dci[2].large1D);

    dci[2].bufIn[0]  = ot;
    dci[2].bufOut[0] = tp;

    dci[2].gridParam.tpb_x = 16;
    dci[2].gridParam.tpb_y = 16;
    dci[2].gridParam.b_x   = dci[2].length[0] / 64;
    dci[2].gridParam.b_y   = (dci[2].length[1] / 64) * dci[2].batch;

    dci[2].devFnCall = &FN_PRFX(transpose_var1_sp);

    // fft 2
    dci[3].dimension = 1;
    dci[3].inStride.push_back(nPad);
    dci[3].outStride.push_back(nPad);
    dci[3].iDist     = n * nPad;
    dci[3].oDist     = n * nPad;
    dci[3].placement = rocfft_placement_inplace;
    dci[3].scheme    = CS_KERNEL_STOCKHAM;
    dci[3].twiddles  = twiddles_create(dci[3].length[0]);

    dci[3].bufIn[0]  = tp;
    dci[3].bufOut[0] = tp;

    dci[3].gridParam.tpb_x = 256;
    dci[3].gridParam.b_x   = dci[3].length[1] * dci[3].batch;

    if(n == 4096)
        dci[3].devFnCall = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_4096);
    else
        dci[3].devFnCall = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_2048);

    // Tranpose 3
    dci[4].dimension = 2;
    dci[4].inStride.push_back(nPad);
    dci[4].outStride.push_back(n);
    dci[4].iDist        = n * nPad;
    dci[4].oDist        = n * n;
    dci[4].placement    = rocfft_placement_notinplace;
    dci[4].scheme       = CS_KERNEL_TRANSPOSE;
    dci[4].transTileDir = TTD_IP_VER;

    dci[4].bufIn[0]  = tp;
    dci[4].bufOut[0] = ot;

    dci[4].gridParam.tpb_x = 16;
    dci[4].gridParam.tpb_y = 16;
    dci[4].gridParam.b_x   = dci[4].length[1] / 64;
    dci[4].gridParam.b_y   = (dci[4].length[0] / 64) * dci[4].batch;

    dci[4].devFnCall = &FN_PRFX(transpose_var1_sp);

    // warm-up run
    for(size_t i = 0; i < 5; i++)
    {
        DevFnCall fn = dci[i].devFnCall;
        fn(&dci[i], &dco[i]);
    }
    hipDeviceSynchronize();

    // timed run
    hipEvent_t start[5], stop[5];
    for(size_t i = 0; i < 5; i++)
    {
        hipEventCreate(&start[i]);
        hipEventCreate(&stop[i]);

        hipEventRecord(start[i]);
        DevFnCall fn = dci[i].devFnCall;
        fn(&dci[i], &dco[i]);
        hipEventRecord(stop[i]);
    }

    hipDeviceSynchronize();

    std::cout << std::endl;
    std::cout << "Running transform size of " << n * n << " with batch of " << b
              << ", for total data size of " << (float)(n * n * b * 2 * sizeof(float)) / 1e6
              << " MB" << std::endl;
    std::cout << "Computation requires 5 kernels" << std::endl << std::endl;

    for(size_t i = 0; i < 5; i++)
    {
        float gpu_time;
        hipEventElapsedTime(&gpu_time, start[i], stop[i]);
        // std::cout << "Kernel " << i << " execution time: " << gpu_time << " ms"
        // << std::endl;

        double gbps = (double)(b * n * n * 2 * sizeof(float)) / (0.5 * 1e6 * gpu_time);
        std::cout << "Kernel " << i << " bandwidth: " << gbps << " GB/s" << std::endl;
        hipEventDestroy(start[i]);
        hipEventDestroy(stop[i]);
    }

    hipFree(in);
    hipFree(ot);
    hipFree(tp);

    free(cin);
    free(cot);

    return 0;
}
