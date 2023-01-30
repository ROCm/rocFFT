// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocfft.h"
#include <cassert>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define CHECK_HIP_ERR(err)                                    \
    if(err != hipSuccess)                                     \
    {                                                         \
        std::cerr << "hip error code : " << err << std::endl; \
        exit(-1);                                             \
    }

#define CHECK_ROCFFT_ERR(err)                                    \
    if(err != rocfft_status_success)                             \
    {                                                            \
        std::cerr << "rocFFT error code : " << err << std::endl; \
        exit(-1);                                                \
    }

struct fft_fixture_t
{
    double2*              cpu_buf;
    double2*              gpu_buf;
    hipStream_t           stream;
    rocfft_execution_info info;
    rocfft_plan           plan;
};

int main(int argc, char* argv[])
{
    std::cout << "rocfft example of 2 inplace transforms with 2 streams.\n" << std::endl;

    size_t length      = 8;
    size_t total_bytes = length * sizeof(double2);

    fft_fixture_t ffts[2];

    /// preparation
    rocfft_setup();
    for(auto& it : ffts)
    {
        // create cpu buffer
        it.cpu_buf = new double2[length];

        // init cpu buffer...

        // create gpu buffer
        CHECK_HIP_ERR(hipMalloc(&(it.gpu_buf), total_bytes));

        // copy host to device
        CHECK_HIP_ERR(hipMemcpy(it.gpu_buf, it.cpu_buf, total_bytes, hipMemcpyHostToDevice));

        // create stream
        CHECK_HIP_ERR(hipStreamCreate(&(it.stream)));

        // create execution info
        CHECK_ROCFFT_ERR(rocfft_execution_info_create(&(it.info)));

        // set stream
        CHECK_ROCFFT_ERR(rocfft_execution_info_set_stream(it.info, it.stream));

        // create plan
        CHECK_ROCFFT_ERR(rocfft_plan_create(&it.plan,
                                            rocfft_placement_inplace,
                                            rocfft_transform_type_complex_forward,
                                            rocfft_precision_double,
                                            1,
                                            &length,
                                            1,
                                            nullptr));
        size_t work_buf_size = 0;
        CHECK_ROCFFT_ERR(rocfft_plan_get_work_buffer_size(it.plan, &work_buf_size));
        assert(work_buf_size == 0); // simple 1D inplace fft doesn't need extra working buffer
    }

    /// execution
    for(auto& it : ffts)
    {
        CHECK_ROCFFT_ERR(
            rocfft_execute(it.plan, (void**)&(it.gpu_buf), (void**)&(it.gpu_buf), nullptr));
    }

    /// wait and copy back
    for(auto& it : ffts)
    {
        CHECK_HIP_ERR(hipStreamSynchronize(it.stream));
        CHECK_HIP_ERR(hipMemcpy(it.cpu_buf, it.gpu_buf, total_bytes, hipMemcpyDeviceToHost));
    }

    /// clean up
    for(auto& it : ffts)
    {
        CHECK_ROCFFT_ERR(rocfft_plan_destroy(it.plan));
        CHECK_ROCFFT_ERR(rocfft_execution_info_destroy(it.info));
        CHECK_HIP_ERR(hipStreamDestroy(it.stream));
        CHECK_HIP_ERR(hipFree(it.gpu_buf));
        delete[] it.cpu_buf;
    }

    rocfft_cleanup();
    return 0;
}
