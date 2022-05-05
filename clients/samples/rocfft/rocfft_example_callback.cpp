/******************************************************************************
* Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocfft.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <iostream>
#include <math.h>
#include <vector>

// example of using load/store callbacks with rocfft

struct load_cbdata
{
    double2* filter;
    double   scale;
};

__device__ double2 load_callback(double2* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto data = static_cast<load_cbdata*>(cbdata);

    // multiply each element by filter element and scale
    return hipCmul(hipCmul(input[offset], data->filter[offset]),
                   make_hipDoubleComplex(data->scale, data->scale));
}
__device__ auto load_callback_dev = load_callback;

int main()
{

    const size_t N = 8;

    std::vector<double2> cx(N), filter(N);

    // initialize data and filter
    for(size_t i = 0; i < N; i++)
    {
        cx[i].x     = i;
        cx[i].y     = i;
        filter[i].x = rand() / static_cast<double>(RAND_MAX);
        filter[i].y = 0;
    }

    // rocfft gpu compute
    // ========================================

    rocfft_setup();

    size_t Nbytes = N * sizeof(double2);

    // Create HIP device object.
    double2 *x, *filter_dev;
    hipMalloc(&x, Nbytes);
    hipMalloc(&filter_dev, Nbytes);

    //  Copy data to device
    hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);
    hipMemcpy(filter_dev, filter.data(), Nbytes, hipMemcpyHostToDevice);

    // Create plan
    rocfft_plan plan   = NULL;
    size_t      length = N;
    rocfft_plan_create(&plan,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_double,
                       1,
                       &length,
                       1,
                       NULL);

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
    void*                 work_buf = nullptr;
    rocfft_execution_info info     = nullptr;
    rocfft_execution_info_create(&info);
    if(work_buf_size)
    {
        hipMalloc(&work_buf, work_buf_size);
        rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
    }

    // prepare callback
    load_cbdata cbdata_host;
    cbdata_host.filter = filter_dev;
    cbdata_host.scale  = 1.0 / static_cast<double>(N);
    void* cbdata_dev;
    hipMalloc(&cbdata_dev, sizeof(load_cbdata));
    hipMemcpy(cbdata_dev, &cbdata_host, sizeof(load_cbdata), hipMemcpyHostToDevice);

    // get a properly-typed host pointer to the device function, as
    // rocfft_execution_info_set_load_callback expects void*.
    void* cbptr_host = nullptr;
    hipMemcpyFromSymbol(&cbptr_host, HIP_SYMBOL(load_callback_dev), sizeof(void*));

    // set callback
    rocfft_execution_info_set_load_callback(info, &cbptr_host, &cbdata_dev, 0);

    // Execute plan
    rocfft_execute(plan, (void**)&x, NULL, info);

    // Clean up work buffer
    if(work_buf_size)
    {
        hipFree(work_buf);
        rocfft_execution_info_destroy(info);
    }

    // Destroy plan
    rocfft_plan_destroy(plan);

    // Copy result back to host
    std::vector<double2> y(N);
    hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost);

    for(size_t i = 0; i < N; i++)
    {
        std::cout << "element " << i << " input:  (" << cx[i].x << "," << cx[i].y << ")"
                  << " output: (" << y[i].x << "," << y[i].y << ")" << std::endl;
    }

    hipFree(cbdata_dev);
    hipFree(filter_dev);
    hipFree(x);

    rocfft_cleanup();

    return 0;
}
