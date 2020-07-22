// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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
#include "../client_utils.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <thread>
#include <vector>

// normalize results of an inverse transform, so it can be directly
// compared to the original data before the forward transform
__global__ void normalize_inverse_results(float2* array, float N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx].x /= N;
    array[idx].y /= N;
}

// Run a transform of specified dimensions, size N on each dimension.
// Data is randomly generated based on the seed value, and we do a
// forward + inverse transform and compare against what we started
// with.
struct Test_Transform
{
    // real constructor sets all the data up and creates the plans
    Test_Transform(size_t _N, size_t _dim, uint32_t _seed)
        : N(_N)
        , dim(_dim)
        , seed(_seed)
    {
        // compute total data size
        size_t datasize = 1;
        for(size_t i = 0; i < dim; ++i)
        {
            datasize *= N;
        }

        size_t Nbytes = datasize * sizeof(float2);

        // Create HIP device buffers
        EXPECT_EQ(hipMalloc(&device_mem_in, Nbytes), hipSuccess);
        EXPECT_EQ(hipMalloc(&device_mem_out, Nbytes), hipSuccess);

        // Initialize data
        std::minstd_rand                      gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        host_mem_in.resize(datasize);
        host_mem_out.resize(datasize);
        for(size_t i = 0; i < datasize; i++)
        {
            host_mem_in[i].x = dist(gen);
            host_mem_in[i].y = dist(gen);
        }

        //  Copy data to device
        EXPECT_EQ(hipMemcpy(device_mem_in, host_mem_in.data(), Nbytes, hipMemcpyHostToDevice),
                  hipSuccess);
    }
    Test_Transform(const Test_Transform&) = delete;
    void operator=(const Test_Transform&) = delete;
    Test_Transform(Test_Transform&& other)
    {
        this->stream         = other.stream;
        other.stream         = nullptr;
        this->work_buffer    = other.work_buffer;
        other.work_buffer    = nullptr;
        this->device_mem_in  = other.device_mem_in;
        other.device_mem_in  = nullptr;
        this->device_mem_out = other.device_mem_out;
        other.device_mem_out = nullptr;
        this->host_mem_in.swap(other.host_mem_in);
        this->host_mem_out.swap(other.host_mem_out);
    }

    void run_transform()
    {
        // Create rocFFT plans (forward + inverse)
        std::vector<size_t> lengths(dim, N);
        EXPECT_EQ(rocfft_plan_create(&plan,
                                     rocfft_placement_notinplace,
                                     rocfft_transform_type_complex_forward,
                                     rocfft_precision_single,
                                     dim,
                                     lengths.data(),
                                     1,
                                     nullptr),
                  rocfft_status_success);

        EXPECT_EQ(rocfft_plan_create(&plan_inv,
                                     rocfft_placement_inplace,
                                     rocfft_transform_type_complex_inverse,
                                     rocfft_precision_single,
                                     dim,
                                     lengths.data(),
                                     1,
                                     nullptr),
                  rocfft_status_success);

        // allocate work buffer if necessary
        EXPECT_EQ(rocfft_plan_get_work_buffer_size(plan, &work_buffer_size), rocfft_status_success);
        // NOTE: assuming that same-sized work buffer is ok for both
        // forward and inverse transforms
        if(work_buffer_size)
        {
            EXPECT_EQ(hipMalloc(&work_buffer, work_buffer_size), hipSuccess);
        }

        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        rocfft_execution_info info;
        EXPECT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);
        EXPECT_EQ(rocfft_execution_info_set_stream(info, stream), rocfft_status_success);
        EXPECT_EQ(rocfft_execution_info_set_work_buffer(info, work_buffer, work_buffer_size),
                  rocfft_status_success);

        // Execute forward plan out-of-place
        EXPECT_EQ(rocfft_execute(plan, &device_mem_in, &device_mem_out, info),
                  rocfft_status_success);
        // Execute inverse plan in-place
        EXPECT_EQ(rocfft_execute(plan_inv, &device_mem_out, nullptr, info), rocfft_status_success);

        EXPECT_EQ(rocfft_execution_info_destroy(info), rocfft_status_success);

        // Apply normalization so the values really are comparable
        hipLaunchKernelGGL(normalize_inverse_results,
                           host_mem_out.size(),
                           1,
                           0, // sharedMemBytes
                           stream, // stream
                           static_cast<float2*>(device_mem_out),
                           static_cast<float>(host_mem_out.size()));
        ran_transform = true;
    }

    void do_cleanup()
    {
        // complain loudly if we set up for a transform but did not
        // actually run it
        if(plan && !ran_transform)
            ADD_FAILURE();

        // wait for execution to finish
        if(stream)
        {
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
            stream = nullptr;
        }

        EXPECT_EQ(hipFree(work_buffer), hipSuccess);
        work_buffer = nullptr;

        EXPECT_EQ(rocfft_plan_destroy(plan), rocfft_status_success);
        plan = nullptr;
        EXPECT_EQ(rocfft_plan_destroy(plan_inv), rocfft_status_success);
        plan_inv = nullptr;

        // Copy result back to host
        if(device_mem_out && !host_mem_out.empty())
        {
	  
            EXPECT_EQ(hipMemcpy(host_mem_out.data(),
                                device_mem_out,
                                host_mem_out.size() * sizeof(float2),
                                hipMemcpyDeviceToHost),
                      hipSuccess);

            // Compare data we got to the original.
	    // We're running 2 transforms (forward+inverse), so we
            // should tolerate 2x the error of a single transform.
	    std::vector<std::pair<size_t, size_t>> linf_failures;
            const double MAX_TRANSFORM_ERROR = 2 * type_epsilon<float>();
            auto diff = LinfL2diff_1to1_complex(
                reinterpret_cast<const std::complex<float>*>(host_mem_in.data()),
                reinterpret_cast<const std::complex<float>*>(host_mem_out.data()),
                // data is all contiguous, we can treat it as 1d
                host_mem_in.size(),
                1,
                1,
                host_mem_in.size(),
                1,
                host_mem_out.size(),
		linf_failures,
		MAX_TRANSFORM_ERROR);
            EXPECT_LT(diff.first, MAX_TRANSFORM_ERROR);
            EXPECT_LT(diff.second, MAX_TRANSFORM_ERROR);

            // Free device buffers
            EXPECT_EQ(hipFree(device_mem_in), hipSuccess);
            device_mem_in = nullptr;
            EXPECT_EQ(hipFree(device_mem_out), hipSuccess);
            device_mem_out = nullptr;
            host_mem_in.clear();
            host_mem_out.clear();
        }
    }
    ~Test_Transform()
    {
        do_cleanup();
    }
    size_t              N                = 0;
    size_t              dim              = 0;
    uint32_t            seed             = 0;
    hipStream_t         stream           = nullptr;
    rocfft_plan         plan             = nullptr;
    rocfft_plan         plan_inv         = nullptr;
    size_t              work_buffer_size = 0;
    void*               work_buffer      = nullptr;
    void*               device_mem_in    = nullptr;
    void*               device_mem_out   = nullptr;
    std::vector<float2> host_mem_in;
    std::vector<float2> host_mem_out;

    // ensure that we don't forget to actually run the transform
    bool ran_transform = false;
};

// run concurrent transforms, one per thread, size N on each dimension
static void multithread_transform(size_t N, size_t dim, size_t num_threads)
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t j = 0; j < num_threads; ++j)
    {
        threads.emplace_back([=]() {
            Test_Transform t(N, dim, j);
            t.run_transform();
        });
    }
    for(auto& t : threads)
        t.join();
}

// for multi-stream tests, set up a bunch of streams, then execute
// all of those transforms from a single thread.  afterwards,
// wait/verify/cleanup in parallel to save wall time during the test.
static void multistream_transform(size_t N, size_t dim, size_t num_streams)
{
    std::vector<std::unique_ptr<Test_Transform>> transforms;
    transforms.resize(num_streams);
    std::vector<std::thread> threads;
    threads.reserve(num_streams);

    // get all data ready in parallel
    for(size_t i = 0; i < num_streams; ++i)
        threads.emplace_back(
            [=, &transforms]() { transforms[i] = std::make_unique<Test_Transform>(N, dim, i); });
    for(auto& t : threads)
        t.join();
    threads.clear();

    // now start the actual transforms serially, but in separate
    // streams
    for(auto& t : transforms)
        t->run_transform();

    // clean up
    for(size_t i = 0; i < transforms.size(); ++i)
        threads.emplace_back([=, &transforms]() { transforms[i]->do_cleanup(); });
    for(auto& t : threads)
        t.join();
}

// pick arbitrary sizes here to get some parallelism while still
// fitting into e.g. 8 GB of GPU memory
TEST(rocfft_UnitTest, simple_multithread_1D)
{
    multithread_transform(1048576, 1, 64);
}

TEST(rocfft_UnitTest, simple_multithread_2D)
{
    multithread_transform(1024, 2, 64);
}

TEST(rocfft_UnitTest, simple_multithread_3D)
{
    multithread_transform(128, 3, 40);
}

TEST(rocfft_UnitTest, simple_multistream_1D)
{
    multistream_transform(1048576, 1, 32);
}

TEST(rocfft_UnitTest, simple_multistream_2D)
{
    multistream_transform(1024, 2, 32);
}

TEST(rocfft_UnitTest, simple_multistream_3D)
{
    multistream_transform(128, 3, 32);
}
