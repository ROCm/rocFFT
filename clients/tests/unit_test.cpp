// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "private.h"
#include "rocfft.h"
#include <boost/scope_exit.hpp>
#include <condition_variable>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <thread>
#include <vector>

TEST(rocfft_UnitTest, plan_description)
{
    rocfft_plan_description desc = nullptr;
    EXPECT_TRUE(rocfft_status_success == rocfft_plan_description_create(&desc));

    rocfft_array_type in_array_type  = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;

    size_t rank = 1;

    size_t i_strides[3] = {1, 1, 1};
    size_t o_strides[3] = {1, 1, 1};

    size_t idist = 0;
    size_t odist = 0;

    rocfft_plan plan   = NULL;
    size_t      length = 8;

    EXPECT_TRUE(rocfft_status_success
                == rocfft_plan_description_set_data_layout(desc,
                                                           in_array_type,
                                                           out_array_type,
                                                           0,
                                                           0,
                                                           rank,
                                                           i_strides,
                                                           idist,
                                                           rank,
                                                           o_strides,
                                                           odist));
    EXPECT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      rank,
                                      &length,
                                      1,
                                      desc));

    EXPECT_TRUE(rocfft_status_success == rocfft_plan_description_destroy(desc));
    EXPECT_TRUE(rocfft_status_success == rocfft_plan_destroy(plan));
}

// This test check duplicated plan creations will cache unique one only in repo.
TEST(rocfft_UnitTest, cache_plans_in_repo)
{
    rocfft_setup();
    size_t plan_unique_count = 0;
    size_t plan_total_count  = 0;
    size_t length            = 8;

    // Create plan0
    rocfft_plan plan0 = NULL;

    rocfft_plan_create(&plan0,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       1,
                       &length,
                       1,
                       NULL);

    rocfft_repo_get_unique_plan_count(&plan_unique_count);
    EXPECT_TRUE(plan_unique_count == 1);
    rocfft_repo_get_total_plan_count(&plan_total_count);
    EXPECT_TRUE(plan_total_count == 1);

    // Create plan1 with the same params to plan0
    rocfft_plan plan1 = NULL;
    rocfft_plan_create(&plan1,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       1,
                       &length,
                       1,
                       NULL);

    // Check we cached only 1 plan_t in repo
    rocfft_repo_get_unique_plan_count(&plan_unique_count);
    EXPECT_TRUE(plan_unique_count == 1);
    rocfft_repo_get_total_plan_count(&plan_total_count);
    EXPECT_TRUE(plan_total_count == 2);

    rocfft_plan_destroy(plan0);

    rocfft_repo_get_unique_plan_count(&plan_unique_count);
    EXPECT_TRUE(plan_unique_count == 1);
    rocfft_repo_get_total_plan_count(&plan_total_count);
    EXPECT_TRUE(plan_total_count == 1);

    rocfft_plan_destroy(plan1);

    rocfft_repo_get_unique_plan_count(&plan_unique_count);
    EXPECT_TRUE(plan_unique_count == 0);
    rocfft_repo_get_total_plan_count(&plan_total_count);
    EXPECT_TRUE(plan_total_count == 0);

    rocfft_cleanup();
}

std::mutex              test_mutex;
std::condition_variable test_cv;
int                     created          = 0;
bool                    ready_to_destroy = false;

void creation_test()
{
    rocfft_plan plan   = NULL;
    size_t      length = 8;

    {
        std::lock_guard<std::mutex> lk(test_mutex);

        rocfft_plan_create(&plan,
                           rocfft_placement_inplace,
                           rocfft_transform_type_complex_forward,
                           rocfft_precision_single,
                           1,
                           &length,
                           1,
                           NULL);
        created++;
        //std::cout << "created " << created << std::endl;
        test_cv.notify_all();
    }

    {
        std::unique_lock<std::mutex> lk(test_mutex);
        test_cv.wait(lk, [] { return ready_to_destroy; });
    }
    //std::cout << "will destroy\n";
    rocfft_plan_destroy(plan);
}

// Multithreading check cache_plans_in_repo case
TEST(rocfft_UnitTest, cache_plans_in_repo_multithreading)
{
    rocfft_setup();

    size_t plan_unique_count = 0;
    size_t plan_total_count  = 0;

    std::thread t0(creation_test);
    std::thread t1(creation_test);

    {
        std::unique_lock<std::mutex> lk(test_mutex);
        test_cv.wait(lk, [] { return created >= 2; });
    }

    //std::cout << "started to count\n";
    rocfft_repo_get_unique_plan_count(&plan_unique_count);
    EXPECT_TRUE(plan_unique_count == 1);
    rocfft_repo_get_total_plan_count(&plan_total_count);
    EXPECT_TRUE(plan_total_count == 2);

    {
        std::lock_guard<std::mutex> lk(test_mutex);
        ready_to_destroy = true;
        //std::cout << "count done, notify destroy\n";
        test_cv.notify_all();
    }

    t0.join();
    t1.join();

    rocfft_repo_get_unique_plan_count(&plan_unique_count);
    EXPECT_TRUE(plan_unique_count == 0);
    rocfft_repo_get_total_plan_count(&plan_total_count);
    EXPECT_TRUE(plan_total_count == 0);

    rocfft_cleanup();
}

// Check whether logs can be emitted from multiple threads properly
TEST(rocfft_UnitTest, log_multithreading)
{
    static const int   NUM_THREADS          = 10;
    static const int   NUM_ITERS_PER_THREAD = 50;
    static const char* TRACE_FILE           = "trace.log";

    // ask for trace logging, since that's the easiest to trigger
    setenv("ROCFFT_LAYER", "1", 1);
    setenv("ROCFFT_LOG_TRACE_PATH", TRACE_FILE, 1);

    // clean up environment and temporary file when we exit
    BOOST_SCOPE_EXIT_ALL(=)
    {
        unsetenv("ROCFFT_LAYER");
        unsetenv("ROCFFT_LOG_TRACE_PATH");
        remove(TRACE_FILE);
    };

    rocfft_setup();

    // run a whole bunch of threads in parallel, each one doing
    // something small that will write to the trace log
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([]() {
            for(int j = 0; j < NUM_ITERS_PER_THREAD; ++j)
            {
                rocfft_plan_description desc;
                rocfft_plan_description_create(&desc);
                rocfft_plan_description_destroy(desc);
            }
        });
    }

    for(auto& t : threads)
    {
        t.join();
    }

    rocfft_cleanup();

    // now verify that the trace log has one message per line, with nothing garbled
    std::ifstream trace_log(TRACE_FILE);
    std::string   line;
    std::regex    validator("^rocfft_(setup|cleanup|plan_description_(create|destroy),"
                         "description,0x[0-9a-f]+)$");
    while(std::getline(trace_log, line))
    {
        bool res = std::regex_match(line, validator);
        ASSERT_TRUE(res) << "line contains invalid content: " << line;
    }
}
