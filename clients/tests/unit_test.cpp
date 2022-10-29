// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../shared/environment.h"
#include "../../shared/gpubuf.h"
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include <boost/scope_exit.hpp>
#include <condition_variable>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <regex>
#include <thread>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

namespace fs = std::filesystem;

#ifndef WIN32
// get program_invocation_name
#include <errno.h>
#endif

TEST(rocfft_UnitTest, plan_description)
{
    rocfft_plan_description desc = nullptr;
    ASSERT_TRUE(rocfft_status_success == rocfft_plan_description_create(&desc));

    rocfft_array_type in_array_type  = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;

    size_t rank = 1;

    size_t i_strides[3] = {1, 1, 1};
    size_t o_strides[3] = {1, 1, 1};

    size_t idist = 0;
    size_t odist = 0;

    rocfft_plan plan   = NULL;
    size_t      length = 8;

    ASSERT_TRUE(rocfft_status_success
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
    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      rank,
                                      &length,
                                      1,
                                      desc));

    ASSERT_TRUE(rocfft_status_success == rocfft_plan_description_destroy(desc));
    ASSERT_TRUE(rocfft_status_success == rocfft_plan_destroy(plan));
}

// check that twiddles are reused between distinct plans
//
// NOTE: since we're observing twiddle creation indirectly by
// checking free device memory, this test can be unstable if other
// stuff is using the GPU
TEST(rocfft_UnitTest, repo_twiddle)
{
    rocfft_plan plan_forward = NULL;
    rocfft_plan plan_inverse = NULL;
    // 16M elements
    const size_t length = 1 << 24;

    // r2c post-processing twiddle table is 1/4 real length
    const auto R2C_TWIDDLE_SIZE = length / 4;

    // forward plan should need the same twiddles as an inverse
    // plan of the same size

    // check device memory usage
    size_t memFreeBefore = 0;
    size_t memFreeTotal  = 0;
    ASSERT_EQ(hipMemGetInfo(&memFreeBefore, &memFreeTotal), hipSuccess);

    // create forward plan
    ASSERT_EQ(rocfft_plan_create(&plan_forward,
                                 rocfft_placement_inplace,
                                 rocfft_transform_type_real_forward,
                                 rocfft_precision_single,
                                 1,
                                 &length,
                                 1,
                                 nullptr),
              rocfft_status_success);

    // we'd expect at least a r2c twiddle's worth of memory to be used
    size_t memFreeAfter1Plan = 0;
    ASSERT_EQ(hipMemGetInfo(&memFreeAfter1Plan, &memFreeTotal), hipSuccess);

    ASSERT_LT(memFreeAfter1Plan, memFreeBefore);
    ASSERT_GE(memFreeBefore - memFreeAfter1Plan, R2C_TWIDDLE_SIZE);

    // create inverse plan
    ASSERT_EQ(rocfft_plan_create(&plan_inverse,
                                 rocfft_placement_inplace,
                                 rocfft_transform_type_real_inverse,
                                 rocfft_precision_single,
                                 1,
                                 &length,
                                 1,
                                 nullptr),
              rocfft_status_success);
    size_t memFreeAfter2Plans = 0;
    ASSERT_EQ(hipMemGetInfo(&memFreeAfter2Plans, &memFreeTotal), hipSuccess);

    // the second plan should allocate some smaller buffers (e.g. for
    // kernel args) but definitely not the large twiddle table
    ASSERT_LE(memFreeAfter2Plans, memFreeAfter1Plan);
    ASSERT_LE(memFreeAfter1Plan - memFreeAfter2Plans, R2C_TWIDDLE_SIZE);

    rocfft_plan_destroy(plan_forward);
    rocfft_plan_destroy(plan_inverse);
}

// Check whether logs can be emitted from multiple threads properly
TEST(rocfft_UnitTest, log_multithreading)
{
    static const int   NUM_THREADS          = 10;
    static const int   NUM_ITERS_PER_THREAD = 50;
    static const char* TRACE_FILE           = "trace.log";

    // clean up environment and temporary file when we exit
    BOOST_SCOPE_EXIT_ALL(=)
    {
        rocfft_cleanup();
        remove(TRACE_FILE);
        // re-init logs with default logging
        rocfft_setup();
    };

    // ask for trace logging, since that's the easiest to trigger
    rocfft_cleanup();
    EnvironmentSetTemp layer("ROCFFT_LAYER", "1");
    EnvironmentSetTemp tracepath("ROCFFT_LOG_TRACE_PATH", TRACE_FILE);

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
                         "description,[x0-9a-fA-F]+)$");
    while(std::getline(trace_log, line))
    {
        bool res = std::regex_match(line, validator);
        ASSERT_TRUE(res) << "line contains invalid content: " << line;
    }
}

// a function that accepts a plan's requested size on input, and
// returns the size to actually allocate for the test
typedef std::function<size_t(size_t)> workmem_sizer;

void workmem_test(workmem_sizer sizer,
                  rocfft_status exec_status_expected,
                  bool          give_null_work_buf = false)
{
    // Prime size requires Bluestein, which guarantees work memory.
    size_t      length = 8191;
    rocfft_plan plan   = NULL;

    ASSERT_EQ(rocfft_plan_create(&plan,
                                 rocfft_placement_inplace,
                                 rocfft_transform_type_complex_forward,
                                 rocfft_precision_single,
                                 1,
                                 &length,
                                 1,
                                 nullptr),
              rocfft_status_success);

    size_t requested_work_size = 0;
    ASSERT_EQ(rocfft_plan_get_work_buffer_size(plan, &requested_work_size), rocfft_status_success);
    ASSERT_GT(requested_work_size, 0);

    rocfft_execution_info info;
    ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);

    size_t alloc_work_size = sizer(requested_work_size);
    gpubuf work_buffer;
    if(alloc_work_size)
    {
        ASSERT_EQ(work_buffer.alloc(alloc_work_size), hipSuccess);

        void*         work_buffer_ptr;
        rocfft_status set_work_expected_status;
        if(give_null_work_buf)
        {
            work_buffer_ptr          = nullptr;
            set_work_expected_status = rocfft_status_invalid_work_buffer;
        }
        else
        {
            work_buffer_ptr          = work_buffer.data();
            set_work_expected_status = rocfft_status_success;
        }
        ASSERT_EQ(rocfft_execution_info_set_work_buffer(info, work_buffer_ptr, alloc_work_size),
                  set_work_expected_status);
    }

    // allocate 2x length for complex
    std::vector<float> data_host(length * 2, 1.0f);
    gpubuf             data_device;
    auto               data_size_bytes = data_host.size() * sizeof(float);

    ASSERT_EQ(data_device.alloc(data_size_bytes), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(data_device.data(), data_host.data(), data_size_bytes, hipMemcpyHostToDevice),
        hipSuccess);
    std::vector<void*> ibuffers(1, static_cast<void*>(data_device.data()));

    ASSERT_EQ(rocfft_execute(plan, ibuffers.data(), nullptr, info), exec_status_expected);

    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
}

// check what happens if work memory is required but is not provided
// - library should allocate
TEST(rocfft_UnitTest, workmem_missing)
{
    workmem_test([](size_t) { return 0; }, rocfft_status_success);
}

// check what happens if work memory is required but not enough is provided
TEST(rocfft_UnitTest, workmem_small)
{
    workmem_test([](size_t requested) { return requested / 2; }, rocfft_status_invalid_work_buffer);
}

// hard to imagine this being a problem, but try giving too much as well
TEST(rocfft_UnitTest, workmem_big)
{
    workmem_test([](size_t requested) { return requested * 2; }, rocfft_status_success);
}

// check if a user explicitly gives a null pointer - set work buffer
// should fail, but transform should succeed because library
// allocates
TEST(rocfft_UnitTest, workmem_null)
{
    workmem_test([](size_t requested) { return requested; }, rocfft_status_success, true);
}

#ifdef ROCFFT_RUNTIME_COMPILE
static const size_t RTC_PROBLEM_SIZE = 2304;
// runtime compilation cache tests
TEST(rocfft_UnitTest, rtc_cache)
{
    // PRECONDITIONS

    // - set cache location to custom path, requires uninitializing
    //   the lib and reinitializing with some env vars
    // - also enable RTC logging so we can tell when something was
    //   actually compiled
    const std::string rtc_cache_path = std::tmpnam(nullptr);
    const std::string rtc_log_path   = std::tmpnam(nullptr);

    void*  empty_cache           = nullptr;
    size_t empty_cache_bytes     = 0;
    void*  onekernel_cache       = nullptr;
    size_t onekernel_cache_bytes = 0;

    // cleanup
    BOOST_SCOPE_EXIT_ALL(=)
    {
        // close log file handles
        rocfft_cleanup();
        remove(rtc_cache_path.c_str());
        remove(rtc_log_path.c_str());
        // re-init lib now that the env vars are gone
        rocfft_setup();
        if(empty_cache)
            rocfft_cache_buffer_free(empty_cache);
        if(onekernel_cache)
            rocfft_cache_buffer_free(onekernel_cache);
    };

    rocfft_cleanup();
    EnvironmentSetTemp cache_env("ROCFFT_RTC_CACHE_PATH", rtc_cache_path.c_str());
    EnvironmentSetTemp layer_env("ROCFFT_LAYER", "32");
    EnvironmentSetTemp log_env("ROCFFT_LOG_RTC_PATH", rtc_log_path.c_str());
    rocfft_setup();

    // - serialize empty cache as baseline
    ASSERT_EQ(rocfft_cache_serialize(&empty_cache, &empty_cache_bytes), rocfft_status_success);

    // END PRECONDITIONS

    // pick a length that's runtime compiled
    auto build_plan = [&]() {
        rocfft_plan plan = nullptr;
        ASSERT_TRUE(rocfft_status_success
                    == rocfft_plan_create(&plan,
                                          rocfft_placement_inplace,
                                          rocfft_transform_type_complex_forward,
                                          rocfft_precision_single,
                                          1,
                                          &RTC_PROBLEM_SIZE,
                                          1,
                                          nullptr));
        // we don't need to actually execute the plan, so we can
        // destroy it right away.  this ensures that we don't hold on
        // to a plan after we cleanup the library
        rocfft_plan_destroy(plan);
        plan = nullptr;
    };
    // check the RTC log to see if a kernel got compiled
    auto kernel_was_compiled = [&]() {
        // HACK: logging is done in a worker thread, so sleep for a
        // bit to give it a chance to actually write.  It at least
        // should flush after writing.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // look for a ROCFFT_RTC_BEGIN line that indicates RTC happened
        std::ifstream logfile(rtc_log_path);
        std::string   line;
        while(logfile >> line)
        {
            if(line.find("ROCFFT_RTC_BEGIN") != std::string::npos)
                return true;
        }
        return false;
    };

    // build a plan that requires runtime compilation,
    // close logs and ensure a kernel was built
    build_plan();
    ASSERT_EQ(rocfft_cache_serialize(&onekernel_cache, &onekernel_cache_bytes),
              rocfft_status_success);
    rocfft_cleanup();
    ASSERT_TRUE(kernel_was_compiled());

    // serialized cache should be bigger than empty cache
    ASSERT_GT(onekernel_cache_bytes, empty_cache_bytes);

    // blow away the cache, reinit the library,
    // retry building the plan again and ensure the kernel was rebuilt
    remove(rtc_cache_path.c_str());
    rocfft_setup();
    build_plan();
    rocfft_cache_buffer_free(onekernel_cache);
    onekernel_cache = nullptr;
    ASSERT_EQ(rocfft_cache_serialize(&onekernel_cache, &onekernel_cache_bytes),
              rocfft_status_success);
    rocfft_cleanup();
    ASSERT_TRUE(kernel_was_compiled());
    ASSERT_GT(onekernel_cache_bytes, empty_cache_bytes);

    // re-init library without blowing away cache.  rebuild plan and
    // check that the kernel was not recompiled.
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_FALSE(kernel_was_compiled());

    // blow away cache again, deserialize one-kernel cache.  re-init
    // library and rebuild plan - kernel should again not be
    // recompiled
    remove(rtc_cache_path.c_str());
    rocfft_setup();
    ASSERT_EQ(rocfft_cache_deserialize(onekernel_cache, onekernel_cache_bytes),
              rocfft_status_success);
    rocfft_cleanup();
    ASSERT_FALSE(kernel_was_compiled());

    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_FALSE(kernel_was_compiled());

    // use the cache as a system cache and make the user one an empty
    // in-memory cache.  kernel should still not be recompiled.
    EnvironmentSetTemp cache_sys_env("ROCFFT_RTC_SYS_CACHE_PATH", rtc_cache_path.c_str());
    EnvironmentSetTemp cache_empty_env("ROCFFT_RTC_CACHE_PATH", ":memory:");
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_FALSE(kernel_was_compiled());

    // check that the system cache is not written to, even if it's
    // writable by the current user.  after removing the cache, the
    // kernel should always be recompiled since rocFFT has no durable
    // place to write it to.
    remove(rtc_cache_path.c_str());
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_TRUE(kernel_was_compiled());
    rocfft_setup();
    build_plan();
    rocfft_cleanup();
    ASSERT_TRUE(kernel_was_compiled());
}

// make sure cache API functions tolerate null pointers without crashing
TEST(rocfft_UnitTest, rtc_cache_null)
{
    void*  buf     = nullptr;
    size_t buf_len = 0;
    ASSERT_EQ(rocfft_cache_serialize(nullptr, &buf_len), rocfft_status_invalid_arg_value);
    ASSERT_EQ(rocfft_cache_serialize(&buf, nullptr), rocfft_status_invalid_arg_value);
    ASSERT_EQ(rocfft_cache_buffer_free(nullptr), rocfft_status_success);
    ASSERT_EQ(rocfft_cache_deserialize(nullptr, 12345), rocfft_status_invalid_arg_value);
    ASSERT_EQ(rocfft_cache_deserialize(&buf_len, 0), rocfft_status_invalid_arg_value);
}

// make sure RTC gracefully handles a helper process that crashes
TEST(rocfft_UnitTest, rtc_helper_crash)
{
#ifdef WIN32
    char filename[MAX_PATH];
    GetModuleFileNameA(NULL, filename, MAX_PATH);
    fs::path test_exe    = filename;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash.exe");
#else
    fs::path test_exe    = program_invocation_name;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash");
#endif

    // use the crashing helper
    EnvironmentSetTemp env_helper("ROCFFT_RTC_PROCESS_HELPER", crasher_exe.string().c_str());
    // don't touch the cache, to force compilation
    EnvironmentSetTemp env_read("ROCFFT_RTC_CACHE_READ_DISABLE", "1");
    EnvironmentSetTemp env_write("ROCFFT_RTC_CACHE_WRITE_DISABLE", "1");
    // force out-of-process compile
    EnvironmentSetTemp env_process("ROCFFT_RTC_PROCESS", "2");

    rocfft_plan plan = nullptr;
    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      1,
                                      &RTC_PROBLEM_SIZE,
                                      1,
                                      nullptr));

    // alloc a complex buffer
    gpubuf_t<float2> data;
    ASSERT_EQ(data.alloc(RTC_PROBLEM_SIZE * sizeof(float2)), hipSuccess);

    std::vector<void*> ibuffers(1, static_cast<void*>(data.data()));

    ASSERT_EQ(rocfft_execute(plan, ibuffers.data(), nullptr, nullptr), rocfft_status_success);

    rocfft_plan_destroy(plan);
    plan = nullptr;

    rocfft_cleanup();
    rocfft_setup();

    // also try with forcing use of the subprocess, which is a
    // different code path from the default "try in-process, then
    // fall back to out-of-process"
    EnvironmentSetTemp env_force("ROCFFT_RTC_PROCESS", "1");

    ASSERT_TRUE(rocfft_status_success
                == rocfft_plan_create(&plan,
                                      rocfft_placement_inplace,
                                      rocfft_transform_type_complex_forward,
                                      rocfft_precision_single,
                                      1,
                                      &RTC_PROBLEM_SIZE,
                                      1,
                                      nullptr));
    ASSERT_EQ(rocfft_execute(plan, ibuffers.data(), nullptr, nullptr), rocfft_status_success);

    rocfft_plan_destroy(plan);
    plan = nullptr;
}

#endif
