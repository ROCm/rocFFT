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

#ifndef _ROCFFT_LOGGING_H_
#define _ROCFFT_LOGGING_H_

#include "rocfft_ostream.hpp"
#include "tuple_helper.hpp"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "rocfft.h"

/************************************************************************************
 * Profile kernel arguments
 ************************************************************************************/
template <typename TUP>
class argument_profile
{
    // Output stream
    rocfft_ostream& os;

    // Mutex for multithreaded access to table
    std::shared_timed_mutex mutex;

    // Table mapping argument tuples into atomic counts
    std::unordered_map<TUP,
                       std::atomic_size_t*,
                       typename tuple_helper::hash_t<TUP>,
                       typename tuple_helper::equal_t<TUP>>
        map;

public:
    // A tuple of arguments is looked up in an unordered map.
    // A count of the number of calls with these arguments is kept.
    // arg is assumed to be an rvalue for efficiency
    void operator()(TUP&& arg)
    {
        decltype(map.end()) p;
        {
            // Acquire a shared lock for reading map
            std::shared_lock<std::shared_timed_mutex> lock(mutex);

            // Look up the tuple in the map
            p = map.find(arg);

            // If tuple already exists, atomically increment count and return
            if(p != map.end())
            {
                ++*p->second;
                return;
            }
        } // Release shared lock

        // Acquire an exclusive lock for modifying map
        std::lock_guard<std::shared_timed_mutex> lock(mutex);

        // If doesn't already exist, insert tuple by moving
        bool inserted;
        std::tie(p, inserted) = map.emplace(std::move(arg), nullptr);

        // If new entry inserted, replace nullptr with new value
        // If tuple already exists, atomically increment count
        if(inserted)
            p->second = new std::atomic_size_t{1};
        else
            ++*p->second;
    }

    // Constructor
    explicit argument_profile(rocfft_ostream& os)
        : os(os)
    {
    }

    // Cleanup handler which dumps profile at destruction
    ~argument_profile()
    try
    {
        // Print all of the tuples in the map
        for(auto& p : map)
        {
            os << "- ";
            tuple_helper::print_tuple_pairs(
                os, std::tuple_cat(p.first, std::make_tuple("call_count", p.second->load())));
            os << "\n";
            delete p.second;
        }
        os.flush();
    }
    catch(...)
    {
        return;
    }
};

extern int log_trace_fd;
extern int log_bench_fd;
extern int log_profile_fd;

class LogSingleton
{
public:
    static LogSingleton& GetInstance()
    {
        static LogSingleton instance;
        return instance;
    }

private:
    LogSingleton() {}

    rocfft_layer_mode layer_mode;

    LogSingleton(LogSingleton const&);
    void operator=(LogSingleton const&);

public:
    void SetLayerMode(rocfft_layer_mode mode)
    {
        layer_mode = mode;
    }
    rocfft_layer_mode const GetLayerMode()
    {
        return layer_mode;
    }
    rocfft_ostream* GetTraceOS()
    {
        if(log_trace_fd == -1)
            return &rocfft_cerr;
        static thread_local rocfft_ostream log_trace_os(log_trace_fd);
        return &log_trace_os;
    }
    rocfft_ostream* GetBenchOS()
    {
        if(log_bench_fd == -1)
            return &rocfft_cerr;
        static thread_local rocfft_ostream log_bench_os(log_bench_fd);
        return &log_bench_os;
    }
    rocfft_ostream* GetProfileOS()
    {
        if(log_profile_fd == -1)
            return &rocfft_cerr;
        static thread_local rocfft_ostream log_profile_os(log_profile_fd);
        return &log_profile_os;
    }
};

#define LOG_TRACE_ENABLED() \
    (LogSingleton::GetInstance().GetLayerMode() & rocfft_layer_mode_log_trace)
#define LOG_BENCH_ENABLED() \
    (LogSingleton::GetInstance().GetLayerMode() & rocfft_layer_mode_log_bench)
#define LOG_PROFILE_ENABLED() \
    (LogSingleton::GetInstance().GetLayerMode() & rocfft_layer_mode_log_profile)

// if profile logging is turned on with
// (layer_mode & rocfft_layer_mode_log_profile) != 0
// log_profile will call argument_profile to profile actual arguments,
// keeping count of the number of times each set of arguments is used
template <typename... Ts>
inline void log_profile(const char* func, Ts&&... xs)
{
    if(!LOG_PROFILE_ENABLED())
    {
        // Make a tuple with the arguments
        auto tup = std::make_tuple("rocfft_function", func, xs...);

        // Set up profile
        static argument_profile<decltype(tup)> profile(*LogSingleton::GetInstance().GetProfileOS());

        // Add at_quick_exit handler in case the program exits early
        static int aqe = at_quick_exit([] { profile.~argument_profile(); });

        // Profile the tuple
        profile(std::move(tup));
    }
}

/********************************************
 * Log values (for log_trace and log_bench) *
 ********************************************/
template <typename H, typename... Ts>
static inline void log_arguments(rocfft_ostream& os, const char* sep, H head, Ts&&... xs)
{
    os << head;
    // TODO: Replace with C++17 fold expression
    // ((os << sep << std::forward<Ts>(xs)), ...);
    (void)(int[]){(os << sep << std::forward<Ts>(xs), 0)...};
    os << std::endl;
}

// if trace logging is turned on with
// (layer_mode & rocbfft_layer_mode_log_trace) != 0
// log_function will call log_arguments to log arguments with a comma separator
template <typename... Ts>
inline void log_trace(Ts&&... xs)
{
    if(LOG_TRACE_ENABLED())
        log_arguments(*LogSingleton::GetInstance().GetTraceOS(), ",", std::forward<Ts>(xs)...);
}

// if bench logging is turned on with
// (layer_mode & rocfft_layer_mode_log_bench) != 0
// log_bench will call log_arguments to log a string that
// can be input to the executable rocfft-rider.
template <typename... Ts>
inline void log_bench(Ts&&... xs)
{
    if(LOG_BENCH_ENABLED())
        log_arguments(*LogSingleton::GetInstance().GetBenchOS(), " ", std::forward<Ts>(xs)...);
}

#endif
