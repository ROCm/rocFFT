/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef RIDER_H
#define RIDER_H

//	Boost headers that we want to use
//	#define BOOST_PROGRAM_OPTIONS_DYN_LINK
#include <boost/program_options.hpp>

#ifdef WIN32

struct Timer
{
    LARGE_INTEGER start, stop, freq;

public:
    Timer()
    {
        QueryPerformanceFrequency(&freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&start);
    }
    double Sample()
    {
        QueryPerformanceCounter(&stop);
        double time = (double)(stop.QuadPart - start.QuadPart) / (double)(freq.QuadPart);
        return time;
    }
};

#elif defined(__APPLE__) || defined(__MACOSX)

#include <mach/mach.h>

#include <mach/clock.h>

struct Timer
{
    clock_serv_t    clock;
    mach_timespec_t start, end;

public:
    Timer()
    {
        host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &clock);
    }
    ~Timer()
    {
        mach_port_deallocate(mach_task_self(), clock);
    }

    void Start()
    {
        clock_get_time(clock, &start);
    }
    double Sample()
    {
        clock_get_time(clock, &end);
        double time = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        return time * 1E-9;
    }
};

#else

#include <time.h>

#include <cmath>

struct Timer
{
    struct timespec start, end;

public:
    Timer() {}

    void Start()
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
    }
    double Sample()
    {
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        return time * 1E-9;
    }
};

#endif

#endif // RIDER_H
