/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef RIDER_H
#define RIDER_H

#include <boost/program_options.hpp>
#include <chrono>

struct Timer
{
    std::chrono::time_point<std::chrono::system_clock> start, end;

public:
    Timer() {}

    void Start()
    {
        start = std::chrono::system_clock::now();
    }

    double Sample()
    {
        end                                   = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // Return elapsed time in seconds
        return elapsed.count();
    }
};

#endif // RIDER_H
