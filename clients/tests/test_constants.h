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

#pragma once
#if !defined(TESTCONSTANTS_H)
#define TESTCONSTANTS_H

#include <stdexcept>

enum data_pattern
{
    impulse,
    sawtooth,
    value,
    erratic
};

enum
{
    dimx = 0,
    dimy = 1,
    dimz = 2
};

enum fftw_dim
{
    one_d   = 1,
    two_d   = 2,
    three_d = 3
};

const bool  use_explicit_intermediate_buffer = true;
const bool  autogenerate_intermediate_buffer = false;
const bool  pointwise_compare                = true;
const bool  root_mean_square                 = false;
extern bool comparison_type;
extern bool suppress_output;

const size_t small2  = 32;
const size_t normal2 = 1024;
const size_t large2  = 8192;
const size_t dlarge2 = 4096;

const size_t small3  = 9;
const size_t normal3 = 729;
const size_t large3  = 6561;
const size_t dlarge3 = 2187;

const size_t small5  = 25;
const size_t normal5 = 625;
const size_t large5  = 15625;
const size_t dlarge5 = 3125;

const size_t small7  = 49;
const size_t normal7 = 343;
const size_t large7  = 16807;
const size_t dlarge7 = 2401;

const size_t large_batch_size                       = 2048;
const size_t do_not_output_any_mismatches           = 0;
const size_t default_number_of_mismatches_to_output = 10;
const size_t max_dimension                          = 3;

const double magnitude_lower_limit = 1.0E-100;

extern float  tolerance;
extern double rmse_tolerance;

// extern size_t number_of_random_tests;
// extern time_t random_test_parameter_seed;
// extern bool   verbose;

void handle_exception(const std::exception& except);

template <typename T>
inline size_t MaxLength2D(size_t rad)
{
    return 0;
}

#endif
