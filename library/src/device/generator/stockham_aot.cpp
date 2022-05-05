// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "stockham_gen.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// parse comma-separated string uints
std::vector<unsigned int> parse_uints_csv(const std::string& arg)
{
    std::vector<unsigned int> uints;

    size_t prev_pos = 0;
    for(;;)
    {
        auto pos = arg.find(',', prev_pos);
        if(pos == std::string::npos)
        {
            uints.push_back(std::stoi(arg.substr(prev_pos)));
            break;
        }
        uints.push_back(std::stoi(arg.substr(prev_pos, pos - prev_pos)));
        prev_pos = pos + 1;
    }
    return uints;
}

int main(int argc, char** _argv)
{
    // convert argv to a vector of strings, for convenience
    std::vector<std::string> argv;
    // start with argv[1] since we don't need the program's name
    for(char** p = _argv + 1; p != _argv + argc; ++p)
    {
        argv.push_back(*p);
    }

    // expected args:
    // factors1d <factors2d> precisions threads_per_transform workgroup_size half_lds direct_to_from_reg scheme output_filename
    //
    // factors1d, factors2d, precisions and threads_per_transform are
    // comma-separated values, factors2d is only present for
    // 2D_SINGLE kernels
    //
    // precisions is a vector containing enums of rocfft_precision (represented by u-int)

    // work backwards from the end
    auto arg = argv.rbegin();

    std::string output_filename = *arg;

    ++arg;
    std::string scheme = *arg;

    ++arg;
    bool direct_to_from_reg = *arg == "1";

    ++arg;
    bool half_lds = *arg == "1";

    ++arg;
    unsigned int workgroup_size = std::stoi(*arg);

    ++arg;
    std::vector<unsigned int> threads_per_transform;
    threads_per_transform = parse_uints_csv(*arg);

    ++arg;
    std::vector<unsigned int> precisions;
    precisions = parse_uints_csv(*arg);

    std::vector<unsigned int> factors;
    std::vector<unsigned int> factors2d;
    if(scheme == "CS_KERNEL_2D_SINGLE")
    {
        ++arg;
        factors2d = parse_uints_csv(*arg);
    }

    ++arg;
    factors = parse_uints_csv(*arg);

    StockhamGeneratorSpecs specs(factors, factors2d, precisions, workgroup_size, scheme);
    specs.half_lds           = half_lds;
    specs.direct_to_from_reg = direct_to_from_reg;

    specs.threads_per_transform = threads_per_transform.front();

    // second dimension for 2D_SINGLE
    StockhamGeneratorSpecs specs2d(factors2d, factors, precisions, workgroup_size, scheme);
    if(!threads_per_transform.empty())
        specs2d.threads_per_transform = threads_per_transform.back();

    try
    {
        std::ofstream output_file(output_filename.c_str());
        output_file << stockham_variants(output_filename, specs, specs2d);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
