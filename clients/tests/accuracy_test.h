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

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include "../client_utils.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"
#include <thread>
#include <vector>

// Simple RAII class for GPU buffers.
class gpubuf
{
public:
    gpubuf()
        : buf(nullptr)
    {
    }

    ~gpubuf()
    {
        if(buf != nullptr)
        {
            hipFree(buf);
            buf = nullptr;
        }
    }

    hipError_t alloc(const size_t size)
    {
        auto ret = hipMalloc(&buf, size);
        if(ret != hipSuccess)
            buf = nullptr;
        return ret;
    }

    void* data()
    {
        return buf;
    }

private:
    // The GPU buffer
    void* buf;
};

// Compute the rocFFT transform and verify the accuracy against the provided CPU data.
// If cpu_output_thread is non-null, join on that thread before looking at cpu_output.
void rocfft_transform(const std::vector<size_t>                                  length,
                      const size_t                                               istride0,
                      const size_t                                               ostride0,
                      const size_t                                               nbatch,
                      const rocfft_precision                                     precision,
                      const rocfft_transform_type                                transformType,
                      const rocfft_array_type                                    itype,
                      const rocfft_array_type                                    otype,
                      const rocfft_result_placement                              place,
                      const std::vector<size_t>&                                 cpu_istride,
                      const std::vector<size_t>&                                 cpu_ostride,
                      const size_t                                               cpu_idist,
                      const size_t                                               cpu_odist,
                      const rocfft_array_type                                    cpu_itype,
                      const rocfft_array_type                                    cpu_otype,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_input_copy,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_output,
                      const std::pair<double, double>& cpu_output_L2Linfnorm,
                      std::thread*                     cpu_output_thread = nullptr);

// Print the test parameters
inline void print_params(const std::vector<size_t>&    length,
                         const size_t                  istride0,
                         const size_t                  ostride0,
                         const size_t                  nbatch,
                         const rocfft_result_placement place,
                         const rocfft_precision        precision,
                         const rocfft_transform_type   transformType,
                         const rocfft_array_type       itype,
                         const rocfft_array_type       otype)
{
    std::cout << "length:";
    for(const auto& i : length)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "istride0: " << istride0 << "\n";
    std::cout << "ostride0: " << ostride0 << "\n";
    std::cout << "nbatch: " << nbatch << "\n";
    if(place == rocfft_placement_inplace)
        std::cout << "in-place\n";
    else
        std::cout << "out-of-place\n";
    if(precision == rocfft_precision_single)
        std::cout << "single-precision\n";
    else
        std::cout << "double-precision\n";
    switch(transformType)
    {
    case rocfft_transform_type_complex_forward:
        std::cout << "complex forward:\t";
        break;
    case rocfft_transform_type_complex_inverse:
        std::cout << "complex inverse:\t";
        break;
    case rocfft_transform_type_real_forward:
        std::cout << "real forward:\t";
        break;
    case rocfft_transform_type_real_inverse:
        std::cout << "real inverse:\t";
        break;
    }
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "rocfft_array_type_complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "rocfft_array_type_complex_planar";
        break;
    case rocfft_array_type_real:
        std::cout << "rocfft_array_type_real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "rocfft_array_type_hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "rocfft_array_type_hermitian_planar";
        break;
    case rocfft_array_type_unset:
        std::cout << "rocfft_array_type_unset";
        break;
    }
    std::cout << " -> ";
    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "rocfft_array_type_complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "rocfft_array_type_complex_planar";
        break;
    case rocfft_array_type_real:
        std::cout << "rocfft_array_type_real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "rocfft_array_type_hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "rocfft_array_type_hermitian_planar";
        break;
    case rocfft_array_type_unset:
        std::cout << "rocfft_array_type_unset";
        break;
    }
    std::cout << std::endl;
}

// Base gtest class for comparison with FFTW.
class accuracy_test
    : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, // length
                                                 std::vector<size_t>, // istride
                                                 std::vector<size_t>, // ostride
                                                 std::vector<size_t>, // batch
                                                 rocfft_precision,
                                                 rocfft_transform_type,
                                                 std::vector<rocfft_result_placement>>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

const static std::vector<size_t> batch_range = {1, 2};

const static std::vector<rocfft_precision> precision_range
    = {rocfft_precision_single, rocfft_precision_double};
const static std::vector<rocfft_result_placement> place_range
    = {rocfft_placement_inplace, rocfft_placement_notinplace};

// Given a vector of vector of lengths, generate all permutations.
inline std::vector<std::vector<size_t>>
    generate_lengths(const std::vector<std::vector<size_t>>& inlengths)
{
    std::vector<std::vector<size_t>> output;
    if(inlengths.size() == 0)
    {
        return output;
    }
    const size_t        dim = inlengths.size();
    std::vector<size_t> looplength(dim);
    for(int i = 0; i < dim; ++i)
    {
        looplength[i] = inlengths[i].size();
    }
    for(int idx = 0; idx < inlengths.size(); ++idx)
    {
        std::vector<size_t> index(dim);
        do
        {
            std::vector<size_t> length(dim);
            for(int i = 0; i < dim; ++i)
            {
                length[i] = inlengths[i][index[i]];
            }
            output.push_back(length);
        } while(increment_colmajor(index, looplength));
    }
    return output;
}

// Return the valid rocFFT input and output types for a given transform type.
inline std::vector<std::pair<rocfft_array_type, rocfft_array_type>>
    iotypes(const rocfft_transform_type transformType, const rocfft_result_placement place)
{
    std::vector<std::pair<rocfft_array_type, rocfft_array_type>> iotypes;
    switch(transformType)
    {
    case rocfft_transform_type_complex_forward:
    case rocfft_transform_type_complex_inverse:
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved));
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_complex_planar, rocfft_array_type_complex_planar));
        if(place == rocfft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_complex_planar, rocfft_array_type_complex_interleaved));
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_complex_interleaved, rocfft_array_type_complex_planar));
        }
        break;
    case rocfft_transform_type_real_forward:
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_real, rocfft_array_type_hermitian_interleaved));
        if(place == rocfft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_real, rocfft_array_type_hermitian_planar));
        }
        break;
    case rocfft_transform_type_real_inverse:
        iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
            rocfft_array_type_hermitian_interleaved, rocfft_array_type_real));
        if(place == rocfft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<rocfft_array_type, rocfft_array_type>(
                rocfft_array_type_hermitian_planar, rocfft_array_type_real));
        }
        break;
    default:
        throw std::runtime_error("Invalid transform type");
    }
    return iotypes;
}

#endif
