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

#include "rocfft.h"

#include <stdexcept>
void handle_exception(const std::exception& except);

extern int    verbose;
extern size_t ramgb;

// Container class for test parameters.
class rocfft_params
{
public:
    // All parameters are row-major.
    std::vector<size_t>     length;
    std::vector<size_t>     istride;
    std::vector<size_t>     ostride;
    size_t                  nbatch;
    rocfft_precision        precision;
    rocfft_transform_type   transform_type;
    rocfft_result_placement placement;
    size_t                  idist;
    size_t                  odist;
    rocfft_array_type       itype;
    rocfft_array_type       otype;
    std::vector<size_t>     ioffset = {0, 0};
    std::vector<size_t>     ooffset = {0, 0};

    size_t isize;
    size_t osize;

    // Given an array type, return the name as a string.
    std::string array_type_name(const rocfft_array_type type) const
    {
        switch(type)
        {
        case rocfft_array_type_complex_interleaved:
            return "rocfft_array_type_complex_interleaved";
        case rocfft_array_type_complex_planar:
            return "rocfft_array_type_complex_planar";
        case rocfft_array_type_real:
            return "rocfft_array_type_real";
        case rocfft_array_type_hermitian_interleaved:
            return "rocfft_array_type_hermitian_interleaved";
        case rocfft_array_type_hermitian_planar:
            return "rocfft_array_type_hermitian_planar";
        case rocfft_array_type_unset:
            return "rocfft_array_type_unset";
        }
        return "";
    }

    // Convert to string for output.
    std::string str() const
    {
        std::stringstream ss;
        ss << "\nGPU params:\n";
        ss << "\tlength:";
        for(auto i : length)
            ss << " " << i;
        ss << "\n";
        ss << "\tistride:";
        for(auto i : istride)
            ss << " " << i;
        ss << "\n";
        ss << "\tidist: " << idist << "\n";

        ss << "\tostride:";
        for(auto i : ostride)
            ss << " " << i;
        ss << "\n";
        ss << "\todist: " << odist << "\n";

        ss << "\tbatch: " << nbatch << "\n";
        ss << "\tisize: " << isize << "\n";
        ss << "\tosize: " << osize << "\n";

        if(placement == rocfft_placement_inplace)
            ss << "\tin-place\n";
        else
            ss << "\tout-of-place\n";
        ss << "\t" << array_type_name(itype) << " -> " << array_type_name(otype) << "\n";
        if(precision == rocfft_precision_single)
            ss << "\tsingle-precision\n";
        else
            ss << "\tdouble-precision\n";

        return ss.str();
    }

    // Dimension of the transform.
    size_t dim() const
    {
        return length.size();
    }

    // Estimate the amount of host memory needed.
    size_t needed_ram() const
    {
        // Host input, output, and input copy: 3 buffers, all contiguous.
        size_t needed_ram
            = 3 * std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());

        // GPU input buffer:
        needed_ram += std::inner_product(length.begin(), length.end(), istride.begin(), 0);

        // GPU output buffer:
        needed_ram += std::inner_product(length.begin(), length.end(), ostride.begin(), 0);

        // Account for precision and data type:
        if(transform_type != rocfft_transform_type_real_forward
           || transform_type != rocfft_transform_type_real_inverse)
        {
            needed_ram *= 2;
        }
        switch(precision)
        {
        case rocfft_precision_single:
            needed_ram *= 4;
            break;
        case rocfft_precision_double:
            needed_ram *= 8;
            break;
        }

        needed_ram *= nbatch;

        if(verbose > 1)
        {
            std::cout << "required host memory (GB): " << needed_ram * 1e-9 << std::endl;
        }

        return needed_ram;
    }

    // For real/complex transofrms, we need to account for differences in intput and output
    // dimensions.
    std::vector<size_t> ilength() const
    {
        auto ilength = length;
        if(transform_type == rocfft_transform_type_real_inverse)
            ilength[dim() - 1] = ilength[dim() - 1] / 2 + 1;
        return ilength;
    }
    std::vector<size_t> olength() const
    {
        auto olength = length;
        if(transform_type == rocfft_transform_type_real_forward)
            olength[dim() - 1] = olength[dim() - 1] / 2 + 1;
        return std::move(olength);
    }

    // Column-major getters:
    std::vector<size_t> ilength_cm() const
    {
        auto ilength_cm = ilength();
        std::reverse(std::begin(ilength_cm), std::end(ilength_cm));
        return std::move(ilength_cm);
    }
    std::vector<size_t> olength_cm() const
    {
        auto olength_cm = olength();
        std::reverse(std::begin(olength_cm), std::end(olength_cm));
        return std::move(olength_cm);
    }
    std::vector<size_t> length_cm() const
    {
        auto length_cm = length;
        std::reverse(std::begin(length_cm), std::end(length_cm));
        return std::move(length_cm);
    }

    std::vector<size_t> istride_cm() const
    {
        auto istride_cm = istride;
        std::reverse(std::begin(istride_cm), std::end(istride_cm));
        return std::move(istride_cm);
    }
    std::vector<size_t> ostride_cm() const
    {
        auto ostride_cm = ostride;
        std::reverse(std::begin(ostride_cm), std::end(ostride_cm));
        return std::move(ostride_cm);
    }

    // Number of input buffers
    int nibuffer() const
    {
        return (itype == rocfft_array_type_complex_planar
                || itype == rocfft_array_type_hermitian_planar)
                   ? 2
                   : 1;
    }

    // Number of output buffers
    int nobuffer() const
    {
        return (otype == rocfft_array_type_complex_planar
                || otype == rocfft_array_type_hermitian_planar)
                   ? 2
                   : 1;
    }

    // Return true if the given GPU parameters would produce a valid transform.
    bool valid() const
    {
        // Check that in-place transforms have the same input and output stride:
        if(placement == rocfft_placement_inplace)
        {
            const auto stridesize = std::min(istride.size(), ostride.size());
            bool       samestride = true;
            for(int i = 0; i < stridesize; ++i)
            {
                if(istride[i] != ostride[i])
                    samestride = false;
            }
            if(!samestride)
            {
                // In-place transforms require identical input and output strides.
                if(verbose)
                {
                    std::cout << "istride:";
                    for(const auto& i : istride)
                        std::cout << " " << i;
                    std::cout << " ostride0:";
                    for(const auto& i : ostride)
                        std::cout << " " << i;
                    std::cout << " differ; skipped for in-place transforms: skipping test"
                              << std::endl;
                }
                // TODO: mark skipped
                return false;
            }

            if((transform_type == rocfft_transform_type_real_forward
                || transform_type == rocfft_transform_type_real_inverse)
               && (istride[0] != 1 || ostride[0] != 1))
            {
                // In-place real/complex transforms require unit strides.
                if(verbose)
                {
                    std::cout
                        << "istride[0]: " << istride[0] << " ostride[0]: " << ostride[0]
                        << " must be unitary for in-place real/complex transforms: skipping test"
                        << std::endl;
                }
                return false;
            }

            if((itype == rocfft_array_type_complex_interleaved
                && otype == rocfft_array_type_complex_planar)
               || (itype == rocfft_array_type_complex_planar
                   && otype == rocfft_array_type_complex_interleaved))
            {
                if(verbose)
                {
                    std::cout << "In-place c2c transforms require identical io types; skipped.\n";
                }
                return false;
            }
        }

        // The parameters are valid.
        return true;
    }
};

#endif
