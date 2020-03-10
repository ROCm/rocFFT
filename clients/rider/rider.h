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

#ifndef RIDER_H
#define RIDER_H

#include <boost/program_options.hpp>
#include <hip/hip_runtime_api.h>
#include <vector>

#include "../client_utils.h"
#include "rocfft.h"

// This is used to either wrap a HIP function call, or to explicitly check a variable
// for an error condition.  If an error occurs, we throw.
// Note: std::runtime_error does not take unicode strings as input, so only strings
// supported
inline hipError_t
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
    return res;
}

inline rocfft_status lib_V_Throw(rocfft_status      res,
                                 const std::string& msg,
                                 size_t             lineno,
                                 const std::string& fileName)
{
    if(res != rocfft_status_success)
    {
        std::stringstream tmp;
        tmp << "LIB_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
    return res;
}

#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status, _message) lib_V_Throw(_status, _message, __LINE__, __FILE__)

// Check that the input and output types are consistent.
void check_iotypes(const rocfft_result_placement place,
                   const rocfft_transform_type   transformType,
                   const rocfft_array_type       itype,
                   const rocfft_array_type       otype)
{
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
    case rocfft_array_type_real:
        break;
    default:
        throw std::runtime_error("Invalid Input array type format");
    }

    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
    case rocfft_array_type_real:
        break;
    default:
        throw std::runtime_error("Invalid Input array type format");
    }

    // Check that format choices are supported
    if(transformType != rocfft_transform_type_real_forward
       && transformType != rocfft_transform_type_real_inverse)
    {
        if(place == rocfft_placement_inplace && itype != otype)
        {
            throw std::runtime_error(
                "In-place transforms must have identical input and output types");
        }
    }

    bool okformat = true;
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
        okformat = (otype == rocfft_array_type_complex_interleaved
                    || otype == rocfft_array_type_complex_planar);
        break;
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
        okformat = otype == rocfft_array_type_real;
        break;
    case rocfft_array_type_real:
        okformat = (otype == rocfft_array_type_hermitian_interleaved
                    || otype == rocfft_array_type_hermitian_planar);
        break;
    default:
        throw std::runtime_error("Invalid Input array type format");
    }
    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
    case rocfft_array_type_real:
        break;
    default:
        okformat = false;
    }
    if(!okformat)
    {
        throw std::runtime_error("Invalid combination of Input/Output array type formats");
    }
}

// Check that the input and output types are consistent.  If they are unset, assign
// default values based on the transform type.
void check_set_iotypes(const rocfft_result_placement place,
                       const rocfft_transform_type   transformType,
                       rocfft_array_type&            itype,
                       rocfft_array_type&            otype)
{
    if(itype == rocfft_array_type_unset)
    {
        switch(transformType)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            itype = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            itype = rocfft_array_type_real;
            break;
        case rocfft_transform_type_real_inverse:
            itype = rocfft_array_type_hermitian_interleaved;
            break;
        default:
            throw std::runtime_error("Invalid transform type");
        }
    }
    if(otype == rocfft_array_type_unset)
    {
        switch(transformType)
        {
        case rocfft_transform_type_complex_forward:
        case rocfft_transform_type_complex_inverse:
            otype = rocfft_array_type_complex_interleaved;
            break;
        case rocfft_transform_type_real_forward:
            otype = rocfft_array_type_hermitian_interleaved;
            break;
        case rocfft_transform_type_real_inverse:
            otype = rocfft_array_type_real;
            break;
        default:
            throw std::runtime_error("Invalid transform type");
        }
    }

    check_iotypes(place, transformType, itype, otype);
}

// Check the input and output stride to make sure the values are valid for the transform.
// If strides are not set, load default values.
void check_set_iostride(const rocfft_result_placement place,
                        const rocfft_transform_type   transformType,
                        const std::vector<size_t>&    length,
                        const rocfft_array_type       itype,
                        const rocfft_array_type       otype,
                        std::vector<size_t>&          istride,
                        std::vector<size_t>&          ostride)
{
    if(!istride.empty() && istride.size() != length.size())
    {
        throw std::runtime_error("Transform dimension doesn't match input stride length");
    }

    if(!ostride.empty() && ostride.size() != length.size())
    {
        throw std::runtime_error("Transform dimension doesn't match output stride length");
    }

    if((transformType == rocfft_transform_type_complex_forward)
       || (transformType == rocfft_transform_type_complex_inverse))
    {
        // Complex-to-complex transform

        // User-specified strides must match for in-place transforms:
        if(place == rocfft_placement_inplace && !istride.empty() && !ostride.empty()
           && istride != ostride)
        {
            throw std::runtime_error("In-place transforms require istride == ostride");
        }

        // If the user only specified istride, use that for ostride for in-place
        // transforms.
        if(place == rocfft_placement_inplace && !istride.empty() && ostride.empty())
        {
            ostride = istride;
        }

        // If the strides are empty, we use contiguous data.
        if(istride.empty())
        {
            istride = compute_stride(length);
        }
        if(ostride.empty())
        {
            ostride = compute_stride(length);
        }
    }
    else
    {
        // Real/complex transform
        const bool forward = itype == rocfft_array_type_real;
        const bool inplace = place == rocfft_placement_inplace;

        // Length of complex data
        auto clength = length;
        clength[0]   = length[0] / 2 + 1;

        if(inplace)
        {
            // Fastest index must be contiguous.
            if(!istride.empty() && istride[0] != 1)
            {
                throw std::runtime_error(
                    "In-place real/complex transforms require contiguous input data.");
            }
            if(!ostride.empty() && ostride[0] != 1)
            {
                throw std::runtime_error(
                    "In-place real/complex transforms require contiguous output data.");
            }
            if(!istride.empty() && !ostride.empty())
            {
                for(int i = 1; i < length.size(); ++i)
                {
                    if(forward && istride[i] != 2 * ostride[i])
                    {
                        throw std::runtime_error(
                            "In-place real-to-complex transforms strides are inconsistent.");
                    }
                    if(!forward && 2 * istride[i] != ostride[i])
                    {
                        throw std::runtime_error(
                            "In-place complex-to-real transforms strides are inconsistent.");
                    }
                }
            }
        }

        if(istride.empty())
        {
            if(forward)
            {
                // real data
                istride = compute_stride(length, inplace ? clength[0] * 2 : 0);
            }
            else
            {
                // complex data
                istride = compute_stride(clength);
            }
        }

        if(ostride.empty())
        {
            if(forward)
            {
                // complex data
                ostride = compute_stride(clength);
            }
            else
            {
                // real data
                ostride = compute_stride(length, inplace ? clength[0] * 2 : 0);
            }
        }
    }
    // Final validation:
    if(istride.size() != length.size())
    {
        throw std::runtime_error("Setup failed; inconsistent istride and length.");
    }
    if(ostride.size() != length.size())
    {
        throw std::runtime_error("Setup failed; inconsistent ostride and length.");
    }
}

#endif // RIDER_H
