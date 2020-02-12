/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef MISC_H
#define MISC_H

#include <algorithm>
#include <complex>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <numeric>
#include <vector>

#include "rocfft.h"

// Increment the index (column-major) for looping over arbitrary dimensional loops with
// dimensions length.
template <class T1, class T2>
bool increment_colmajor(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(int idim = 0; idim < length.size(); ++idim)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            break;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

// Output a formatted general-dimensional array with given length and stride in batches separated by
// dist.
template <class Toutput, class T1, class T2>
void printbuffer(const Toutput*        output,
                 const std::vector<T1> length,
                 const std::vector<T2> stride,
                 const size_t          nbatch,
                 const size_t          dist)
{
    for(size_t b = 0; b < nbatch; b++)
    {
        std::vector<int> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const int i = std::inner_product(index.begin(), index.end(), stride.begin(), b * dist);
            std::cout << output[i] << " ";
            for(int i = 0; i < index.size(); ++i)
            {
                if(index[i] == (length[i] - 1))
                {
                    std::cout << "\n";
                }
                else
                {
                    break;
                }
            }
        } while(increment_colmajor(index, length));
        std::cout << std::endl;
    }
}

// FIXME: document
template <class T1, class T2>
void printbuffer(const rocfft_precision                precision,
                 const rocfft_array_type               itype,
                 const std::vector<std::vector<char>>& buf,
                 const std::vector<T1>                 length,
                 const std::vector<T2>                 stride,
                 const size_t                          nbatch,
                 const size_t                          dist)
{
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        if(precision == rocfft_precision_double)
        {
            printbuffer((std::complex<double>*)buf[0].data(), length, stride, nbatch, dist);
        }
        else
        {
            printbuffer((std::complex<float>*)buf[0].data(), length, stride, nbatch, dist);
        }
        break;
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        if(precision == rocfft_precision_double)
        {
            printbuffer((double*)buf[0].data(), length, stride, nbatch, dist);
            printbuffer((double*)buf[1].data(), length, stride, nbatch, dist);
        }
        else
        {
            printbuffer((float*)buf[0].data(), length, stride, nbatch, dist);
            printbuffer((float*)buf[1].data(), length, stride, nbatch, dist);
        }
        break;
    case rocfft_array_type_real:
        if(precision == rocfft_precision_double)
        {
            printbuffer((double*)buf[0].data(), length, stride, nbatch, dist);
        }
        else
        {
            printbuffer((float*)buf[0].data(), length, stride, nbatch, dist);
        }
        break;
    default:
        std::cout << "unkown array type\n";
    }
}

// This is used with the program_options class so that the user can type an integer on the
// command line and we store into an enum varaible
template <class _Elem, class _Traits>
std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& stream,
                                               rocfft_array_type&                  atype)
{
    unsigned tmp;
    stream >> tmp;
    atype = rocfft_array_type(tmp);
    return stream;
}

// similarly for transform type
template <class _Elem, class _Traits>
std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& stream,
                                               rocfft_transform_type&              ttype)
{
    unsigned tmp;
    stream >> tmp;
    ttype = rocfft_transform_type(tmp);
    return stream;
}

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

#endif // MISC_H
