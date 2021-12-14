// Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_PRINTBUFFER_H
#define ROCFFT_PRINTBUFFER_H

#include "increment.h"
#include "rocfft.h"
#include <algorithm>
#include <vector>

// Output a formatted general-dimensional array with given length and stride in batches
// separated by dist.
template <typename Toutput, typename T1, typename T2, typename Tsize, typename Tstream>
inline void printbuffer(const Toutput*         output,
                        const std::vector<T1>& length,
                        const std::vector<T2>& stride,
                        const Tsize            nbatch,
                        const Tsize            dist,
                        const size_t           offset,
                        Tstream&               stream)
{
    auto i_base = 0;
    for(auto b = 0; b < nbatch; b++, i_base += dist)
    {
        std::vector<int> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const int i
                = std::inner_product(index.begin(), index.end(), stride.begin(), i_base + offset);
            stream << output[i] << " ";
            for(int li = index.size(); li-- > 0;)
            {
                if(index[li] == (length[li] - 1))
                {
                    stream << "\n";
                }
                else
                {
                    break;
                }
            }
        } while(increment_rowmajor(index, length));
        stream << std::endl;
    }
}

// Print a buffer stored as a std::vector of chars.
// Template types Tint1 and Tint2 are integer types
template <typename Tint1, typename Tint2, typename Tallocator, typename Tstream = std::ostream>
inline void printbuffer(const rocfft_precision                            precision,
                        const rocfft_array_type                           type,
                        const std::vector<std::vector<char, Tallocator>>& buf,
                        const std::vector<Tint1>&                         length,
                        const std::vector<Tint2>&                         stride,
                        const size_t                                      nbatch,
                        const size_t                                      dist,
                        const std::vector<size_t>&                        offset,
                        Tstream&                                          stream = std::cout)
{
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        if(precision == rocfft_precision_double)
        {
            printbuffer((std::complex<double>*)buf[0].data(),
                        length,
                        stride,
                        nbatch,
                        dist,
                        offset[0],
                        stream);
        }
        else
        {
            printbuffer((std::complex<float>*)buf[0].data(),
                        length,
                        stride,
                        nbatch,
                        dist,
                        offset[0],
                        stream);
        }
        break;
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        if(precision == rocfft_precision_double)
        {
            printbuffer((double*)buf[0].data(), length, stride, nbatch, dist, offset[0], stream);
            printbuffer((double*)buf[1].data(), length, stride, nbatch, dist, offset[1], stream);
        }
        else
        {
            printbuffer((float*)buf[0].data(), length, stride, nbatch, dist, offset[0], stream);
            printbuffer((float*)buf[1].data(), length, stride, nbatch, dist, offset[1], stream);
        }
        break;
    case rocfft_array_type_real:
        if(precision == rocfft_precision_double)
        {
            printbuffer((double*)buf[0].data(), length, stride, nbatch, dist, offset[0], stream);
        }
        else
        {
            printbuffer((float*)buf[0].data(), length, stride, nbatch, dist, offset[0], stream);
        }
        break;
    default:
        std::cerr << "unknown array type\n";
    }
}

// Partial template specializations for printing different buffer types.
template <typename Toutput>
class buffer_printer
{
public:
    template <typename Tallocator,
              typename Tint1,
              typename Tint2,
              typename Tsize,
              typename Tstream = std::ostream>
    void print_buffer(const std::vector<std::vector<char, Tallocator>>& buf,
                      const std::vector<Tint1>&                         length,
                      const std::vector<Tint2>&                         stride,
                      const Tsize                                       nbatch,
                      const Tsize                                       dist,
                      const std::vector<size_t>&                        offset,
                      Tstream&                                          stream = std::cout){
        // Not implemented.
    };
    template <typename Tallocator>
    void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                           const std::vector<size_t>&                        size,
                           const std::vector<size_t>&                        offset){
        // Not implemented
    };
};

template <>
class buffer_printer<float>
{
    // The scalar versions might be part of a planar format.
public:
    template <typename Tallocator,
              typename Tint1,
              typename Tint2,
              typename Tsize,
              typename Tstream = std::ostream>
    static void print_buffer(const std::vector<std::vector<char, Tallocator>>& buf,
                             const std::vector<Tint1>&                         length,
                             const std::vector<Tint2>&                         stride,
                             const Tsize                                       nbatch,
                             const Tsize                                       dist,
                             const std::vector<size_t>&                        offset,
                             Tstream&                                          stream = std::cout)
    {
        for(const auto& vec : buf)
        {
            printbuffer(
                (const float*)(vec.data()), length, stride, nbatch, dist, offset[0], stream);
        }
    };
    template <typename Tallocator>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset)
    {
        for(const auto& vec : buf)
        {
            auto data = reinterpret_cast<const float*>(vec.data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
    };
};

template <>
class buffer_printer<double>
{
    // The scalar versions might be part of a planar format.
public:
    template <typename Tallocator,
              typename Tint1,
              typename Tint2,
              typename Tsize,
              typename Tstream = std::ostream>
    static void print_buffer(const std::vector<std::vector<char, Tallocator>>& buf,
                             const std::vector<Tint1>&                         length,
                             const std::vector<Tint2>&                         stride,
                             const Tsize                                       nbatch,
                             const Tsize                                       dist,
                             const std::vector<size_t>&                        offset,
                             Tstream&                                          stream = std::cout)
    {
        for(const auto& vec : buf)
        {
            printbuffer(
                (const double*)(vec.data()), length, stride, nbatch, dist, offset[0], stream);
        }
    };
    template <typename Tallocator>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset)
    {
        for(const auto& vec : buf)
        {
            auto data = reinterpret_cast<const double*>(vec.data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
    };
};

template <>
class buffer_printer<std::complex<float>>
{
public:
    template <typename Tallocator,
              typename Tint1,
              typename Tint2,
              typename Tsize,
              typename Tstream = std::ostream>
    static void print_buffer(const std::vector<std::vector<char, Tallocator>>& buf,
                             const std::vector<Tint1>&                         length,
                             const std::vector<Tint2>&                         stride,
                             const Tsize                                       nbatch,
                             const Tsize                                       dist,
                             const std::vector<size_t>&                        offset,
                             Tstream&                                          stream = std::cout)
    {
        printbuffer((const std::complex<float>*)(buf[0].data()),
                    length,
                    stride,
                    nbatch,
                    dist,
                    offset[0],
                    stream);
    };
    template <typename Tallocator>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset)
    {
        auto data = reinterpret_cast<const std::complex<float>*>(buf[0].data());
        for(size_t i = 0; i < size[0]; ++i)
            std::cout << " " << data[i];
        std::cout << std::endl;
    };
};

template <>
class buffer_printer<std::complex<double>>
{
public:
    template <typename Tallocator,
              typename Tint1,
              typename Tint2,
              typename Tsize,
              typename Tstream = std::ostream>
    static void print_buffer(const std::vector<std::vector<char, Tallocator>>& buf,
                             const std::vector<Tint1>&                         length,
                             const std::vector<Tint2>&                         stride,
                             const Tsize                                       nbatch,
                             const Tsize                                       dist,
                             const std::vector<size_t>&                        offset,
                             Tstream&                                          stream = std::cout)
    {
        printbuffer((const std::complex<double>*)(buf[0].data()),
                    length,
                    stride,
                    nbatch,
                    dist,
                    offset[0],
                    stream);
    };
    template <typename Tallocator>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset)
    {
        auto data = reinterpret_cast<const std::complex<double>*>(buf[0].data());
        for(size_t i = 0; i < size[0]; ++i)
            std::cout << " " << data[i];
        std::cout << std::endl;
    };
};

// Print the contents of a buffer stored as a std::vector of chars.  The output is flat,
// ie the entire memory range is printed as though it were a contiguous 1D array.
template <typename Tallocator>
inline void printbuffer_flat(const rocfft_precision                            precision,
                             const rocfft_array_type                           type,
                             const std::vector<std::vector<char, Tallocator>>& buf,
                             const std::vector<size_t>&                        size,
                             const std::vector<size_t>&                        offset)
{
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        if(precision == rocfft_precision_double)
        {
            auto data = reinterpret_cast<const std::complex<double>*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        else
        {
            auto data = reinterpret_cast<const std::complex<float>*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        break;
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        if(precision == rocfft_precision_double)
        {
            for(int idx = 0; idx < buf.size(); ++idx)
            {
                auto data = reinterpret_cast<const double*>(buf[idx].data());
                std::cout << "idx " << idx;
                for(size_t i = 0; i < size[idx]; ++i)
                    std::cout << " " << data[i];
                std::cout << std::endl;
            }
        }
        else
        {
            for(int idx = 0; idx < buf.size(); ++idx)
            {
                auto data = reinterpret_cast<const float*>(buf[idx].data());
                std::cout << "idx " << idx;
                for(size_t i = 0; i < size[idx]; ++i)
                    std::cout << " " << data[i];
                std::cout << std::endl;
            }
        }
        break;
    case rocfft_array_type_real:
        if(precision == rocfft_precision_double)
        {
            auto data = reinterpret_cast<const double*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        else
        {
            auto data = reinterpret_cast<const float*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        break;
    default:
        std::cout << "unknown array type\n";
    }
}

#endif
