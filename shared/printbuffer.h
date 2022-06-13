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

#ifndef PRINTBUFFER_H
#define PRINTBUFFER_H

#include "increment.h"
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
    for(unsigned int b = 0; b < nbatch; b++, i_base += dist)
    {
        std::vector<size_t> index(length.size());
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
                      Tstream&                                          stream = std::cout)
    {
        throw std::runtime_error("base class for buffer_printer print_buffer not implemented.");
    };
    template <typename Tallocator, typename Tstream = std::ostream>
    void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                           const std::vector<size_t>&                        size,
                           const std::vector<size_t>&                        offset)
    {
        throw std::runtime_error(
            "base class for buffer_printer print_buffer_flat not implemented.");
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
    template <typename Tallocator, typename Tstream = std::ostream>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset,
                                  Tstream& stream = std::cout)
    {
        for(const auto& vec : buf)
        {
            auto data = reinterpret_cast<const float*>(vec.data());
            stream << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                stream << " " << data[i];
            stream << std::endl;
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
    template <typename Tallocator, typename Tstream = std::ostream>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset,
                                  Tstream& stream = std::cout)
    {
        for(const auto& vec : buf)
        {
            auto data = reinterpret_cast<const double*>(vec.data());
            stream << "idx " << 0;
            for(size_t i = 0; i < size[0]; ++i)
                stream << " " << data[i];
            stream << std::endl;
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
    template <typename Tallocator, typename Tstream = std::ostream>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset,
                                  Tstream& stream = std::cout)
    {
        auto data = reinterpret_cast<const std::complex<float>*>(buf[0].data());
        for(size_t i = 0; i < size[0]; ++i)
            stream << " " << data[i];
        stream << std::endl;
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
    template <typename Tallocator, typename Tstream = std::ostream>
    static void print_buffer_flat(const std::vector<std::vector<char, Tallocator>>& buf,
                                  const std::vector<size_t>&                        size,
                                  const std::vector<size_t>&                        offset,
                                  Tstream& stream = std::cout)
    {
        auto data = reinterpret_cast<const std::complex<double>*>(buf[0].data());
        for(size_t i = 0; i < size[0]; ++i)
            stream << " " << data[i];
        stream << std::endl;
    };
};

#endif
