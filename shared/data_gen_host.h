// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DATA_GEN_HOST_H
#define DATA_GEN_HOST_H

#include "../shared/arithmetic.h"
#include "../shared/hostbuf.h"
#include "../shared/increment.h"
#include <complex>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

// element-wise addition of two ints/tuples-of-ints
template <typename Tint>
Tint element_add(Tint a, Tint b)
{
    return a + b;
}
template <typename Tint>
std::tuple<Tint, Tint> element_add(std::tuple<Tint, Tint> a, std::tuple<Tint, Tint> b)
{
    return std::make_tuple(std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b));
}

template <typename Tint>
std::tuple<Tint, Tint, Tint> element_add(std::tuple<Tint, Tint, Tint> a,
                                         std::tuple<Tint, Tint, Tint> b)
{
    return std::make_tuple(std::get<0>(a) + std::get<0>(b),
                           std::get<1>(a) + std::get<1>(b),
                           std::get<2>(a) + std::get<2>(b));
}

// Specialized computation of index given 1-, 2-, 3- dimension length + stride
template <typename T1, typename T2>
size_t compute_index(T1 length, T2 stride, size_t base)
{
    return (length * stride) + base;
}

template <typename T1, typename T2>
size_t
    compute_index(const std::tuple<T1, T1>& length, const std::tuple<T2, T2>& stride, size_t base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (std::get<0>(length) * std::get<0>(stride)) + (std::get<1>(length) * std::get<1>(stride))
           + base;
}

template <typename T1, typename T2>
size_t compute_index(const std::tuple<T1, T1, T1>& length,
                     const std::tuple<T2, T2, T2>& stride,
                     size_t                        base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (std::get<0>(length) * std::get<0>(stride)) + (std::get<1>(length) * std::get<1>(stride))
           + (std::get<2>(length) * std::get<2>(stride)) + base;
}

template <typename T1>
T1 make_unit_stride(const T1& whole_length)
{
    return static_cast<T1>(1);
}

template <typename T1>
std::tuple<T1, T1> make_unit_stride(const std::tuple<T1, T1>& whole_length)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(std::get<0>(whole_length)));
}

template <typename T1>
std::tuple<T1, T1, T1> make_unit_stride(const std::tuple<T1, T1, T1>& whole_length)
{
    return std::make_tuple(static_cast<T1>(1),
                           static_cast<T1>(std::get<0>(whole_length)),
                           static_cast<T1>(std::get<0>(whole_length))
                               * static_cast<T1>(std::get<1>(whole_length)));
}

// Work out how many partitions to break our iteration problem into
template <typename T1>
static size_t compute_partition_count(T1 length)
{
#ifdef _OPENMP
    // we seem to get contention from too many threads, which slows
    // things down.  particularly noticeable with mix_3D tests
    static const size_t MAX_PARTITIONS = 8;
    size_t              iters          = count_iters(length);
    size_t hw_threads = std::min(MAX_PARTITIONS, static_cast<size_t>(omp_get_num_procs()));
    if(!hw_threads)
        return 1;

    // don't bother threading problem sizes that are too small. pick
    // an arbitrary number of iterations and ensure that each thread
    // has at least that many iterations to process
    static const size_t MIN_ITERS_PER_THREAD = 2048;

    // either use the whole CPU, or use ceil(iters/iters_per_thread)
    return std::min(hw_threads, (iters + MIN_ITERS_PER_THREAD + 1) / MIN_ITERS_PER_THREAD);
#else
    return 1;
#endif
}

// Break a scalar length into some number of pieces, returning
// [(start0, end0), (start1, end1), ...]
template <typename T1>
std::vector<std::pair<T1, T1>> partition_base(const T1& length, size_t num_parts)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");

    // make sure we don't exceed the length
    num_parts = std::min(length, num_parts);

    std::vector<std::pair<T1, T1>> ret(num_parts);
    auto                           partition_size = length / num_parts;
    T1                             cur_partition  = 0;
    for(size_t i = 0; i < num_parts; ++i, cur_partition += partition_size)
    {
        ret[i].first  = cur_partition;
        ret[i].second = cur_partition + partition_size;
    }
    // last partition might not divide evenly, fix it up
    ret.back().second = length;
    return ret;
}

// Returns pairs of startindex, endindex, for 1D, 2D, 3D lengths
template <typename T1>
std::vector<std::pair<T1, T1>> partition_rowmajor(const T1& length)
{
    return partition_base(length, compute_partition_count(length));
}

// Partition on the leftmost part of the tuple, for row-major indexing
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>>
    partition_rowmajor(const std::tuple<T1, T1>& length)
{
    auto partitions = partition_base(std::get<0>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<0>(ret[i].first)  = partitions[i].first;
        std::get<1>(ret[i].first)  = 0;
        std::get<0>(ret[i].second) = partitions[i].second;
        std::get<1>(ret[i].second) = std::get<1>(length);
    }
    return ret;
}
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>>
    partition_rowmajor(const std::tuple<T1, T1, T1>& length)
{
    auto partitions = partition_base(std::get<0>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<0>(ret[i].first)  = partitions[i].first;
        std::get<1>(ret[i].first)  = 0;
        std::get<2>(ret[i].first)  = 0;
        std::get<0>(ret[i].second) = partitions[i].second;
        std::get<1>(ret[i].second) = std::get<1>(length);
        std::get<2>(ret[i].second) = std::get<2>(length);
    }
    return ret;
}

// For complex-to-real transforms, the input data must be Hermitiam-symmetric.
// That is, u_k is the complex conjugate of u_{-k}, where k is the wavevector in Fourier
// space.  For multi-dimensional data, this means that we only need to store a bit more
// than half of the complex values; the rest are redundant.  However, there are still
// some restrictions:
// * the origin and Nyquist value(s) must be real-valued
// * some of the remaining values are still redundant, and you might get different results
//   than you expect if the values don't agree.
// Below are some example kernels which impose Hermitian symmetry on a complex array
// of the given dimensions.

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_interleaved_1D(std::vector<hostbuf>&     vals,
                                                     const std::vector<Tsize>& length,
                                                     const std::vector<Tsize>& istride,
                                                     const Tsize               idist,
                                                     const Tsize               nbatch)
{
    for(unsigned int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        auto data = ((std::complex<Tfloat>*)vals[0].data()) + ibatch * idist;

        data[0].imag(0.0);

        if(length[0] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2)].imag(0.0);
        }
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_planar_1D(std::vector<hostbuf>&     vals,
                                                const std::vector<Tsize>& length,
                                                const std::vector<Tsize>& istride,
                                                const Tsize               idist,
                                                const Tsize               nbatch)
{
    for(unsigned int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        auto data_imag = ((Tfloat*)vals[1].data()) + ibatch * idist;

        data_imag[0] = 0.0;

        if(length[0] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2)] = 0.0;
        }
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_interleaved_2D(std::vector<hostbuf>&     vals,
                                                     const std::vector<Tsize>& length,
                                                     const std::vector<Tsize>& istride,
                                                     const Tsize               idist,
                                                     const Tsize               nbatch)
{
    for(unsigned int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        auto data = ((std::complex<Tfloat>*)vals[0].data()) + ibatch * idist;

        data[0].imag(0.0);

        if(length[0] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2)].imag(0.0);
        }

        if(length[1] % 2 == 0)
        {
            data[istride[1] * (length[1] / 2)].imag(0.0);
        }

        if(length[0] % 2 == 0 && length[1] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)].imag(0.0);
        }

        for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
        {
            data[istride[0] * (length[0] - i)] = std::conj(data[istride[0] * i]);
        }

        if(length[1] % 2 == 0)
        {
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                data[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                    = std::conj(data[istride[0] * i + istride[1] * (length[1] / 2)]);
            }
        }
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_planar_2D(std::vector<hostbuf>&     vals,
                                                const std::vector<Tsize>& length,
                                                const std::vector<Tsize>& istride,
                                                const Tsize               idist,
                                                const Tsize               nbatch)
{
    for(unsigned int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        auto data_real = ((Tfloat*)vals[0].data()) + ibatch * idist;
        auto data_imag = ((Tfloat*)vals[1].data()) + ibatch * idist;

        data_imag[0] = 0.0;

        if(length[0] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2)] = 0.0;
        }

        if(length[1] % 2 == 0)
        {
            data_imag[istride[1] * (length[1] / 2)] = 0.0;
        }

        if(length[0] % 2 == 0 && length[1] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)] = 0.0;
        }

        for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
        {
            data_real[istride[0] * (length[0] - i)] = data_real[istride[0] * i];
            data_imag[istride[0] * (length[0] - i)] = -data_imag[istride[0] * i];
        }

        if(length[1] % 2 == 0)
        {
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                data_real[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                    = data_real[istride[0] * i + istride[1] * (length[1] / 2)];
                data_imag[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                    = -data_imag[istride[0] * i + istride[1] * (length[1] / 2)];
            }
        }
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_interleaved_3D(std::vector<hostbuf>&     vals,
                                                     const std::vector<Tsize>& length,
                                                     const std::vector<Tsize>& istride,
                                                     const Tsize               idist,
                                                     const Tsize               nbatch)
{
    for(unsigned int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        auto data = ((std::complex<Tfloat>*)vals[0].data()) + ibatch * idist;

        data[0].imag(0.0);

        if(length[0] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2)].imag(0.0);
        }

        if(length[1] % 2 == 0)
        {
            data[istride[1] * (length[1] / 2)].imag(0.0);
        }

        if(length[2] % 2 == 0)
        {
            data[istride[2] * (length[2] / 2)].imag(0.0);
        }

        if(length[0] % 2 == 0 && length[1] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)].imag(0.0);
        }

        if(length[0] % 2 == 0 && length[2] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2) + istride[2] * (length[2] / 2)].imag(0.0);
        }
        if(length[1] % 2 == 0 && length[2] % 2 == 0)
        {
            data[istride[1] * (length[1] / 2) + istride[2] * (length[2] / 2)].imag(0.0);
        }

        if(length[0] % 2 == 0 && length[1] % 2 == 0 && length[2] % 2 == 0)
        {
            data[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)
                 + istride[2] * (length[2] / 2)]
                .imag(0.0);
        }

        // y-axis:
        for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
        {
            data[istride[1] * (length[1] - j)] = std::conj(data[istride[1] * j]);
        }

        if(length[0] % 2 == 0)
        {
            // y-axis at x-nyquist
            for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
            {
                data[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)]
                    = std::conj(data[istride[0] * (length[0] / 2) + istride[1] * j]);
            }
        }

        // x-axis:
        for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
        {
            data[istride[0] * (length[0] - i)] = std::conj(data[istride[0] * i]);
        }

        if(length[1] % 2 == 0)
        {
            // x-axis at y-nyquist
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                data[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                    = std::conj(data[istride[0] * i + istride[1] * (length[1] / 2)]);
            }
        }

        // x-y plane:
        for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
        {
            for(unsigned int j = 1; j < length[1]; ++j)
            {
                data[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)]
                    = std::conj(data[istride[0] * i + istride[1] * j]);
            }
        }

        if(length[2] % 2 == 0)
        {
            // x-axis at z-nyquist
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                data[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                    = std::conj(data[istride[0] * i + istride[2] * (length[2] / 2)]);
            }
            if(length[1] % 2 == 0)
            {
                // x-axis at yz-nyquist
                for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
                {
                    data[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                        = std::conj(data[istride[0] * i + istride[2] * (length[2] / 2)]);
                }
            }

            // y-axis: at z-nyquist
            for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
            {
                data[istride[1] * (length[1] - j) + istride[2] * (length[2] / 2)]
                    = std::conj(data[istride[1] * j + istride[2] * (length[2] / 2)]);
            }

            if(length[0] % 2 == 0)
            {
                // y-axis: at xz-nyquist
                for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
                {
                    data[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)
                         + istride[2] * (length[2] / 2)]
                        = std::conj(data[istride[0] * (length[0] / 2) + istride[1] * j
                                         + istride[2] * (length[2] / 2)]);
                }
            }

            // x-y plane: at z-nyquist
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                for(unsigned int j = 1; j < length[1]; ++j)
                {
                    data[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)
                         + istride[2] * (length[2] / 2)]
                        = std::conj(
                            data[istride[0] * i + istride[1] * j + istride[2] * (length[2] / 2)]);
                }
            }
        }
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_planar_3D(std::vector<hostbuf>&     vals,
                                                const std::vector<Tsize>& length,
                                                const std::vector<Tsize>& istride,
                                                const Tsize               idist,
                                                const Tsize               nbatch)
{
    for(unsigned int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        auto data_real = ((Tfloat*)vals[0].data()) + ibatch * idist;
        auto data_imag = ((Tfloat*)vals[1].data()) + ibatch * idist;

        data_imag[0] = 0.0;

        if(length[0] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2)] = 0.0;
        }

        if(length[1] % 2 == 0)
        {
            data_imag[istride[1] * (length[1] / 2)] = 0.0;
        }

        if(length[2] % 2 == 0)
        {
            data_imag[istride[2] * (length[2] / 2)] = 0.0;
        }

        if(length[0] % 2 == 0 && length[1] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)] = 0.0;
        }

        if(length[0] % 2 == 0 && length[2] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2) + istride[2] * (length[2] / 2)] = 0.0;
        }
        if(length[1] % 2 == 0 && length[2] % 2 == 0)
        {
            data_imag[istride[1] * (length[1] / 2) + istride[2] * (length[2] / 2)] = 0.0;
        }

        if(length[0] % 2 == 0 && length[1] % 2 == 0 && length[2] % 2 == 0)
        {
            data_imag[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)
                      + istride[2] * (length[2] / 2)]
                = 0.0;
        }

        // y-axis:
        for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
        {
            data_real[istride[1] * (length[1] - j)] = data_real[istride[1] * j];
            data_imag[istride[1] * (length[1] - j)] = -data_imag[istride[1] * j];
        }

        if(length[0] % 2 == 0)
        {
            // y-axis at x-nyquist
            for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
            {
                data_real[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)]
                    = data_real[istride[0] * (length[0] / 2) + istride[1] * j];
                data_imag[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)]
                    = -data_imag[istride[0] * (length[0] / 2) + istride[1] * j];
            }
        }

        // x-axis:
        for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
        {
            data_real[istride[0] * (length[0] - i)] = data_real[istride[0] * i];
            data_imag[istride[0] * (length[0] - i)] = -data_imag[istride[0] * i];
        }

        if(length[1] % 2 == 0)
        {
            // x-axis at y-nyquist
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                data_real[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                    = data_real[istride[0] * i + istride[1] * (length[1] / 2)];
                data_imag[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                    = -data_imag[istride[0] * i + istride[1] * (length[1] / 2)];
            }
        }

        // x-y plane:
        for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
        {
            for(unsigned int j = 1; j < length[1]; ++j)
            {
                data_real[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)]
                    = data_real[istride[0] * i + istride[1] * j];
                data_imag[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)]
                    = -data_imag[istride[0] * i + istride[1] * j];
            }
        }

        if(length[2] % 2 == 0)
        {
            // x-axis at z-nyquist
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                data_real[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                    = data_real[istride[0] * i + istride[2] * (length[2] / 2)];
                data_imag[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                    = -data_imag[istride[0] * i + istride[2] * (length[2] / 2)];
            }
            if(length[1] % 2 == 0)
            {
                // x-axis at yz-nyquist
                for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
                {
                    data_real[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                        = data_real[istride[0] * i + istride[2] * (length[2] / 2)];
                    data_imag[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                        = -data_imag[istride[0] * i + istride[2] * (length[2] / 2)];
                }
            }

            // y-axis: at z-nyquist
            for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
            {
                data_real[istride[1] * (length[1] - j) + istride[2] * (length[2] / 2)]
                    = data_real[istride[1] * j + istride[2] * (length[2] / 2)];
                data_imag[istride[1] * (length[1] - j) + istride[2] * (length[2] / 2)]
                    = -data_imag[istride[1] * j + istride[2] * (length[2] / 2)];
            }

            if(length[0] % 2 == 0)
            {
                // y-axis: at xz-nyquist
                for(unsigned int j = 1; j < (length[1] + 1) / 2; ++j)
                {
                    data_real[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)
                              + istride[2] * (length[2] / 2)]
                        = data_real[istride[0] * (length[0] / 2) + istride[1] * j
                                    + istride[2] * (length[2] / 2)];
                    data_imag[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)
                              + istride[2] * (length[2] / 2)]
                        = -data_imag[istride[0] * (length[0] / 2) + istride[1] * j
                                     + istride[2] * (length[2] / 2)];
                }
            }

            // x-y plane: at z-nyquist
            for(unsigned int i = 1; i < (length[0] + 1) / 2; ++i)
            {
                for(unsigned int j = 1; j < length[1]; ++j)
                {
                    data_real[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)
                              + istride[2] * (length[2] / 2)]
                        = data_real[istride[0] * i + istride[1] * j + istride[2] * (length[2] / 2)];
                    data_imag[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)
                              + istride[2] * (length[2] / 2)]
                        = -data_imag[istride[0] * i + istride[1] * j
                                     + istride[2] * (length[2] / 2)];
                }
            }
        }
    }
}

template <typename Tfloat, typename Tint1>
static void generate_random_interleaved_data(std::vector<hostbuf>& input,
                                             const Tint1&          whole_length,
                                             const Tint1&          whole_stride,
                                             const size_t          idist,
                                             const size_t          nbatch,
                                             const Tint1           field_lower,
                                             const size_t          field_lower_batch,
                                             const Tint1           field_contig_stride,
                                             const size_t          field_contig_dist)
{
    auto   idata      = (std::complex<Tfloat>*)input[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);
    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            // seed RNG with logical index in the field
            std::mt19937 gen(compute_index(element_add(index, field_lower),
                                           field_contig_stride,
                                           (b + field_lower_batch) * field_contig_dist));
            do
            {
                // brick index to write to
                auto write_idx = compute_index(index, whole_stride, i_base);

                const Tfloat               x = (Tfloat)gen() / (Tfloat)gen.max();
                const Tfloat               y = (Tfloat)gen() / (Tfloat)gen.max();
                const std::complex<Tfloat> val(x, y);
                idata[write_idx] = val;
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat, typename Tint1>
static void generate_interleaved_data(std::vector<hostbuf>& input,
                                      const Tint1&          whole_length,
                                      const Tint1&          whole_stride,
                                      const size_t          idist,
                                      const size_t          nbatch)
{
    auto   idata       = (std::complex<Tfloat>*)input[0].data();
    size_t i_base      = 0;
    auto   partitions  = partition_rowmajor(whole_length);
    auto   unit_stride = make_unit_stride(whole_length);

    const Tfloat inv_scale = 1.0 / static_cast<Tfloat>(count_iters(whole_length) - 1);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto val_xy
                    = -0.5 + static_cast<Tfloat>(compute_index(index, unit_stride, 0)) * inv_scale;

                const std::complex<Tfloat> val(val_xy, val_xy);

                const auto i = compute_index(index, whole_stride, i_base);

                idata[i] = val;
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat, typename Tint1>
static void generate_random_planar_data(std::vector<hostbuf>& input,
                                        const Tint1&          whole_length,
                                        const Tint1&          whole_stride,
                                        const size_t          idist,
                                        const size_t          nbatch,
                                        const Tint1           field_lower,
                                        const size_t          field_lower_batch,
                                        const Tint1           field_contig_stride,
                                        const size_t          field_contig_dist)
{
    auto   ireal      = (Tfloat*)input[0].data();
    auto   iimag      = (Tfloat*)input[1].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);
    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            // seed RNG with logical index in the field
            std::mt19937 gen(compute_index(element_add(index, field_lower),
                                           field_contig_stride,
                                           (b + field_lower_batch) * field_contig_dist));
            do
            {
                // brick index to write to
                auto write_idx = compute_index(index, whole_stride, i_base);

                const std::complex<Tfloat> val((Tfloat)gen() / (Tfloat)gen.max(),
                                               (Tfloat)gen() / (Tfloat)gen.max());
                ireal[write_idx] = val.real();
                iimag[write_idx] = val.imag();
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat, typename Tint1>
static void generate_planar_data(std::vector<hostbuf>& input,
                                 const Tint1&          whole_length,
                                 const Tint1&          whole_stride,
                                 const size_t          idist,
                                 const size_t          nbatch)
{

    auto   ireal       = (Tfloat*)input[0].data();
    auto   iimag       = (Tfloat*)input[1].data();
    size_t i_base      = 0;
    auto   partitions  = partition_rowmajor(whole_length);
    auto   unit_stride = make_unit_stride(whole_length);

    const Tfloat inv_scale = 1.0 / static_cast<Tfloat>(count_iters(whole_length) - 1);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto val_xy
                    = -0.5 + static_cast<Tfloat>(compute_index(index, unit_stride, 0)) * inv_scale;

                const auto i = compute_index(index, whole_stride, i_base);

                ireal[i] = val_xy;
                iimag[i] = val_xy;
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat, typename Tint1>
static void generate_random_real_data(std::vector<hostbuf>& input,
                                      const Tint1&          whole_length,
                                      const Tint1&          whole_stride,
                                      const size_t          idist,
                                      const size_t          nbatch,
                                      const Tint1           field_lower,
                                      const size_t          field_lower_batch,
                                      const Tint1           field_contig_stride,
                                      const size_t          field_contig_dist)
{
    auto   idata      = (Tfloat*)input[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);
    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;

            // seed RNG with logical index in the field
            std::mt19937 gen(compute_index(element_add(index, field_lower),
                                           field_contig_stride,
                                           (b + field_lower_batch) * field_contig_dist));
            do
            {
                // brick index to write to
                auto write_idx = compute_index(index, whole_stride, i_base);

                const Tfloat val = (Tfloat)gen() / (Tfloat)gen.max();
                idata[write_idx] = val;
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat, typename Tint1>
static void generate_real_data(std::vector<hostbuf>& input,
                               const Tint1&          whole_length,
                               const Tint1&          whole_stride,
                               const size_t          idist,
                               const size_t          nbatch)
{

    auto   idata       = (Tfloat*)input[0].data();
    size_t i_base      = 0;
    auto   partitions  = partition_rowmajor(whole_length);
    auto   unit_stride = make_unit_stride(whole_length);

    const Tfloat inv_scale = 1.0 / static_cast<Tfloat>(count_iters(whole_length) - 1);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i = compute_index(index, whole_stride, i_base);

                idata[i]
                    = -0.5 + static_cast<Tfloat>(compute_index(index, unit_stride, 0)) * inv_scale;
            } while(increment_rowmajor(index, length));
        }
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_interleaved(std::vector<hostbuf>&     vals,
                                                  const std::vector<Tsize>& length,
                                                  const std::vector<Tsize>& istride,
                                                  const Tsize               idist,
                                                  const Tsize               nbatch)
{
    switch(length.size())
    {
    case 1:
        impose_hermitian_symmetry_interleaved_1D<Tfloat>(vals, length, istride, idist, nbatch);
        break;
    case 2:
        impose_hermitian_symmetry_interleaved_2D<Tfloat>(vals, length, istride, idist, nbatch);
        break;
    case 3:
        impose_hermitian_symmetry_interleaved_3D<Tfloat>(vals, length, istride, idist, nbatch);
        break;
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }
}

template <typename Tfloat, typename Tsize>
static void impose_hermitian_symmetry_planar(std::vector<hostbuf>&     vals,
                                             const std::vector<Tsize>& length,
                                             const std::vector<Tsize>& istride,
                                             const Tsize               idist,
                                             const Tsize               nbatch)
{
    switch(length.size())
    {
    case 1:
        impose_hermitian_symmetry_planar_1D<Tfloat>(vals, length, istride, idist, nbatch);
        break;
    case 2:
        impose_hermitian_symmetry_planar_2D<Tfloat>(vals, length, istride, idist, nbatch);
        break;
    case 3:
        impose_hermitian_symmetry_planar_3D<Tfloat>(vals, length, istride, idist, nbatch);
        break;
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }
}

#endif // DATA_GEN_HOST_H
