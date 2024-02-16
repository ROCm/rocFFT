// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef FFT_HASH_H
#define FFT_HASH_H

#include "arithmetic.h"
#include "fft_params.h"
#include "hostbuf.h"

struct hash_input
{
    hash_input(const fft_params& params, const bool use_ibuffer)
    {
        precision = params.precision;

        if(use_ibuffer)
        {
            buf_length   = params.ilength();
            buffer_sizes = params.ibuffer_sizes();
            buf_stride   = params.istride;
            buf_dist     = params.idist;
            buf_size     = params.isize;
            buf_type     = params.itype;
        }
        else
        {
            buf_length   = params.olength();
            buffer_sizes = params.obuffer_sizes();
            buf_stride   = params.ostride;
            buf_dist     = params.odist;
            buf_size     = params.osize;
            buf_type     = params.otype;
        }

        length = params.length;

        nbatch = params.nbatch;

        token = params.token();
    }

    ~hash_input() {}

    fft_precision precision;

    std::vector<size_t> buf_length;
    std::vector<size_t> buffer_sizes;
    std::vector<size_t> buf_stride;
    size_t              buf_dist;
    std::vector<size_t> buf_size;
    fft_array_type      buf_type;

    std::vector<size_t> length;

    size_t nbatch;

    std::string token;
};

template <typename Tint>
struct hash_output
{
    hash_output()
        : buffer_real(static_cast<Tint>(0))
        , buffer_imag(static_cast<Tint>(0))
        , params_token(static_cast<Tint>(0))
    {
    }

    ~hash_output() {}

    Tint buffer_real;
    Tint buffer_imag;

    Tint params_token;
};

static inline double get_weight(const size_t counter, const size_t max_counter)
{
    return (static_cast<double>(counter) / static_cast<double>(max_counter));
}

template <typename Tint>
static inline void hash_value(Tint&           hash_value,
                              const size_t    counter,
                              const size_t    max_counter,
                              const _Float16& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value += std::hash<float>{}(weight * input_value);
}
template <typename Tint>
static inline void hash_value(Tint&        hash_value,
                              const size_t counter,
                              const size_t max_counter,
                              const float& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value += std::hash<float>{}(weight * input_value);
}
template <typename Tint>
static inline void hash_value(Tint&         hash_value,
                              const size_t  counter,
                              const size_t  max_counter,
                              const double& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value  = std::hash<double>{}(weight * input_value) + hash_value;
}

template <typename Tint>
static inline size_t get_max_counter(const Tint& whole_length, const size_t nbatch)
{
    return static_cast<size_t>(count_iters(whole_length) * nbatch);
}

template <typename T1>
static inline T1 get_unit_value(const T1& val)
{
    return static_cast<T1>(1);
}
template <typename T1>
static inline std::tuple<T1, T1> get_unit_value(const std::tuple<T1, T1>& val)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(1));
}
template <typename T1>
static inline std::tuple<T1, T1, T1> get_unit_value(const std::tuple<T1, T1, T1>& val)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(1), static_cast<T1>(1));
}

template <typename T1>
static inline void sum_hash_from_partitions(const std::vector<T1>& partition_hash, T1& hash_value)
{
    hash_value = 0;
    hash_value = std::accumulate(partition_hash.begin(), partition_hash.end(), hash_value);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_real_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                            const Tint1&                whole_length,
                                            const Tint1&                whole_stride,
                                            const size_t                idist,
                                            const size_t                nbatch,
                                            Tint2&                      hash_real,
                                            Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   idata      = (Tfloat*)ibuffer[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, idata[i]);
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    hash_imag = 0;
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_planar_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                              const Tint1&                whole_length,
                                              const Tint1&                whole_stride,
                                              const size_t                idist,
                                              const size_t                nbatch,
                                              Tint2&                      hash_real,
                                              Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   ireal      = (Tfloat*)ibuffer[0].data();
    auto   iimag      = (Tfloat*)ibuffer[1].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);
    std::vector<Tint2> partition_hash_imag(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, ireal[i]);
                hash_value<Tint2>(partition_hash_imag[part], counter, max_counter, iimag[i]);
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    sum_hash_from_partitions(partition_hash_imag, hash_imag);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_interleaved_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                                   const Tint1&                whole_length,
                                                   const Tint1&                whole_stride,
                                                   const size_t                idist,
                                                   const size_t                nbatch,
                                                   Tint2&                      hash_real,
                                                   Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   idata      = (std::complex<Tfloat>*)ibuffer[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);
    std::vector<Tint2> partition_hash_imag(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, idata[i].real());
                hash_value<Tint2>(partition_hash_imag[part], counter, max_counter, idata[i].imag());
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    sum_hash_from_partitions(partition_hash_imag, hash_imag);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                       const fft_array_type        itype,
                                       const Tint1&                whole_length,
                                       const Tint1&                whole_stride,
                                       const size_t                idist,
                                       const size_t                nbatch,
                                       Tint2&                      hash_real,
                                       Tint2&                      hash_imag)
{
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
        compute_interleaved_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
        compute_planar_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    case fft_array_type_real:
        compute_real_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    default:
        throw std::runtime_error("Input layout format not yet supported");
    }
}

template <typename Tfloat, typename Tint>
static inline void compute_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                       const fft_array_type        itype,
                                       const std::vector<size_t>&  length,
                                       const std::vector<size_t>&  ilength,
                                       const std::vector<size_t>&  istride,
                                       const size_t                idist,
                                       const size_t                nbatch,
                                       Tint&                       hash_real,
                                       Tint&                       hash_imag)
{
    switch(length.size())
    {
    case 1:
        compute_buffer_hash<Tfloat, size_t, Tint>(
            ibuffer, itype, ilength[0], istride[0], idist, nbatch, hash_real, hash_imag);
        break;
    case 2:
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[0], ilength[1]),
            std::make_tuple(istride[0], istride[1]),
            idist,
            nbatch,
            hash_real,
            hash_imag);
        break;
    case 3:
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[0], ilength[1], ilength[2]),
            std::make_tuple(istride[0], istride[1], istride[2]),
            idist,
            nbatch,
            hash_real,
            hash_imag);
        break;
    default:
        abort();
    }
}

template <typename Tint>
static inline void compute_buffer_hash(const std::vector<hostbuf>& buffer,
                                       const hash_input&           hash_in,
                                       hash_output<Tint>&          hash_out)
{
    auto bsizes  = hash_in.buffer_sizes;
    auto blength = hash_in.buf_length;
    auto bstride = hash_in.buf_stride;
    auto bdist   = hash_in.buf_dist;
    auto btype   = hash_in.buf_type;
    auto bsize   = hash_in.buf_size;

    switch(hash_in.precision)
    {
    case fft_precision_half:
        compute_buffer_hash<_Float16>(buffer,
                                      btype,
                                      hash_in.length,
                                      blength,
                                      bstride,
                                      bdist,
                                      hash_in.nbatch,
                                      hash_out.buffer_real,
                                      hash_out.buffer_imag);
        break;
    case fft_precision_double:
        compute_buffer_hash<double>(buffer,
                                    btype,
                                    hash_in.length,
                                    blength,
                                    bstride,
                                    bdist,
                                    hash_in.nbatch,
                                    hash_out.buffer_real,
                                    hash_out.buffer_imag);
        break;
    case fft_precision_single:
        compute_buffer_hash<float>(buffer,
                                   btype,
                                   hash_in.length,
                                   blength,
                                   bstride,
                                   bdist,
                                   hash_in.nbatch,
                                   hash_out.buffer_real,
                                   hash_out.buffer_imag);
        break;
    default:
        abort();
    }
}

template <typename Tint>
static inline void compute_hash(const std::vector<hostbuf>& buffer,
                                const hash_input&           hash_in,
                                hash_output<Tint>&          hash_out)
{
    hash_out.params_token = std::hash<std::string>{}(hash_in.token);

    compute_buffer_hash<Tint>(buffer, hash_in, hash_out);
}

#endif // FFT_HASH_H