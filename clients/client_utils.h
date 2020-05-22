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

#ifndef CLIENT_UTILS_H
#define CLIENT_UTILS_H

#include <algorithm>
#include <complex>
#include <numeric>

#include "rocfft.h"
#include <hip/hip_runtime_api.h>

// This is used with the program_options class so that the user can type an integer on the
// command line and we store into an enum varaible
template <typename _Elem, typename _Traits>
std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& stream,
                                               rocfft_array_type&                  atype)
{
    unsigned tmp;
    stream >> tmp;
    atype = rocfft_array_type(tmp);
    return stream;
}

// similarly for transform type
template <typename _Elem, typename _Traits>
std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& stream,
                                               rocfft_transform_type&              ttype)
{
    unsigned tmp;
    stream >> tmp;
    ttype = rocfft_transform_type(tmp);
    return stream;
}

template <typename T1, typename T2>
static bool increment_base(T1& index, const T2& length)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    if(index < length - 1)
    {
        ++index;
        return true;
    }
    index = 0;
    return false;
}

// Increment the index (column-major) for looping over 1, 2, and 3 dimensions length.
template <typename T1, typename T2>
static bool increment_colmajor(T1& index, const T2& length)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return increment_base(index, length);
}

template <typename T1, typename T2>
static bool increment_colmajor(std::tuple<T1, T1>& index, const std::tuple<T2, T2>& length)
{
    if(increment_base(std::get<0>(index), std::get<0>(length)))
        // we incremented ok, nothing further to do
        return true;
    // otherwise, we rolled over
    return increment_base(std::get<1>(index), std::get<1>(length));
}

template <typename T1, typename T2>
static bool increment_colmajor(std::tuple<T1, T1, T1>& index, const std::tuple<T2, T2, T2>& length)
{
    if(increment_base(std::get<0>(index), std::get<0>(length)))
        // we incremented ok, nothing further to do
        return true;
    if(increment_base(std::get<1>(index), std::get<1>(length)))
        // we incremented ok, nothing further to do
        return true;
    // otherwise, we rolled over
    return increment_base(std::get<2>(index), std::get<2>(length));
}

// Increment column-major index over arbitrary dimension length
template <typename T1, typename T2>
inline bool increment_colmajor(std::vector<T1>& index, const std::vector<T2>& length)
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
            // we know we were able to increment something and didn't hit the end
            return true;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

// Increment the index (row-major) for looping over 1, 2, and 3 dimensions length.
template <typename T1, typename T2>
static bool increment_rowmajor(T1& index, const T2& length)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return increment_base(index, length);
}

template <typename T1, typename T2>
static bool increment_rowmajor(std::tuple<T1, T1>& index, const std::tuple<T2, T2>& length)
{
    if(increment_base(std::get<1>(index), std::get<1>(length)))
        // we incremented ok, nothing further to do
        return true;
    // otherwise, we rolled over
    return increment_base(std::get<0>(index), std::get<0>(length));
}

template <typename T1, typename T2>
static bool increment_rowmajor(std::tuple<T1, T1, T1>& index, const std::tuple<T2, T2, T2>& length)
{
    if(increment_base(std::get<2>(index), std::get<2>(length)))
        // we incremented ok, nothing further to do
        return true;
    if(increment_base(std::get<1>(index), std::get<1>(length)))
        // we incremented ok, nothing further to do
        return true;
    // otherwise, we rolled over
    return increment_base(std::get<0>(index), std::get<0>(length));
}

// Increment row-major index over arbitrary dimension length
template <typename T1, typename T2>
bool increment_rowmajor(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(int idim = length.size(); idim-- > 0;)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            // we know we were able to increment something and didn't hit the end
            return true;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

// specialized computation of index given 1-, 2-, 3- dimension length + stride
template <typename T1, typename T2>
int compute_index(T1 length, T2 stride, size_t base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (length * stride) + base;
}

template <typename T1, typename T2>
int compute_index(const std::tuple<T1, T1>& length, const std::tuple<T2, T2>& stride, size_t base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (std::get<0>(length) * std::get<0>(stride)) + (std::get<1>(length) * std::get<1>(stride))
           + base;
}

template <typename T1, typename T2>
int compute_index(const std::tuple<T1, T1, T1>& length,
                  const std::tuple<T2, T2, T2>& stride,
                  size_t                        base)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return (std::get<0>(length) * std::get<0>(stride)) + (std::get<1>(length) * std::get<1>(stride))
           + (std::get<2>(length) * std::get<2>(stride)) + base;
}

// Output a formatted general-dimensional array with given length and stride in batches
// separated by dist.
template <typename Toutput, typename T1, typename T2>
inline void printbuffer(const Toutput*         output,
                        const std::vector<T1>& length,
                        const std::vector<T2>& stride,
                        const size_t           nbatch,
                        const size_t           dist)
{
    size_t i_base = 0;
    for(size_t b = 0; b < nbatch; b++, i_base += dist)
    {
        std::vector<int> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const int i = std::inner_product(index.begin(), index.end(), stride.begin(), i_base);
            std::cout << output[i] << " ";
            for(int i = index.size(); i-- > 0;)
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
        } while(increment_rowmajor(index, length));
        std::cout << std::endl;
    }
}

// Print a buffer stored as a std::vector of chars.
// Template types Tint1 and Tint2 are integer types
template <typename Tint1, typename Tint2, typename Tallocator>
inline void printbuffer(const rocfft_precision                            precision,
                        const rocfft_array_type                           itype,
                        const std::vector<std::vector<char, Tallocator>>& buf,
                        const std::vector<Tint1>&                         length,
                        const std::vector<Tint2>&                         stride,
                        const size_t                                      nbatch,
                        const size_t                                      dist)
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

// Print the contents of a buffer stored as a std::vector of chars.  The output is flat,
// ie the entire memory range is printed as though it were a contiguous 1D array.
template <typename Tallocator>
inline void printbuffer_flat(const rocfft_precision                            precision,
                             const rocfft_array_type                           itype,
                             const std::vector<std::vector<char, Tallocator>>& buf,
                             const size_t                                      dist)
{
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        if(precision == rocfft_precision_double)
        {
            auto data = reinterpret_cast<const std::complex<double>*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < dist; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        else
        {
            auto data = reinterpret_cast<const std::complex<float>*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < dist; ++i)
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
                for(size_t i = 0; i < dist; ++i)
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
                for(size_t i = 0; i < dist; ++i)
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
            for(size_t i = 0; i < dist; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        else
        {
            auto data = reinterpret_cast<const float*>(buf[0].data());
            std::cout << "idx " << 0;
            for(size_t i = 0; i < dist; ++i)
                std::cout << " " << data[i];
            std::cout << std::endl;
        }
        break;
    default:
        std::cout << "unkown array type\n";
    }
}

// Given a length vector, set the rest of the strides.
// The optional argument stride0 sets the stride for the contiguous dimension.
// The optional rcpadding argument sets the stride correctly for in-place
// multi-dimensional real/complex transforms.
template <typename T1>
inline std::vector<T1> compute_stride(const std::vector<T1>& length,
                                      const int              stride0   = 1,
                                      const bool             rcpadding = false)
{
    const int       dim = length.size();
    std::vector<T1> stride(dim);
    stride[dim - 1] = stride0;
    for(int i = dim - 1; i-- > 0;)
    {
        auto lengthip1 = length[i + 1];
        if(rcpadding && i == dim - 2)
        {
            lengthip1 = 2 * (lengthip1 / 2 + 1);
        }
        stride[i] = stride[i + 1] * lengthip1;
    }
    return stride;
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input and output
// types are identical.
template <typename Tval, typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers_1to1(const Tval*  input,
                              Tval*        output,
                              const Tint1& length,
                              const size_t nbatch,
                              const Tint2& istride,
                              const size_t idist,
                              const Tint3& ostride,
                              const size_t odist)
{
    bool   idx_equals_odx = istride == ostride && idist == odist;
    size_t idx_base       = 0;
    size_t odx_base       = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
        Tint1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int idx = compute_index(index, istride, idx_base);
            const int odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
            output[odx]   = input[idx];
        } while(increment_rowmajor(index, length));
    }
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input type is
// planar and the output type is complex interleaved.
template <typename Tval, typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers_2to1(const Tval*         input0,
                              const Tval*         input1,
                              std::complex<Tval>* output,
                              const Tint1&        length,
                              const size_t        nbatch,
                              const Tint2&        istride,
                              const size_t        idist,
                              const Tint3&        ostride,
                              const size_t        odist)
{
    bool   idx_equals_odx = istride == ostride && idist == odist;
    size_t idx_base       = 0;
    size_t odx_base       = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
        Tint1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int idx = compute_index(index, istride, idx_base);
            const int odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
            output[odx]   = std::complex<Tval>(input0[idx], input1[idx]);
        } while(increment_rowmajor(index, length));
    }
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input type is
// complex interleaved and the output type is planar.
template <typename Tval, typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers_1to2(const std::complex<Tval>* input,
                              Tval*                     output0,
                              Tval*                     output1,
                              const Tint1&              length,
                              const size_t              nbatch,
                              const Tint2&              istride,
                              const size_t              idist,
                              const Tint3&              ostride,
                              const size_t              odist)
{
    bool   idx_equals_odx = istride == ostride && idist == odist;
    size_t idx_base       = 0;
    size_t odx_base       = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
        Tint1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int idx = compute_index(index, istride, idx_base);
            const int odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
            output0[odx]  = input[idx].real();
            output1[odx]  = input[idx].imag();
        } while(increment_rowmajor(index, length));
    }
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input type given
// by itype, and the output type is given by otype.
template <typename Tallocator1,
          typename Tallocator2,
          typename Tint1,
          typename Tint2,
          typename Tint3>
inline void copy_buffers(const std::vector<std::vector<char, Tallocator1>>& input,
                         std::vector<std::vector<char, Tallocator2>>&       output,
                         const Tint1&                                       length,
                         const size_t                                       nbatch,
                         const rocfft_precision                             precision,
                         const rocfft_array_type                            itype,
                         const Tint2&                                       istride,
                         const size_t                                       idist,
                         const rocfft_array_type                            otype,
                         const Tint3&                                       ostride,
                         const size_t                                       odist)
{
    if(itype == otype)
    {
        switch(itype)
        {
        case rocfft_array_type_complex_interleaved:
        case rocfft_array_type_hermitian_interleaved:
            switch(precision)
            {
            case rocfft_precision_single:
                copy_buffers_1to1(reinterpret_cast<const std::complex<float>*>(input[0].data()),
                                  reinterpret_cast<std::complex<float>*>(output[0].data()),
                                  length,
                                  nbatch,
                                  istride,
                                  idist,
                                  ostride,
                                  odist);
                break;
            case rocfft_precision_double:
                copy_buffers_1to1(reinterpret_cast<const std::complex<double>*>(input[0].data()),
                                  reinterpret_cast<std::complex<double>*>(output[0].data()),
                                  length,
                                  nbatch,
                                  istride,
                                  idist,
                                  ostride,
                                  odist);
                break;
            }
            break;
        case rocfft_array_type_real:
        case rocfft_array_type_complex_planar:
        case rocfft_array_type_hermitian_planar:
            for(int idx = 0; idx < input.size(); ++idx)
            {
                switch(precision)
                {
                case rocfft_precision_single:
                    copy_buffers_1to1(reinterpret_cast<const float*>(input[idx].data()),
                                      reinterpret_cast<float*>(output[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist,
                                      ostride,
                                      odist);
                    break;
                case rocfft_precision_double:
                    copy_buffers_1to1(reinterpret_cast<const double*>(input[idx].data()),
                                      reinterpret_cast<double*>(output[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist,
                                      ostride,
                                      odist);
                    break;
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid data type");
            break;
        }
    }
    else if((itype == rocfft_array_type_complex_interleaved
             && otype == rocfft_array_type_complex_planar)
            || (itype == rocfft_array_type_hermitian_interleaved
                && otype == rocfft_array_type_hermitian_planar))
    {
        // copy 1to2
        switch(precision)
        {
        case rocfft_precision_single:
            copy_buffers_1to2(reinterpret_cast<const std::complex<float>*>(input[0].data()),
                              reinterpret_cast<float*>(output[0].data()),
                              reinterpret_cast<float*>(output[1].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist);
            break;
        case rocfft_precision_double:
            copy_buffers_1to2(reinterpret_cast<const std::complex<double>*>(input[0].data()),
                              reinterpret_cast<double*>(output[0].data()),
                              reinterpret_cast<double*>(output[1].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist);
            break;
        }
    }
    else if((itype == rocfft_array_type_complex_planar
             && otype == rocfft_array_type_complex_interleaved)
            || (itype == rocfft_array_type_hermitian_planar
                && otype == rocfft_array_type_hermitian_interleaved))
    {
        // copy 2 to 1
        switch(precision)
        {
        case rocfft_precision_single:
            copy_buffers_2to1(reinterpret_cast<const float*>(input[0].data()),
                              reinterpret_cast<const float*>(input[1].data()),
                              reinterpret_cast<std::complex<float>*>(output[0].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist);
            break;
        case rocfft_precision_double:
            copy_buffers_2to1(reinterpret_cast<const double*>(input[0].data()),
                              reinterpret_cast<const double*>(input[1].data()),
                              reinterpret_cast<std::complex<double>*>(output[0].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist);
            break;
        }
    }
    else
    {
        throw std::runtime_error("Invalid input and output types.");
    }
}

// unroll arbitrary-dimension copy_buffers into specializations for 1-, 2-, 3-dimensions
template <typename Tallocator1,
          typename Tallocator2,
          typename Tint1,
          typename Tint2,
          typename Tint3>
inline void copy_buffers(const std::vector<std::vector<char, Tallocator1>>& input,
                         std::vector<std::vector<char, Tallocator2>>&       output,
                         const std::vector<Tint1>&                          length,
                         const size_t                                       nbatch,
                         const rocfft_precision                             precision,
                         const rocfft_array_type                            itype,
                         const std::vector<Tint2>&                          istride,
                         const size_t                                       idist,
                         const rocfft_array_type                            otype,
                         const std::vector<Tint3>&                          ostride,
                         const size_t                                       odist)
{
    switch(length.size())
    {
    case 1:
        return copy_buffers(input,
                            output,
                            length[0],
                            nbatch,
                            precision,
                            itype,
                            istride[0],
                            idist,
                            otype,
                            ostride[0],
                            odist);
    case 2:
        return copy_buffers(input,
                            output,
                            std::make_tuple(length[0], length[1]),
                            nbatch,
                            precision,
                            itype,
                            std::make_tuple(istride[0], istride[1]),
                            idist,
                            otype,
                            std::make_tuple(ostride[0], ostride[1]),
                            odist);
    case 3:
        return copy_buffers(input,
                            output,
                            std::make_tuple(length[0], length[1], length[2]),
                            nbatch,
                            precision,
                            itype,
                            std::make_tuple(istride[0], istride[1], istride[2]),
                            idist,
                            otype,
                            std::make_tuple(ostride[0], ostride[1], ostride[2]),
                            odist);
    default:
        abort();
    }
}

// Compute the L-infinity and L-2 distance between two buffers with strides istride and
// length idist between batches to a buffer with strides ostride and length odist between
// batches.  Both buffers are of complex type.
template <typename Tcomplex, typename Tint1, typename Tint2, typename Tint3>
inline std::pair<double, double> LinfL2diff_1to1_complex(const Tcomplex* input,
                                                         const Tcomplex* output,
                                                         const Tint1&    length,
                                                         const size_t    nbatch,
                                                         const Tint2&    istride,
                                                         const size_t    idist,
                                                         const Tint3&    ostride,
                                                         const size_t    odist)
{
    double linf = 0.0;
    double l2   = 0.0;

    bool   idx_equals_odx = istride == ostride && idist == odist;
    size_t idx_base       = 0;
    size_t odx_base       = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
        Tint1 index;
        memset(&index, 0, sizeof(index));

        do
        {
            const int    idx   = compute_index(index, istride, idx_base);
            const int    odx   = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
            const double rdiff = std::abs(output[odx].real() - input[idx].real());
            linf               = std::max(rdiff, linf);
            l2 += rdiff * rdiff;

            const double idiff = std::abs(output[odx].imag() - input[idx].imag());
            linf               = std::max(idiff, linf);
            l2 += idiff * idiff;

        } while(increment_colmajor(index, length));
    }
    return std::make_pair(linf, l2);
}

// Compute the L-infinity and L-2 distance between two buffers with strides istride and
// length idist between batches to a buffer with strides ostride and length odist between
// batches.  Both buffers are of real type.
template <typename Tfloat, typename Tint1, typename Tint2, typename Tint3>
inline std::pair<double, double> LinfL2diff_1to1_real(const Tfloat* input,
                                                      const Tfloat* output,
                                                      const Tint1&  length,
                                                      const size_t  nbatch,
                                                      const Tint2&  istride,
                                                      const size_t  idist,
                                                      const Tint3&  ostride,
                                                      const size_t  odist)
{
    double linf           = 0.0;
    double l2             = 0.0;
    bool   idx_equals_odx = istride == ostride && idist == odist;
    size_t idx_base       = 0;
    size_t odx_base       = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
        Tint1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int    idx  = compute_index(index, istride, idx_base);
            const int    odx  = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
            const double diff = std::abs(output[odx] - input[idx]);
            linf              = std::max(diff, linf);
            l2 += diff * diff;

        } while(increment_rowmajor(index, length));
    }
    return std::make_pair(linf, l2);
}

// Compute the L-infinity and L-2 distance between two buffers with strides istride and
// length idist between batches to a buffer with strides ostride and length odist between
// batches.  input is complex-interleaved, output is complex-planar.
template <typename Tval, typename T1, typename T2, typename T3>
inline std::pair<double, double> LinfL2diff_1to2(const std::complex<Tval>* input,
                                                 const Tval*               output0,
                                                 const Tval*               output1,
                                                 const T1&                 length,
                                                 const size_t              nbatch,
                                                 const T2&                 istride,
                                                 const size_t              idist,
                                                 const T3&                 ostride,
                                                 const size_t              odist)
{
    double linf           = 0.0;
    double l2             = 0.0;
    bool   idx_equals_odx = istride == ostride && idist == odist;
    size_t idx_base       = 0;
    size_t odx_base       = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
        T1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int    idx   = compute_index(index, istride, idx_base);
            const int    odx   = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
            const double rdiff = std::abs(output0[odx] - input[idx].real());
            linf               = std::max(rdiff, linf);
            l2 += rdiff * rdiff;

            const double idiff = std::abs(output1[odx] - input[idx].imag());
            linf               = std::max(idiff, linf);
            l2 += idiff * idiff;

        } while(increment_rowmajor(index, length));
    }
    return std::make_pair(linf, l2);
}

// Compute the L-inifnity and L-2 distance between two buffers of dimension length and
// with types given by itype, otype, and precision.
template <typename Tallocator1,
          typename Tallocator2,
          typename Tint1,
          typename Tint2,
          typename Tint3>
inline std::pair<double, double>
    LinfL2diff(const std::vector<std::vector<char, Tallocator1>>& input,
               const std::vector<std::vector<char, Tallocator2>>& output,
               const Tint1&                                       length,
               const size_t                                       nbatch,
               const rocfft_precision                             precision,
               const rocfft_array_type                            itype,
               const Tint2&                                       istride,
               const size_t                                       idist,
               const rocfft_array_type                            otype,
               const Tint3&                                       ostride,
               const size_t                                       odist)
{
    auto LinfL2 = std::make_pair<double, double>(0.0, 0.0);
    if(itype == otype)
    {
        switch(itype)
        {
        case rocfft_array_type_complex_interleaved:
        case rocfft_array_type_hermitian_interleaved:
            switch(precision)
            {
            case rocfft_precision_single:
                LinfL2 = LinfL2diff_1to1_complex(
                    reinterpret_cast<const std::complex<float>*>(input[0].data()),
                    reinterpret_cast<const std::complex<float>*>(output[0].data()),
                    length,
                    nbatch,
                    istride,
                    idist,
                    ostride,
                    odist);
                break;
            case rocfft_precision_double:
                LinfL2 = LinfL2diff_1to1_complex(
                    reinterpret_cast<const std::complex<double>*>(input[0].data()),
                    reinterpret_cast<const std::complex<double>*>(output[0].data()),
                    length,
                    nbatch,
                    istride,
                    idist,
                    ostride,
                    odist);
                break;
            }
            break;
        case rocfft_array_type_real:
        case rocfft_array_type_complex_planar:
        case rocfft_array_type_hermitian_planar:
            for(int idx = 0; idx < input.size(); ++idx)
            {
                auto val = std::make_pair<double, double>(0.0, 0.0);
                switch(precision)
                {
                case rocfft_precision_single:
                    val = LinfL2diff_1to1_real(reinterpret_cast<const float*>(input[idx].data()),
                                               reinterpret_cast<const float*>(output[idx].data()),
                                               length,
                                               nbatch,
                                               istride,
                                               idist,
                                               ostride,
                                               odist);
                    break;
                case rocfft_precision_double:
                    val = LinfL2diff_1to1_real(reinterpret_cast<const double*>(input[idx].data()),
                                               reinterpret_cast<const double*>(output[idx].data()),
                                               length,
                                               nbatch,
                                               istride,
                                               idist,
                                               ostride,
                                               odist);
                    break;
                }
                LinfL2.first = std::max(val.first, LinfL2.first);
                LinfL2.second += val.second;
            }
            break;
        default:
            throw std::runtime_error("Invalid input and output types.");
            break;
        }
    }
    else if((itype == rocfft_array_type_complex_interleaved
             && otype == rocfft_array_type_complex_planar)
            || (itype == rocfft_array_type_hermitian_interleaved
                && otype == rocfft_array_type_hermitian_planar))
    {
        switch(precision)
        {
        case rocfft_precision_single:
            LinfL2 = LinfL2diff_1to2(reinterpret_cast<const std::complex<float>*>(input[0].data()),
                                     reinterpret_cast<const float*>(output[0].data()),
                                     reinterpret_cast<const float*>(output[1].data()),
                                     length,
                                     nbatch,
                                     istride,
                                     idist,
                                     ostride,
                                     odist);
            break;
        case rocfft_precision_double:
            LinfL2 = LinfL2diff_1to2(reinterpret_cast<const std::complex<double>*>(input[0].data()),
                                     reinterpret_cast<const double*>(output[0].data()),
                                     reinterpret_cast<const double*>(output[1].data()),
                                     length,
                                     nbatch,
                                     istride,
                                     idist,
                                     ostride,
                                     odist);
            break;
        }
    }
    else if((itype == rocfft_array_type_complex_planar
             && otype == rocfft_array_type_complex_interleaved)
            || (itype == rocfft_array_type_hermitian_planar
                && otype == rocfft_array_type_hermitian_interleaved))
    {
        switch(precision)
        {
        case rocfft_precision_single:
            LinfL2 = LinfL2diff_1to2(reinterpret_cast<const std::complex<float>*>(output[0].data()),
                                     reinterpret_cast<const float*>(input[0].data()),
                                     reinterpret_cast<const float*>(input[1].data()),
                                     length,
                                     nbatch,
                                     ostride,
                                     odist,
                                     istride,
                                     idist);
            break;
        case rocfft_precision_double:
            LinfL2
                = LinfL2diff_1to2(reinterpret_cast<const std::complex<double>*>(output[0].data()),
                                  reinterpret_cast<const double*>(input[0].data()),
                                  reinterpret_cast<const double*>(input[1].data()),
                                  length,
                                  nbatch,
                                  ostride,
                                  odist,
                                  istride,
                                  idist);
            break;
        }
    }
    else
    {
        throw std::runtime_error("Invalid input and output types.");
    }
    LinfL2.second = sqrt(LinfL2.second);
    return LinfL2;
}

// unroll arbitrary-dimension LinfL2diff into specializations for 1-, 2-, 3-dimensions
template <typename Tallocator1,
          typename Tallocator2,
          typename Tint1,
          typename Tint2,
          typename Tint3>
inline std::pair<double, double>
    LinfL2diff(const std::vector<std::vector<char, Tallocator1>>& input,
               const std::vector<std::vector<char, Tallocator2>>& output,
               const std::vector<Tint1>&                          length,
               const size_t                                       nbatch,
               const rocfft_precision                             precision,
               const rocfft_array_type                            itype,
               const std::vector<Tint2>&                          istride,
               const size_t                                       idist,
               const rocfft_array_type                            otype,
               const std::vector<Tint3>&                          ostride,
               const size_t                                       odist)
{
    switch(length.size())
    {
    case 1:
        return LinfL2diff(input,
                          output,
                          length[0],
                          nbatch,
                          precision,
                          itype,
                          istride[0],
                          idist,
                          otype,
                          ostride[0],
                          odist);
    case 2:
        return LinfL2diff(input,
                          output,
                          std::make_tuple(length[0], length[1]),
                          nbatch,
                          precision,
                          itype,
                          std::make_tuple(istride[0], istride[1]),
                          idist,
                          otype,
                          std::make_tuple(ostride[0], ostride[1]),
                          odist);
    case 3:
        return LinfL2diff(input,
                          output,
                          std::make_tuple(length[0], length[1], length[2]),
                          nbatch,
                          precision,
                          itype,
                          std::make_tuple(istride[0], istride[1], istride[2]),
                          idist,
                          otype,
                          std::make_tuple(ostride[0], ostride[1], ostride[2]),
                          odist);
    default:
        abort();
    }
}

// Compute the L-infinity and L-2 norm of abuffer with strides istride and
// length idist.  Data is std::complex.
template <typename Tcomplex, typename T1, typename T2>
inline std::pair<double, double> LinfL2norm_complex(const Tcomplex* input,
                                                    const T1&       length,
                                                    const size_t    nbatch,
                                                    const T2&       istride,
                                                    const size_t    idist)
{
    double linf = 0.0;
    double l2   = 0.0;

    size_t idx_base = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist)
    {
        T1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int idx = compute_index(index, istride, idx_base);

            const double rval = std::abs(input[idx].real());
            linf              = std::max(rval, linf);
            l2 += rval * rval;

            const double ival = std::abs(input[idx].imag());
            linf              = std::max(ival, linf);
            l2 += ival * ival;

        } while(increment_rowmajor(index, length));
    }
    return std::make_pair(linf, l2);
}

// Compute the L-infinity and L-2 norm of abuffer with strides istride and
// length idist.  Data is real-valued.
template <typename Tfloat, typename T1, typename T2>
inline std::pair<double, double> LinfL2norm_real(const Tfloat* input,
                                                 const T1&     length,
                                                 const size_t  nbatch,
                                                 const T2&     istride,
                                                 const size_t  idist)
{
    double linf = 0.0;
    double l2   = 0.0;

    size_t idx_base = 0;
    for(size_t b = 0; b < nbatch; b++, idx_base += idist)
    {
        T1 index;
        memset(&index, 0, sizeof(index));
        do
        {
            const int    idx = compute_index(index, istride, idx_base);
            const double val = std::abs(input[idx]);
            linf             = std::max(val, linf);
            l2 += val * val;

        } while(increment_rowmajor(index, length));
    }
    return std::make_pair(linf, l2);
}

// Compute the L-infinity and L-2 norm of abuffer with strides istride and
// length idist.  Data format is given by precision and itype.
template <typename Tallocator1, typename T1, typename T2>
inline std::pair<double, double>
    LinfL2norm(const std::vector<std::vector<char, Tallocator1>>& input,
               const T1&                                          length,
               const size_t                                       nbatch,
               const rocfft_precision                             precision,
               const rocfft_array_type                            itype,
               const T2&                                          istride,
               const size_t                                       idist)
{
    auto LinfL2 = std::make_pair<double, double>(0.0, 0.0);

    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        switch(precision)
        {
        case rocfft_precision_single:
            LinfL2
                = LinfL2norm_complex(reinterpret_cast<const std::complex<float>*>(input[0].data()),
                                     length,
                                     nbatch,
                                     istride,
                                     idist);
            break;
        case rocfft_precision_double:
            LinfL2
                = LinfL2norm_complex(reinterpret_cast<const std::complex<double>*>(input[0].data()),
                                     length,
                                     nbatch,
                                     istride,
                                     idist);
            break;
        }
        break;
    case rocfft_array_type_real:
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        for(int idx = 0; idx < input.size(); ++idx)
        {
            auto val = std::make_pair<double, double>(0.0, 0.0);
            switch(precision)
            {
            case rocfft_precision_single:
                val = LinfL2norm_real(reinterpret_cast<const float*>(input[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist);
                break;
            case rocfft_precision_double:
                val = LinfL2norm_real(reinterpret_cast<const double*>(input[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist);
                break;
            }
            LinfL2.first = std::max(val.first, LinfL2.first);
            LinfL2.second += val.second;
        }
        break;
    default:
        throw std::runtime_error("Invalid data type");
        break;
    }

    LinfL2.second = sqrt(LinfL2.second);
    return LinfL2;
}

// unroll arbitrary-dimension LinfL2norm into specializations for 1-, 2-, 3-dimensions
template <typename Tallocator1, typename T1, typename T2>
inline std::pair<double, double>
    LinfL2norm(const std::vector<std::vector<char, Tallocator1>>& input,
               const std::vector<T1>&                             length,
               const size_t                                       nbatch,
               const rocfft_precision                             precision,
               const rocfft_array_type                            itype,
               const std::vector<T2>&                             istride,
               const size_t                                       idist)
{
    switch(length.size())
    {
    case 1:
        return LinfL2norm(input, length[0], nbatch, precision, itype, istride[0], idist);
    case 2:
        return LinfL2norm(input,
                          std::make_tuple(length[0], length[1]),
                          nbatch,
                          precision,
                          itype,
                          std::make_tuple(istride[0], istride[1]),
                          idist);
    case 3:
        return LinfL2norm(input,
                          std::make_tuple(length[0], length[1], length[2]),
                          nbatch,
                          precision,
                          itype,
                          std::make_tuple(istride[0], istride[1], istride[2]),
                          idist);
    default:
        abort();
    }
}

// Given a buffer of complex values stored in a vector of chars (or two vectors in the
// case of planar format), impose Hermitian symmetry.
// NB: length is the dimensions of the FFT, not the data layout dimensions.
template <typename Tfloat, typename Tallocator>
inline void impose_hermitian_symmetry(std::vector<std::vector<char, Tallocator>>& vals,
                                      const std::vector<size_t>&                  length,
                                      const std::vector<size_t>&                  istride,
                                      const size_t                                idist,
                                      const size_t                                nbatch)
{
    switch(vals.size())
    {
    case 1:
    {
        // Complex interleaved data
        for(int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            auto data = ((std::complex<Tfloat>*)vals[0].data()) + ibatch * idist;
            switch(length.size())
            {
            case 3:
                if(length[2] % 2 == 0)
                {
                    data[istride[2] * (length[2] / 2)].imag(0.0);
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
                    // clang format off
                    data[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)
                         + istride[2] * (length[2] / 2)]
                        .imag(0.0);
                    // clang format off
                }

                // y-axis:
                for(int j = 1; j < (length[1] + 1) / 2; ++j)
                {
                    data[istride[1] * (length[1] - j)] = std::conj(data[istride[1] * j]);
                }

                if(length[0] % 2 == 0)
                {
                    // y-axis at x-nyquist
                    for(int j = 1; j < (length[1] + 1) / 2; ++j)
                    {
                        // clang format off
                        data[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)]
                            = std::conj(data[istride[0] * (length[0] / 2) + istride[1] * j]);
                        // clang format on
                    }
                }

                // x-axis:
                for(int i = 1; i < (length[0] + 1) / 2; ++i)
                {
                    data[istride[0] * (length[0] - i)] = std::conj(data[istride[0] * i]);
                }

                if(length[1] % 2 == 0)
                {
                    // x-axis at y-nyquist
                    for(int i = 1; i < (length[0] + 1) / 2; ++i)
                    {
                        // clang format off
                        data[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                            = std::conj(data[istride[0] * i + istride[1] * (length[1] / 2)]);
                        // clang format on
                    }
                }

                // x-y plane:
                for(int i = 1; i < (length[0] + 1) / 2; ++i)
                {
                    for(int j = 1; j < length[1]; ++j)
                    {
                        // clang format off
                        data[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)]
                            = std::conj(data[istride[0] * i + istride[1] * j]);
                        // clang format on
                    }
                }

                if(length[2] % 2 == 0)
                {
                    // x-axis at z-nyquist
                    for(int i = 1; i < (length[0] + 1) / 2; ++i)
                    {
                        data[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                            = std::conj(data[istride[0] * i + istride[2] * (length[2] / 2)]);
                    }
                    if(length[1] % 2 == 0)
                    {
                        // x-axis at yz-nyquist
                        for(int i = 1; i < (length[0] + 1) / 2; ++i)
                        {
                            data[istride[0] * (length[0] - i) + istride[2] * (length[2] / 2)]
                                = std::conj(data[istride[0] * i + istride[2] * (length[2] / 2)]);
                        }
                    }

                    // y-axis: at z-nyquist
                    for(int j = 1; j < (length[1] + 1) / 2; ++j)
                    {
                        data[istride[1] * (length[1] - j) + istride[2] * (length[2] / 2)]
                            = std::conj(data[istride[1] * j + istride[2] * (length[2] / 2)]);
                    }

                    if(length[0] % 2 == 0)
                    {
                        // y-axis: at xz-nyquist
                        for(int j = 1; j < (length[1] + 1) / 2; ++j)
                        {
                            // clang format off
                            data[istride[0] * (length[0] / 2) + istride[1] * (length[1] - j)
                                 + istride[2] * (length[2] / 2)]
                                = std::conj(data[istride[0] * (length[0] / 2) + istride[1] * j
                                                 + istride[2] * (length[2] / 2)]);
                            // clang format on
                        }
                    }

                    // x-y plane: at z-nyquist
                    for(int i = 1; i < (length[0] + 1) / 2; ++i)
                    {
                        for(int j = 1; j < length[1]; ++j)
                        {
                            // clang format off
                            data[istride[0] * (length[0] - i) + istride[1] * (length[1] - j)
                                 + istride[2] * (length[2] / 2)]
                                = std::conj(data[istride[0] * i + istride[1] * j
                                                 + istride[2] * (length[2] / 2)]);
                            // clang format on
                        }
                    }
                }

                // fall-through
            case 2:
                if(length[1] % 2 == 0)
                {
                    data[istride[1] * (length[1] / 2)].imag(0.0);
                }

                if(length[0] % 2 == 0 && length[1] % 2 == 0)
                {
                    data[istride[0] * (length[0] / 2) + istride[1] * (length[1] / 2)].imag(0.0);
                }

                for(int i = 1; i < (length[0] + 1) / 2; ++i)
                {
                    data[istride[0] * (length[0] - i)] = std::conj(data[istride[0] * i]);
                }

                if(length[1] % 2 == 0)
                {
                    for(int i = 1; i < (length[0] + 1) / 2; ++i)
                    {
                        data[istride[0] * (length[0] - i) + istride[1] * (length[1] / 2)]
                            = std::conj(data[istride[0] * i + istride[1] * (length[1] / 2)]);
                    }
                }

                // fall-through

            case 1:
                data[0].imag(0.0);

                if(length[0] % 2 == 0)
                {
                    data[istride[0] * (length[0] / 2)].imag(0.0);
                }
                break;

            default:
                throw std::runtime_error("Invalid dimension for imposeHermitianSymmetry");
                break;
            }
        }
        break;
    }
    case 2:
    {
        // Complex planar data
        for(int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            auto rdata = ((Tfloat*)vals[0].data()) + ibatch * idist;
            auto idata = ((Tfloat*)vals[1].data()) + ibatch * idist;
            switch(length.size())
            {
            case 3:
                throw std::runtime_error("Not implemented");
                // FIXME: implement
            case 2:
                throw std::runtime_error("Not implemented");
                // FIXME: implement
            case 1:
                idata[0] = 0.0;
                if(length[0] % 2 == 0)
                {
                    idata[istride[0] * (length[0] / 2)] = 0.0;
                }
                break;
            default:
                throw std::runtime_error("Invalid dimension for imposeHermitianSymmetry");
                break;
            }
        }
        break;
    }
    default:
        throw std::runtime_error("Invalid data type");
        break;
    }
}

// Given an array type and transform length, strides, etc, load random floats in [0,1]
// into the input array of floats/doubles or complex floats/doubles, which is stored in a
// vector of chars (or two vectors in the case of planar format).
// lengths are the memory lengths (ie not the transform parameters)
template <typename Tfloat, typename Tallocator>
inline void set_input(std::vector<std::vector<char, Tallocator>>& input,
                      const rocfft_array_type                     itype,
                      const std::vector<size_t>&                  length,
                      const std::vector<size_t>&                  istride,
                      const size_t                                idist,
                      const size_t                                nbatch)
{
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
    {
        auto   idata  = (std::complex<Tfloat>*)input[0].data();
        size_t i_base = 0;
        for(size_t b = 0; b < nbatch; b++, i_base += idist)
        {
            std::vector<int> index(length.size());
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), i_base);
                const std::complex<Tfloat> val((Tfloat)rand() / (Tfloat)RAND_MAX,
                                               (Tfloat)rand() / (Tfloat)RAND_MAX);
                idata[i] = val;
            } while(increment_rowmajor(index, length));
        }
        break;
    }
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
    {
        auto   ireal  = (Tfloat*)input[0].data();
        auto   iimag  = (Tfloat*)input[1].data();
        size_t i_base = 0;
        for(size_t b = 0; b < nbatch; b++, i_base += idist)
        {
            std::vector<int> index(length.size());
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), i_base);
                const std::complex<Tfloat> val((Tfloat)rand() / (Tfloat)RAND_MAX,
                                               (Tfloat)rand() / (Tfloat)RAND_MAX);
                ireal[i] = val.real();
                iimag[i] = val.imag();
            } while(increment_rowmajor(index, length));
        }
        break;
    }
    case rocfft_array_type_real:
    {
        auto   idata  = (Tfloat*)input[0].data();
        size_t i_base = 0;
        for(size_t b = 0; b < nbatch; b++, i_base += idist)
        {
            std::vector<int> index(length.size());
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), i_base);
                const Tfloat val = (Tfloat)rand() / (Tfloat)RAND_MAX;
                idata[i]         = val;
            } while(increment_rowmajor(index, length));
        }
        break;
    }
    default:
        throw std::runtime_error("Input layout format not yet supported");
        break;
    }
}

// Compute the idist for a given transform based on the placeness, transform type, and
// data layout.
inline size_t set_idist(const rocfft_result_placement place,
                        const rocfft_transform_type   transformType,
                        const std::vector<size_t>&    length,
                        const std::vector<size_t>&    istride)
{
    const size_t dim   = length.size();
    size_t       idist = 0;
    if(transformType == rocfft_transform_type_real_inverse && dim == 1)
    {
        idist = (length[0] / 2 + 1) * istride[0];
    }
    else
    {
        idist = length[0] * istride[0];
    }

    // In-place 1D transforms need extra dist.
    if(transformType == rocfft_transform_type_real_forward && dim == 1
       && place == rocfft_placement_inplace)
    {
        idist = 2 * (length[0] / 2 + 1) * istride[0];
    }
    return idist;
}

// Compute the odist for a given transform based on the placeness, transform type, and
// data layout.
inline size_t set_odist(const rocfft_result_placement place,
                        const rocfft_transform_type   transformType,
                        const std::vector<size_t>&    length,
                        const std::vector<size_t>&    ostride)
{
    const size_t dim   = length.size();
    size_t       odist = 0;
    if(transformType == rocfft_transform_type_real_forward && dim == 1)
    {
        odist = (length[0] / 2 + 1) * ostride[0];
    }
    else
    {
        odist = length[0] * ostride[0];
    }
    // in-place 1D transforms need extra dist.
    if(transformType == rocfft_transform_type_real_inverse && dim == 1
       && place == rocfft_placement_inplace)
    {
        odist = 2 * (length[0] / 2 + 1) * ostride[0];
    }
    return odist;
}

// Determine the size of the data type given the precision and type.
inline size_t var_size(const rocfft_precision precision, const rocfft_array_type type)
{
    size_t var_size = 0;
    switch(precision)
    {
    case rocfft_precision_single:
        var_size = sizeof(float);
        break;
    case rocfft_precision_double:
        var_size = sizeof(double);
        break;
    }
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        var_size *= 2;
        break;
    default:
        break;
    }
    return var_size;
}

// Given a data type and precision, the distance between batches, and the batch size,
// return the required buffer size(s).
inline std::vector<size_t> buffer_sizes(const rocfft_precision  precision,
                                        const rocfft_array_type type,
                                        const size_t            dist,
                                        const size_t            nbatch)
{
    const size_t size              = nbatch * dist * var_size(precision, type);
    unsigned     number_of_buffers = 0;
    switch(type)
    {
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        number_of_buffers = 2;
        break;
    default:
        number_of_buffers = 1;
    }
    std::vector<size_t> sizes(number_of_buffers);
    for(unsigned i = 0; i < number_of_buffers; i++)
    {
        sizes[i] = size;
    }
    return sizes;
}

// Given a data type and precision, the distance between batches, and the batch size,
// allocate the required host buffer(s).
template <typename Allocator = std::allocator<char>>
inline std::vector<std::vector<char, Allocator>>
    allocate_host_buffer(const rocfft_precision     precision,
                         const rocfft_array_type    type,
                         const std::vector<size_t>& length,
                         const std::vector<size_t>& stride,
                         const size_t               dist,
                         const size_t               nbatch)
{
    const int nbuf
        = (type == rocfft_array_type_complex_planar || type == rocfft_array_type_hermitian_planar)
              ? 2
              : 1;
    std::vector<std::vector<char, Allocator>> buffers(nbuf);
    const bool   iscomplex = (type == rocfft_array_type_complex_interleaved
                            || type == rocfft_array_type_hermitian_interleaved);
    const size_t size      = dist * nbatch * var_size(precision, type);
    for(auto& i : buffers)
    {
        i.resize(size);
    }
    return buffers;
}

// Given a data type and dimensions, fill the buffer, imposing Hermitian symmetry if
// necessary.
// NB: length is the logical size of the FFT, and not necessarily the data dimensions
template <typename Allocator = std::allocator<char>>
inline std::vector<std::vector<char, Allocator>> compute_input(const rocfft_precision     precision,
                                                               const rocfft_array_type    itype,
                                                               const std::vector<size_t>& length,
                                                               const std::vector<size_t>& istride,
                                                               const size_t               idist,
                                                               const size_t               nbatch,
                                                               const bool make_contiguous = false)
{
    const size_t dim = length.size();

    auto input = allocate_host_buffer<Allocator>(precision, itype, length, istride, idist, nbatch);

    for(auto& i : input)
    {
        std::fill(i.begin(), i.end(), 0.0);
    }

    std::vector<size_t> ilength = length;
    if(itype == rocfft_array_type_hermitian_interleaved
       || itype == rocfft_array_type_hermitian_planar)
    {
        ilength[dim - 1] = length[dim - 1] / 2 + 1;
    }

    switch(precision)
    {
    case rocfft_precision_double:
        set_input<double>(input, itype, ilength, istride, idist, nbatch);
        break;
    case rocfft_precision_single:
        set_input<float>(input, itype, ilength, istride, idist, nbatch);
        break;
    }

    if(itype == rocfft_array_type_hermitian_interleaved
       || itype == rocfft_array_type_hermitian_planar)
    {
        switch(precision)
        {
        case rocfft_precision_double:
            impose_hermitian_symmetry<double>(input, length, istride, idist, nbatch);
            break;
        case rocfft_precision_single:
            impose_hermitian_symmetry<float>(input, length, istride, idist, nbatch);
            break;
        }
    }
    return input;
}

// Check that the input and output types are consistent.
inline void check_iotypes(const rocfft_result_placement place,
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
inline void check_set_iotypes(const rocfft_result_placement place,
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

#endif
