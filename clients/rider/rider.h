/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef RIDER_H
#define RIDER_H

#include <boost/program_options.hpp>
#include <hip/hip_runtime_api.h>
#include <vector>

#include "misc.h"
#include "rocfft.h"

// Given a length vector, set the rest of the strides.
// The optinal argument stride1 is useful for setting contiguous-ish data for in-place
// real/complex transforms.
template <class T>
std::vector<T> contiguous_stride(const std::vector<T>& length, const int stride1 = 0)
{
    std::vector<T> stride;
    stride.push_back(1);
    if(stride1 == 0)
    {
        for(auto i = 1; i < length.size(); ++i)
        {
            stride.push_back(stride[i - 1] * length[i - 1]);
        }
    }
    else
    {
        if(length.size() > 1)
        {
            stride.push_back(stride1);
        }
        for(auto i = 2; i < length.size(); ++i)
        {
            stride.push_back(stride[i - 1] * length[i - 1]);
        }
    }
    return stride;
}

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
            istride = contiguous_stride(length);
        }
        if(ostride.empty())
        {
            ostride = contiguous_stride(length);
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
                istride = contiguous_stride(length, inplace ? clength[0] * 2 : 0);
            }
            else
            {
                // complex data
                istride = contiguous_stride(clength);
            }
        }

        if(ostride.empty())
        {
            if(forward)
            {
                // complex data
                ostride = contiguous_stride(clength);
            }
            else
            {
                // real data
                ostride = contiguous_stride(length, inplace ? clength[0] * 2 : 0);
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

// Set the input and output distance for batched transforms, if not already set.
void set_iodist(const rocfft_result_placement place,
                const rocfft_transform_type   transformType,
                const std::vector<size_t>&    length,
                const std::vector<size_t>&    istride,
                const std::vector<size_t>&    ostride,
                size_t&                       idist,
                size_t&                       odist)
{
    if(idist == 0)
    {
        if(transformType == rocfft_transform_type_real_inverse && length.size() == 1)
        {
            idist = length[0] / 2 + 1;
        }
        else
        {
            idist = length[length.size() - 1] * istride[istride.size() - 1];
        }

        // in-place 1D transforms need extra dist.
        if(transformType == rocfft_transform_type_real_forward && length.size() == 1
           && place == rocfft_placement_inplace)
        {
            idist += 2;
        }
    }

    if(odist == 0)
    {
        if(transformType == rocfft_transform_type_real_forward && length.size() == 1)
        {
            odist = length[0] / 2 + 1;
        }
        else
        {
            odist = length[length.size() - 1] * ostride[ostride.size() - 1];
        }

        // in-place 1D transforms need extra dist.
        if(transformType == rocfft_transform_type_real_inverse && length.size() == 1
           && place == rocfft_placement_inplace)
        {
            odist += 2;
        }
    }
}

// Given a data type, a dist, and a batch size, return the required device buffer size.
template <class Tfloat>
size_t bufsize(const rocfft_array_type type, const size_t dist, const size_t nbatch)
{
    size_t size = dist * nbatch;
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        size *= sizeof(std::complex<Tfloat>);
        break;
    default:
        size *= sizeof(Tfloat);
    }
    return size;
}

// Given a data type and precision, the distance between batches, and the batch size,
// allocate the required device buffer(s).
std::vector<void*> alloc_buffer(const rocfft_precision  precision,
                                const rocfft_array_type type,
                                const size_t            dist,
                                const size_t            nbatch)
{
    size_t size = precision == rocfft_precision_double ? bufsize<double>(type, dist, nbatch)
                                                       : bufsize<float>(type, dist, nbatch);

    unsigned number_of_buffers = 0;
    switch(type)
    {
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        number_of_buffers = 2;
        break;
    default:
        number_of_buffers = 1;
    }

    std::vector<void*> buffer;
    for(unsigned i = 0; i < number_of_buffers; i++)
    {
        buffer.push_back(NULL);
        HIP_V_THROW(hipMalloc(&buffer[i], size), "hipMalloc failed");
    }

    return buffer;
}

// Given a buffer of complex values stored in a vector of chars (or two vectors in the
// case of planar format), impose Hermitian symmetry.
// NB: length is the dimensions of the FFT, not the data layout dimensions.
template <class Tfloat>
void impose_hermitian_symmetry(std::vector<std::vector<char>>& vals,
                               const std::vector<size_t>&      length,
                               const std::vector<size_t>&      istride,
                               const size_t                    idist,
                               const size_t                    nbatch)
{

    // NB: the fall-through algorithm only works with row-major data, so we reverse the
    // indices here.
    std::vector<size_t> rlength(length.size());
    std::vector<size_t> ristride(istride.size());
    for(int i = 0; i < length.size(); ++i)
    {
        rlength[i]  = length[length.size() - i - 1];
        ristride[i] = istride[istride.size() - i - 1];
    }

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
                if(rlength[2] % 2 == 0)
                {
                    data[ristride[2] * (rlength[2] / 2)].imag(0.0);
                }
                if(rlength[0] % 2 == 0 && rlength[2] % 2 == 0)
                {
                    data[ristride[0] * (rlength[0] / 2) + ristride[2] * (rlength[2] / 2)].imag(0.0);
                }
                if(rlength[1] % 2 == 0 && rlength[2] % 2 == 0)
                {
                    data[ristride[1] * (rlength[1] / 2) + ristride[2] * (rlength[2] / 2)].imag(0.0);
                }

                if(rlength[0] % 2 == 0 && rlength[1] % 2 == 0 && rlength[2] % 2 == 0)
                {
                    data[ristride[0] * (rlength[0] / 2) + ristride[1] * (rlength[1] / 2)
                         + ristride[2] * (rlength[2] / 2)]
                        .imag(0.0);
                }

                // y-axis:
                for(int j = 1; j < (rlength[1] + 1) / 2; ++j)
                {
                    data[ristride[1] * (rlength[1] - j)] = std::conj(data[ristride[1] * j]);
                }

                if(rlength[1] % 2 == 0)
                {
                    // x-axis:
                    for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                    {
                        data[ristride[0] * (rlength[0] - i) + ristride[1] * (rlength[1] / 2)]
                            = std::conj(data[ristride[0] * i + ristride[1] * (rlength[1] / 2)]);
                    }
                }

                if(rlength[0] % 2 == 0)
                {
                    for(int j = 1; j < (rlength[1] + 1) / 2; ++j)
                    {
                        data[ristride[1] * (rlength[1] - j)] = std::conj(data[ristride[1] * j]);
                    }
                }

                if(rlength[1] % 2 == 0)
                {
                    for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                    {
                        data[ristride[0] * (rlength[0] - i)] = std::conj(data[ristride[0] * i]);
                    }
                }

                // x-y plane:
                for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                {
                    for(int j = 1; j < rlength[1]; ++j)
                    {
                        data[ristride[0] * (rlength[0] - i) + ristride[1] * (rlength[1] - j)]
                            = std::conj(data[ristride[0] * i + ristride[1] * j]);
                    }
                }

                if(rlength[2] % 2 == 0)
                {
                    // x-axis:
                    for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                    {
                        data[ristride[0] * (rlength[0] - i) + ristride[2] * (rlength[2] / 2)]
                            = std::conj(data[ristride[0] * i + ristride[2] * (rlength[2] / 2)]);
                    }

                    // y-axis:
                    for(int j = 1; j < (length[1] + 1) / 2; ++j)
                    {
                        data[ristride[1] * (length[1] - j) + ristride[2] * (rlength[2] / 2)]
                            = std::conj(data[ristride[1] * j + ristride[2] * (rlength[2] / 2)]);
                    }

                    if(rlength[0] % 2 == 0)
                    {
                        for(int j = 1; j < (rlength[1] + 1) / 2; ++j)
                        {
                            data[ristride[1] * (rlength[1] - j) + ristride[2] * (rlength[2] / 2)]
                                = std::conj(data[ristride[1] * j + ristride[2] * (rlength[2] / 2)]);
                        }
                    }
                    if(rlength[1] % 2 == 0)
                    {
                        for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                        {
                            data[ristride[0] * (rlength[0] - i) + ristride[2] * (rlength[2] / 2)]
                                = std::conj(data[ristride[0] * i + ristride[2] * (rlength[2] / 2)]);
                        }
                    }

                    // x-y plane:
                    for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                    {
                        for(int j = 1; j < rlength[1]; ++j)
                        {
                            data[ristride[0] * (rlength[0] - i) + ristride[1] * (rlength[1] - j)
                                 + ristride[2] * (rlength[2] / 2)]
                                = std::conj(data[ristride[0] * i + ristride[1] * j
                                                 + ristride[2] * (rlength[2] / 2)]);
                        }
                    }
                }

                // fall-through

            case 2:

                if(rlength[1] % 2 == 0)
                {
                    data[ristride[1] * (rlength[1] / 2)].imag(0.0);
                }

                if(rlength[0] % 2 == 0 && rlength[1] % 2 == 0)
                {
                    data[ristride[0] * (rlength[0] / 2) + ristride[1] * (rlength[1] / 2)].imag(0.0);
                }

                for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                {
                    data[ristride[0] * (rlength[0] - i)] = std::conj(data[ristride[0] * i]);
                }

                if(rlength[1] % 2 == 0)
                {
                    for(int i = 1; i < (rlength[0] + 1) / 2; ++i)
                    {
                        data[ristride[0] * (rlength[0] - i) + ristride[1] * (rlength[1] / 2)]
                            = std::conj(data[ristride[0] * i + ristride[1] * (rlength[1] / 2)]);
                    }
                }

                // fall-through

            case 1:
                data[0].imag(0.0);

                if(rlength[0] % 2 == 0)
                {
                    data[ristride[0] * (rlength[0] / 2)].imag(0.0);
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
                // FIXME: implement
            case 2:
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
template <class Tfloat>
void set_input(std::vector<std::vector<char>>& input,
               const rocfft_array_type         itype,
               const std::vector<size_t>&      length,
               const std::vector<size_t>&      istride,
               const size_t                    idist,
               const size_t                    nbatch)
{
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
    {
        auto idata = (std::complex<Tfloat>*)input[0].data();
        for(size_t b = 0; b < nbatch; b++)
        {
            std::vector<int> index(length.size());
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), b * idist);
                const std::complex<Tfloat> val((Tfloat)rand() / (Tfloat)RAND_MAX,
                                               (Tfloat)rand() / (Tfloat)RAND_MAX);
                idata[i] = val;
            } while(increment_colmajor(index, length));
        }
        break;
    }
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
    {
        auto ireal = (Tfloat*)input[0].data();
        auto iimag = (Tfloat*)input[1].data();
        for(size_t b = 0; b < nbatch; b++)
        {
            std::vector<int> index(length.size());
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), b * idist);
                const std::complex<Tfloat> val((Tfloat)rand() / (Tfloat)RAND_MAX,
                                               (Tfloat)rand() / (Tfloat)RAND_MAX);
                ireal[i] = val.real();
                iimag[i] = val.imag();
            } while(increment_colmajor(index, length));
        }
        break;
    }
    case rocfft_array_type_real:
    {
        auto idata = (Tfloat*)input[0].data();
        for(size_t b = 0; b < nbatch; b++)
        {
            std::vector<int> index(length.size());
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), b * idist);
                const Tfloat val = (Tfloat)rand() / (Tfloat)RAND_MAX;
                idata[i]         = val;
            } while(increment_colmajor(index, length));
        }
        break;
    }
    default:
        throw std::runtime_error("Input layout format not yet supported");
        break;
    }
}

// Given a data type and dimensions, fill the buffer, imposing Hermitian symmetry if
// necessary.
// NB: length is the logical size of the FFT, and not necessarily the data dimensions
std::vector<std::vector<char>> compute_input(const rocfft_precision     precision,
                                             const rocfft_array_type    itype,
                                             const std::vector<size_t>& length,
                                             const std::vector<size_t>& istride,
                                             const size_t               idist,
                                             const size_t               nbatch)
{

    const int ninput
        = (itype == rocfft_array_type_complex_planar || itype == rocfft_array_type_hermitian_planar)
              ? 2
              : 1;

    std::vector<std::vector<char>> input(ninput);

    const bool iscomplex = (itype == rocfft_array_type_complex_interleaved
                            || itype == rocfft_array_type_hermitian_interleaved);

    const size_t isize = idist * nbatch * (iscomplex ? 2 : 1)
                         * (precision == rocfft_precision_double ? sizeof(double) : sizeof(float));

    for(auto& i : input)
    {
        i.resize(isize);
        std::fill(i.begin(), i.end(), 0.0);
    }

    std::vector<size_t> ilength = length;
    if(itype == rocfft_array_type_complex_interleaved
       || itype == rocfft_array_type_hermitian_interleaved)
    {
        ilength[0] = length[0] / 2 + 1;
    }

    if(precision == rocfft_precision_double)
    {
        set_input<double>(input, itype, ilength, istride, idist, nbatch);
    }
    else
    {
        set_input<float>(input, itype, ilength, istride, idist, nbatch);
    }

    if(itype == rocfft_array_type_hermitian_interleaved
       || itype == rocfft_array_type_hermitian_planar)
    {
        if(precision == rocfft_precision_double)
        {
            impose_hermitian_symmetry<double>(input, length, istride, idist, nbatch);
        }
        else
        {
            impose_hermitian_symmetry<float>(input, length, istride, idist, nbatch);
        }
    }
    return input;
}

// TODO: deprecated
template <class T>
void fill_ibuffer(std::vector<void*>         ibuffer,
                  const rocfft_array_type&   itype,
                  const std::vector<size_t>& length,
                  const std::vector<size_t>& istride,
                  const size_t               idist,
                  const size_t               nbatch,
                  const bool                 verbose)
{
    const size_t isize = idist * nbatch;

    // Fill the input buffers
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    {
        std::vector<std::complex<T>> input(idist * nbatch);

        // Input data is constant; output should be delta.
        std::fill(input.begin(), input.end(), 0.0);
        for(size_t b = 0; b < nbatch; b++)
        {
            std::vector<int> index(length.size());
            std::fill(index.begin(), index.end(), 0);
            do
            {
                int i = std::inner_product(index.begin(), index.end(), istride.begin(), b * idist);
                input[i] = 1.0;

            } while(increment_colmajor(index, length));
        }

        if(verbose)
        {
            std::cout << "input:\n";
            printbuffer(input.data(), length, istride, nbatch, idist);
        }

        HIP_V_THROW(hipMemcpy(ibuffer[0], input.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
    break;
    case rocfft_array_type_complex_planar:
    {
        std::vector<T> real(idist * nbatch);
        std::vector<T> imag(idist * nbatch);

        // Input data is constant; output should be delta.
        std::fill(real.begin(), real.end(), 0.0);
        std::fill(imag.begin(), imag.end(), 0.0);
        for(size_t b = 0; b < nbatch; b++)
        {
            std::vector<int> index(length.size());
            std::fill(index.begin(), index.end(), 0);
            do
            {
                int i = std::inner_product(index.begin(), index.end(), istride.begin(), b * idist);
                real[i] = 1.0;
                imag[i] = 0.0;

            } while(increment_colmajor(index, length));
        }

        HIP_V_THROW(hipMemcpy(ibuffer[0], real.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
        HIP_V_THROW(hipMemcpy(ibuffer[1], imag.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
    break;
    case rocfft_array_type_hermitian_interleaved:
    {
        std::vector<std::complex<T>> input(idist * nbatch);

        // Input data is the delta function; output should be constant.
        T delta = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
        std::fill(input.begin(), input.end(), 0.0);
        for(size_t b = 0; b < nbatch; b++)
        {
            size_t p3 = b * idist;
            input[p3] = delta;
        }
        if(verbose)
        {
            std::cout << "\ninput:\n";
            printbuffer(input.data(), length, istride, nbatch, idist);
        }

        HIP_V_THROW(hipMemcpy(ibuffer[0], input.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
    break;
    case rocfft_array_type_hermitian_planar:
    {
        std::vector<T> real(isize);
        std::vector<T> imag(isize);

        // Input data is the delta function; output should be constant.
        std::fill(real.begin(), real.end(), 0.0);
        std::fill(imag.begin(), imag.end(), 0.0);
        T delta = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
        for(size_t b = 0; b < nbatch; b++)
        {
            size_t p3 = b * idist;
            real[p3]  = delta;
        }
        HIP_V_THROW(hipMemcpy(ibuffer[0], real.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
        HIP_V_THROW(hipMemcpy(ibuffer[1], imag.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
    break;
    case rocfft_array_type_real:
    {
        std::vector<T> input(idist * nbatch);

        // Input data is constant; output should be delta.
        std::fill(input.begin(), input.end(), 0.0);
        for(size_t b = 0; b < nbatch; b++)
        {
            std::vector<int> index(length.size());
            std::fill(index.begin(), index.end(), 0);
            do
            {
                const int i
                    = std::inner_product(index.begin(), index.end(), istride.begin(), b * idist);
                input[i] = 1.0;

            } while(increment_colmajor(index, length));
        }

        if(verbose)
        {
            std::cout << "\ninput:\n";
            printbuffer(input.data(), length, istride, nbatch, idist);
        }

        HIP_V_THROW(hipMemcpy(ibuffer[0], input.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
    break;
    default:
        throw std::runtime_error("Input layout format not yet supported");
        break;
    }
}

#endif // RIDER_H
