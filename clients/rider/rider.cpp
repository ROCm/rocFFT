/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>

#include "./rider.h"
#include "rocfft.h"
#include <boost/program_options.hpp>

#include "./misc.h"

namespace po = boost::program_options;

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

// Perform a transform using rocFFT.  We assume that all input is valid at this point.
// ntrial is the number of trials; if this is 0, then we just do a correctness check.
template <typename T>
int transform(const std::vector<size_t> length,
              const std::vector<size_t> istride,
              const std::vector<size_t> ostride,
              const size_t              idist,
              const size_t              odist,
              size_t                    nbatch,
              const std::vector<size_t> ioffset,
              const std::vector<size_t> ooffset,
              rocfft_array_type         itype,
              rocfft_array_type         otype,
              rocfft_result_placement   place,
              rocfft_precision          precision,
              rocfft_transform_type     transformType,
              double                    scale,
              int                       deviceId,
              const int                 ntrial)
{
    HIP_V_THROW(hipSetDevice(deviceId), " hipSetDevice failed");

    const unsigned dim = length.size();

    unsigned number_of_input_buffers = 0;
    switch(itype)
    {
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        number_of_input_buffers = 2;
        break;
    default:
        number_of_input_buffers = 1;
    }

    unsigned number_of_output_buffers = 0;
    switch(otype)
    {
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        number_of_output_buffers = 2;
        break;
    default:
        number_of_output_buffers = 1;
    }

    size_t isize = idist * nbatch;
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        isize *= sizeof(std::complex<T>);
        break;
    default:
        isize *= sizeof(T);
    }

    size_t osize = odist * nbatch;
    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        osize *= sizeof(std::complex<T>);
        break;
    default:
        osize *= sizeof(T);
    }

    std::vector<void*> ibuffer;
    std::vector<void*> obuffer;

    for(unsigned i = 0; i < number_of_input_buffers; i++)
    {
        ibuffer.push_back(NULL);
        HIP_V_THROW(hipMalloc(&ibuffer[i], isize), "hipMalloc failed");
    }

    for(unsigned i = 0; i < number_of_output_buffers; i++)
    {
        obuffer.push_back(NULL);
        if(place != rocfft_placement_inplace)
            HIP_V_THROW(hipMalloc(&obuffer[i], osize), "hipMalloc failed");
    }

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
        HIP_V_THROW(hipMemcpy(ibuffer[0], input.data(), isize, hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
    break;
    default:
    {
        throw std::runtime_error("Input layout format not yet supported");
    }
    break;
    }

    LIB_V_THROW(rocfft_setup(), " rocfft_setup failed");

    rocfft_plan_description desc = NULL;
    LIB_V_THROW(rocfft_plan_description_create(&desc), "rocfft_plan_description_create failed");

    LIB_V_THROW(rocfft_plan_description_set_data_layout(desc,
                                                        itype,
                                                        otype,
                                                        ioffset.data(),
                                                        ooffset.data(),
                                                        istride.size(),
                                                        istride.data(),
                                                        idist,
                                                        ostride.size(),
                                                        ostride.data(),
                                                        odist),
                "rocfft_plan_description_data_layout failed");

    // NB: not yet implemented in rocFFT.
    // if(scale != 1.0)
    // {
    //     if(precision == rocfft_precision_single)
    //         LIB_V_THROW(
    //             rocfft_plan_description_set_scale_float( desc, (float)scale ),
    //             "rocfft_plan_description_set_scale_float failed" );
    //     else
    //         LIB_V_THROW(
    //             rocfft_plan_description_set_scale_double( desc, scale ),
    //             "rocfft_plan_description_set_scale_double failed" );
    // }

    rocfft_plan plan = NULL;
    LIB_V_THROW(
        rocfft_plan_create(
            &plan, place, transformType, precision, length.size(), length.data(), nbatch, desc),
        "rocfft_plan_create failed");

    LIB_V_THROW(rocfft_plan_get_print(plan), "rocfft_plan_get_print failed");

    // Get the buffersize
    size_t workBufferSize = 0;
    LIB_V_THROW(rocfft_plan_get_work_buffer_size(plan, &workBufferSize),
                "rocfft_plan_get_work_buffer_size failed");

    // Allocate the work buffer
    void* workBuffer = NULL;
    if(workBufferSize)
    {
        HIP_V_THROW(hipMalloc(&workBuffer, workBufferSize), "Creating intmediate Buffer failed");
    }

    void** BuffersOut = (place == rocfft_placement_inplace) ? NULL : obuffer.data();

    rocfft_execution_info info = NULL;
    LIB_V_THROW(rocfft_execution_info_create(&info), "rocfft_execution_info_create failed");
    if(workBuffer != NULL)
    {
        LIB_V_THROW(rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize),
                    "rocfft_execution_info_set_work_buffer failed");
    }

    // Execute once for basic functional test
    LIB_V_THROW(rocfft_execute(plan,
                               ibuffer.data(),
                               (place == rocfft_placement_inplace) ? NULL : obuffer.data(),
                               info),
                "rocfft_execute failed");

    HIP_V_THROW(hipDeviceSynchronize(), "hipDeviceSynchronize failed");

    std::vector<float> gpu_time(ntrial);

    if(ntrial > 0)
    {
        hipEvent_t start, stop;
        HIP_V_THROW(hipEventCreate(&start), "hipEventCreate failed");
        HIP_V_THROW(hipEventCreate(&stop), "hipEventCreate failed");

        for(int itrial = 0; itrial < ntrial; ++itrial)
        {
            HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

            LIB_V_THROW(rocfft_execute(plan, ibuffer.data(), BuffersOut, info),
                        "rocfft_execute failed");

            HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
            HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

            hipEventElapsedTime(&gpu_time[itrial], start, stop);
            gpu_time[itrial];
        }

        HIP_V_THROW(hipDeviceSynchronize(), "hipDeviceSynchronize failed");

        std::cout << "\nExecution gpu time:";
        for(const auto& i : gpu_time)
        {
            std::cout << " " << i;
        }
        std::cout << " ms" << std::endl;

        std::cout << "Execution gflops:  ";
        const double totsize
            = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
        const double k
            = ((itype == rocfft_array_type_real) || (otype == rocfft_array_type_real)) ? 2.5 : 5.0;
        const double opscount = (double)nbatch * k * totsize * log(totsize) / log(2.0);
        for(const auto& i : gpu_time)
        {
            std::cout << " " << opscount / (1e6 * i);
        }
        std::cout << std::endl;
    }

    if(workBuffer)
        HIP_V_THROW(hipFree(workBuffer), "hipFree failed");

    LIB_V_THROW(rocfft_plan_description_destroy(desc), "rocfft_plan_description_destroy failed");

    LIB_V_THROW(rocfft_execution_info_destroy(info), "rocfft_execution_info_destroy failed");

    bool   checkflag = false;
    double err_ratio = 1E-6;

    // Read and check output data
    // This check is not valid if the FFT is executed multiple times.
    if(ntrial == 0)
    {
        switch(otype)
        {
        case rocfft_array_type_hermitian_interleaved:
        case rocfft_array_type_complex_interleaved:
        {
            std::vector<std::complex<T>> output(odist * nbatch);

            HIP_V_THROW(hipMemcpy(output.data(),
                                  (place == rocfft_placement_inplace) ? ibuffer[0] : BuffersOut[0],
                                  osize,
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");

            // check output data: we should get a peak at the index, and zero otherwise.
            T delta = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
            for(size_t b = 0; b < nbatch; b++)
            {
                std::vector<int> index(length.size());
                std::fill(index.begin(), index.end(), 0);
                do
                {
                    int i = std::inner_product(
                        index.begin(), index.end(), ostride.begin(), b * odist);
                    if(i == b * odist)
                    {
                        // DC mode
                        if(fabs(output[i] - std::complex<T>(delta, 0)) / delta > err_ratio)
                        {
                            checkflag = true;
                            break;
                        }
                    }
                    else
                    {
                        if(fabs(output[i]) / delta > err_ratio)
                        {
                            checkflag = true;
                            break;
                        }
                    }
                } while(increment_colmajor(index, length));
            }
        }
        break;
        case rocfft_array_type_hermitian_planar:
        case rocfft_array_type_complex_planar:
        {
            std::vector<T> real(osize);
            HIP_V_THROW(hipMemcpy(real.data(),
                                  (place == rocfft_placement_inplace) ? ibuffer[0] : BuffersOut[0],
                                  osize,
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");

            std::vector<T> imag(osize);
            HIP_V_THROW(hipMemcpy(imag.data(),
                                  (place == rocfft_placement_inplace) ? ibuffer[1] : BuffersOut[1],
                                  osize,
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");

            T delta = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
            for(size_t b = 0; b < nbatch; b++)
            {
                std::vector<int> index(length.size());
                std::fill(index.begin(), index.end(), 0);
                do
                {
                    int i = std::inner_product(
                        index.begin(), index.end(), ostride.begin(), b * odist);
                    if(i == b * odist)
                    {
                        // DC mode
                        if(fabs(std::complex<T>(real[i], imag[i]) - std::complex<T>(delta, 0))
                               / delta
                           > err_ratio)
                        {
                            checkflag = true;
                            break;
                        }
                    }
                    else
                    {
                        if(fabs(real[i]) / delta > err_ratio || fabs(imag[i]) / delta > err_ratio)
                        {
                            checkflag = true;
                            break;
                        }
                    }
                } while(increment_colmajor(index, length));
            }
        }
        break;
        case rocfft_array_type_real:
        {
            std::vector<T> output(odist * nbatch);

            HIP_V_THROW(hipMemcpy(output.data(),
                                  (place == rocfft_placement_inplace) ? ibuffer[0] : BuffersOut[0],
                                  osize,
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");

            // NB: we compare with delta, because the inverse transform isn't normalized.
            T delta = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());

            std::vector<int> index(length.size());
            std::fill(index.begin(), index.end(), 0);
            for(size_t b = 0; b < nbatch; b++)
            {
                do
                {
                    const int i = std::inner_product(
                        index.begin(), index.end(), ostride.begin(), b * odist);
                    if(fabs(output[i] - delta) > err_ratio)
                    {
                        checkflag = true;
                        break;
                    }
                } while(increment_colmajor(index, length));
            }
        }
        break;
        default:
        {
            throw std::runtime_error("Input layout format not yet supported");
        }
        break;
        }

        if(checkflag)
        {
            std::cout << "\n\n\t\tRider Test *****FAIL*****" << std::endl;
        }
        else
        {
            std::cout << "\n\n\t\tRider Test *****PASS*****" << std::endl;
        }
    }

    LIB_V_THROW(rocfft_plan_destroy(plan), "rocfft_plan_destroy failed");
    LIB_V_THROW(rocfft_cleanup(), "rocfft_cleanup failed");

    for(auto& buf : ibuffer)
    {
        if(buf != NULL)
        {
            HIP_V_THROW(hipFree(buf), "hipFree failed");
            buf = NULL;
        }
    }
    for(auto& buf : obuffer)
    {
        if(buf != NULL)
        {
            HIP_V_THROW(hipFree(buf), "hipFree failed");
            buf = NULL;
        }
    }

    return checkflag ? -1 : 0;
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

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // hip Device number for running tests:
    int deviceId;

    // Transform type parameters:
    rocfft_transform_type transformType;
    rocfft_array_type     itype;
    rocfft_array_type     otype;

    // Number of performance trial samples
    int ntrial;

    // Number of batches:
    size_t nbatch = 1;

    // Scale for transform
    double scale = 1.0;

    // Transform length:
    std::vector<size_t> length;

    // Transform input and output strides:
    std::vector<size_t> istride;
    std::vector<size_t> ostride;

    // Offset to start of buffer (or buffers, for planar format):
    std::vector<size_t> ioffset;
    std::vector<size_t> ooffset;

    // Input and output distances:
    size_t idist;
    size_t odist;

    // clang-format doesn't handle boost program options very well:
    // clang-format off
        
    // Declare the supported options.
    po::options_description desc("rocfft rider command line options");
    desc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the rocfft library")
        ("device", po::value<int>(&deviceId)->default_value(0),
         "Select a specific device id")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1),
         "Trial size for the problem")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ("transformType,t", po::value<rocfft_transform_type>(&transformType)
         ->default_value(rocfft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "idist", po::value<size_t>(&idist)->default_value(0),
          "input distance between successive members when batch size > 1")
        ( "odist", po::value<size_t>(&odist)->default_value(0),
          "output distance between successive members when batch size > 1")
        ("scale", po::value<double>(&scale)->default_value(1.0), "Specify the scaling factor ")
        ( "batchSize,b", po::value<size_t>(&nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<rocfft_array_type>(&itype)
          ->default_value(rocfft_array_type_complex_interleaved),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<rocfft_array_type>(&otype)
          ->default_value(rocfft_array_type_complex_interleaved),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  po::value<std::vector<size_t>>(&length)->multitoken(), "Lengths.")
        ("istride", po::value<std::vector<size_t>>(&istride)->multitoken(), "Input strides.")
        ("ostride", po::value<std::vector<size_t>>(&ostride)->multitoken(), "Output strides.")
        ("ioffset", po::value<std::vector<size_t>>(&ioffset)->multitoken(), "Input offsets.")
        ("ooffset", po::value<std::vector<size_t>>(&ooffset)->multitoken(), "Output offsets.");

    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(vm.count("version"))
    {
        char v[256];
        rocfft_get_version_string(v, 256);
        std::cout << "version " << v << std::endl;
        return 0;
    }

    if(!vm.count("length"))
    {
        std::cout << "Please specify transform length!" << std::endl;
        std::cout << desc << std::endl;
        return 0;
    }

    const rocfft_result_placement place
        = vm.count("notInPlace") ? rocfft_placement_notinplace : rocfft_placement_inplace;
    const rocfft_precision precision
        = vm.count("double") ? rocfft_precision_double : rocfft_precision_single;

    if(vm.count("notInPlace"))
    {
        std::cout << "out-of-place\n";
    }
    else
    {
        std::cout << "in-place\n";
    }

    if(vm.count("ntrial"))
    {
        std::cout << "Running profile with " << ntrial << " samples\n";
    }

    if(vm.count("length"))
    {
        std::cout << "length:";
        for(auto& i : length)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(vm.count("istride"))
    {
        std::cout << "istride:";
        for(auto& i : istride)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(vm.count("ostride"))
    {
        std::cout << "ostride:";
        for(auto& i : ostride)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(vm.count("idist"))
    {
        std::cout << "idist: " << idist << "\n";
    }
    if(vm.count("odist"))
    {
        std::cout << "odist: " << odist << "\n";
    }

    if(vm.count("ioffset"))
    {
        std::cout << "ioffset:";
        for(auto& i : ioffset)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ooffset"))
    {
        std::cout << "ooffset:";
        for(auto& i : ooffset)
            std::cout << " " << i;
        std::cout << "\n";
    }

    std::cout << std::flush;

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

    if(idist == 0)
    {
        idist = length[length.size() - 1] * istride[istride.size() - 1];
    }

    if(odist == 0)
    {
        odist = length[length.size() - 1] * ostride[ostride.size() - 1];
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

    int tret = 0;
    try
    {
        if(precision == rocfft_precision_single)
        {
            tret = transform<float>(length,
                                    istride,
                                    ostride,
                                    idist,
                                    odist,
                                    nbatch,
                                    ioffset,
                                    ooffset,
                                    itype,
                                    otype,
                                    place,
                                    precision,
                                    transformType,
                                    scale,
                                    deviceId,
                                    ntrial);
        }
        else
        {
            tret = transform<double>(length,
                                     istride,
                                     ostride,
                                     idist,
                                     odist,
                                     nbatch,
                                     ioffset,
                                     ooffset,
                                     itype,
                                     otype,
                                     place,
                                     precision,
                                     transformType,
                                     scale,
                                     deviceId,
                                     ntrial);
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "rocfft error condition reported:" << std::endl << e.what() << std::endl;
        return 1;
    }

    return tret;
}
