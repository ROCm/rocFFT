// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_PARAMS_H
#define ROCFFT_PARAMS_H

#include "../shared/fft_params.h"
#include "../shared/gpubuf.h"
#include "rocfft/rocfft.h"

// Return the string of the rocfft_status code
static std::string rocfft_status_to_string(const rocfft_status ret)
{
    switch(ret)
    {
    case rocfft_status_success:
        return "rocfft_status_success";
    case rocfft_status_failure:
        return "rocfft_status_failure";
    case rocfft_status_invalid_arg_value:
        return "rocfft_status_invalid_arg_value";
    case rocfft_status_invalid_dimensions:
        return "rocfft_status_invalid_dimensions";
    case rocfft_status_invalid_array_type:
        return "rocfft_status_invalid_array_type";
    case rocfft_status_invalid_strides:
        return "rocfft_status_invalid_strides";
    case rocfft_status_invalid_distance:
        return "rocfft_status_invalid_distance";
    case rocfft_status_invalid_offset:
        return "rocfft_status_invalid_offset";
    case rocfft_status_invalid_work_buffer:
        return "rocfft_status_invalid_work_buffer";
    default:
        throw std::runtime_error("unknown rocfft_status");
    }
}

inline fft_status fft_status_from_rocfftparams(const rocfft_status val)
{
    switch(val)
    {
    case rocfft_status_success:
        return fft_status_success;
    case rocfft_status_failure:
        return fft_status_failure;
    case rocfft_status_invalid_arg_value:
        return fft_status_invalid_arg_value;
    case rocfft_status_invalid_dimensions:
        return fft_status_invalid_dimensions;
    case rocfft_status_invalid_array_type:
        return fft_status_invalid_array_type;
    case rocfft_status_invalid_strides:
        return fft_status_invalid_strides;
    case rocfft_status_invalid_distance:
        return fft_status_invalid_distance;
    case rocfft_status_invalid_offset:
        return fft_status_invalid_offset;
    case rocfft_status_invalid_work_buffer:
        return fft_status_invalid_work_buffer;
    default:
        throw std::runtime_error("Invalid status");
    }
}

inline rocfft_precision rocfft_precision_from_fftparams(const fft_precision val)
{
    switch(val)
    {
    case fft_precision_single:
        return rocfft_precision_single;
    case fft_precision_double:
        return rocfft_precision_double;
    case fft_precision_half:
        return rocfft_precision_half;
    default:
        throw std::runtime_error("Invalid precision");
    }
}

inline rocfft_array_type rocfft_array_type_from_fftparams(const fft_array_type val)
{
    switch(val)
    {
    case fft_array_type_complex_interleaved:
        return rocfft_array_type_complex_interleaved;
    case fft_array_type_complex_planar:
        return rocfft_array_type_complex_planar;
    case fft_array_type_real:
        return rocfft_array_type_real;
    case fft_array_type_hermitian_interleaved:
        return rocfft_array_type_hermitian_interleaved;
    case fft_array_type_hermitian_planar:
        return rocfft_array_type_hermitian_planar;
    case fft_array_type_unset:
        return rocfft_array_type_unset;
    }
    return rocfft_array_type_unset;
}

inline rocfft_transform_type rocfft_transform_type_from_fftparams(const fft_transform_type val)
{
    switch(val)
    {
    case fft_transform_type_complex_forward:
        return rocfft_transform_type_complex_forward;
    case fft_transform_type_complex_inverse:
        return rocfft_transform_type_complex_inverse;
    case fft_transform_type_real_forward:
        return rocfft_transform_type_real_forward;
    case fft_transform_type_real_inverse:
        return rocfft_transform_type_real_inverse;
    default:
        throw std::runtime_error("Invalid transform type");
    }
}

inline rocfft_result_placement
    rocfft_result_placement_from_fftparams(const fft_result_placement val)
{
    switch(val)
    {
    case fft_placement_inplace:
        return rocfft_placement_inplace;
    case fft_placement_notinplace:
        return rocfft_placement_notinplace;
    default:
        throw std::runtime_error("Invalid result placement");
    }
}

class rocfft_params : public fft_params
{
public:
    rocfft_plan             plan = nullptr;
    rocfft_execution_info   info = nullptr;
    rocfft_plan_description desc = nullptr;
    gpubuf_t<void>          wbuffer;

    explicit rocfft_params(){};

    explicit rocfft_params(const fft_params& p)
        : fft_params(p){};

    rocfft_params(const rocfft_params&) = delete;
    rocfft_params& operator=(const rocfft_params&) = delete;

    ~rocfft_params()
    {
        free();
    };

    void free()
    {
        if(plan != nullptr)
        {
            rocfft_plan_destroy(plan);
            plan = nullptr;
        }
        if(info != nullptr)
        {
            rocfft_execution_info_destroy(info);
            info = nullptr;
        }
        if(desc != nullptr)
        {
            rocfft_plan_description_destroy(desc);
            desc = nullptr;
        }
        wbuffer.free();
    }

    void validate_fields() const override
    {
        // row-major lengths including batch (i.e. batch is at the front)
        std::vector<size_t> length_with_batch{nbatch};
        std::copy(length.begin(), length.end(), std::back_inserter(length_with_batch));

        auto validate_field = [&](const fft_field& f) {
            for(const auto& b : f.bricks)
            {
                // bricks must have same dim as FFT, including batch
                if(b.lower.size() != length.size() + 1 || b.upper.size() != length.size() + 1
                   || b.stride.size() != length.size() + 1)
                    throw std::runtime_error(
                        "brick dimension does not match FFT + batch dimension");

                // ensure lower < upper, and that both fit in the FFT + batch dims
                if(!std::lexicographical_compare(
                       b.lower.begin(), b.lower.end(), b.upper.begin(), b.upper.end()))
                    throw std::runtime_error("brick lower index is not less than upper index");

                if(!std::lexicographical_compare(b.lower.begin(),
                                                 b.lower.end(),
                                                 length_with_batch.begin(),
                                                 length_with_batch.end()))
                    throw std::runtime_error(
                        "brick lower index is not less than FFT + batch length");

                if(!std::lexicographical_compare(b.upper.begin(),
                                                 b.upper.end(),
                                                 length_with_batch.begin(),
                                                 length_with_batch.end())
                   && b.upper != length_with_batch)
                    throw std::runtime_error("brick upper index is not <= FFT + batch length");
            }
        };

        for(const auto& ifield : ifields)
            validate_field(ifield);
        for(const auto& ofield : ofields)
            validate_field(ofield);
    }

    rocfft_precision get_rocfft_precision()
    {
        return rocfft_precision_from_fftparams(precision);
    }

    size_t vram_footprint() override
    {
        size_t val = fft_params::vram_footprint();
        if(setup_structs() != fft_status_success)
        {
            throw std::runtime_error("Struct setup failed");
        }
        val += workbuffersize;

        return val;
    }

    // Convert the generic fft_field structure to a rocfft_field
    // structure that can be passed to rocFFT.  In particular, we need
    // to convert from row-major to column-major.
    static rocfft_field fft_field_to_rocfft_field(const fft_field& f)
    {
        rocfft_field rfield = nullptr;
        if(f.bricks.empty())
            return rfield;

        if(rocfft_field_create(&rfield) != rocfft_status_success)
            throw std::runtime_error("rocfft_field_create failed");
        for(const auto& b : f.bricks)
        {
            // rocFFT wants column-major bricks and fft_params stores
            // row-major
            std::vector<size_t> lower_cm;
            std::copy(b.lower.rbegin(), b.lower.rend(), std::back_inserter(lower_cm));
            std::vector<size_t> upper_cm;
            std::copy(b.upper.rbegin(), b.upper.rend(), std::back_inserter(upper_cm));
            std::vector<size_t> stride_cm;
            std::copy(b.stride.rbegin(), b.stride.rend(), std::back_inserter(stride_cm));

            rocfft_brick rbrick = nullptr;
            if(rocfft_brick_create(&rbrick,
                                   lower_cm.data(), // field_lower
                                   upper_cm.data(), // field_upper
                                   stride_cm.data(), // brick_stride
                                   lower_cm.size(), // dim
                                   b.device) // deviceID
               != rocfft_status_success)
                throw std::runtime_error("rocfft_brick_create failed");

            if(rocfft_field_add_brick(rfield, rbrick) != rocfft_status_success)
                throw std::runtime_error("rocfft_field_add_brick failed");

            rocfft_brick_destroy(rbrick);
        }
        return rfield;
    }

    fft_status setup_structs()
    {
        rocfft_status fft_status = rocfft_status_success;
        if(desc == nullptr)
        {
            rocfft_plan_description_create(&desc);
            if(fft_status != rocfft_status_success)
                return fft_status_from_rocfftparams(fft_status);

            fft_status
                = rocfft_plan_description_set_data_layout(desc,
                                                          rocfft_array_type_from_fftparams(itype),
                                                          rocfft_array_type_from_fftparams(otype),
                                                          ioffset.data(),
                                                          ooffset.data(),
                                                          istride_cm().size(),
                                                          istride_cm().data(),
                                                          idist,
                                                          ostride_cm().size(),
                                                          ostride_cm().data(),
                                                          odist);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_plan_description_set_data_layout failed");
            }

            if(scale_factor != 1.0)
            {
                fft_status = rocfft_plan_description_set_scale_factor(desc, scale_factor);
                if(fft_status != rocfft_status_success)
                {
                    throw std::runtime_error("rocfft_plan_description_set_scale_factor failed");
                }
            }

            for(const auto& ifield : ifields)
            {
                rocfft_field infield = fft_field_to_rocfft_field(ifield);
                if(rocfft_plan_description_add_infield(desc, infield) != rocfft_status_success)
                    throw std::runtime_error("rocfft_description_add_infield failed");
                rocfft_field_destroy(infield);
            }

            for(const auto& ofield : ofields)
            {
                rocfft_field outfield = fft_field_to_rocfft_field(ofield);
                if(rocfft_plan_description_add_outfield(desc, outfield) != rocfft_status_success)
                    throw std::runtime_error("rocfft_description_add_outfield failed");
                rocfft_field_destroy(outfield);
            }
        }

        if(plan == nullptr)
        {
            fft_status = rocfft_plan_create(&plan,
                                            rocfft_result_placement_from_fftparams(placement),
                                            rocfft_transform_type_from_fftparams(transform_type),
                                            get_rocfft_precision(),
                                            length_cm().size(),
                                            length_cm().data(),
                                            nbatch,
                                            desc);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_plan_create failed");
            }
        }

        if(info == nullptr)
        {
            fft_status = rocfft_execution_info_create(&info);
            if(fft_status != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_execution_info_create failed");
            }
        }

        fft_status = rocfft_plan_get_work_buffer_size(plan, &workbuffersize);
        if(fft_status != rocfft_status_success)
        {
            throw std::runtime_error("rocfft_plan_get_work_buffer_size failed");
        }

        return fft_status_from_rocfftparams(fft_status);
    }

    fft_status create_plan() override
    {
        fft_status ret = setup_structs();
        if(ret != fft_status_success)
        {
            return ret;
        }
        if(workbuffersize > 0)
        {
            hipError_t hip_status = hipSuccess;
            hip_status            = wbuffer.alloc(workbuffersize);
            if(hip_status != hipSuccess)
            {
                std::ostringstream oss;
                oss << "work buffer allocation failed (" << workbuffersize << " requested)";
                size_t mem_free  = 0;
                size_t mem_total = 0;
                hip_status       = hipMemGetInfo(&mem_free, &mem_total);
                if(hip_status == hipSuccess)
                {
                    oss << "free vram: " << mem_free << " total vram: " << mem_total;
                }
                else
                {
                    oss << "hipMemGetInfo also failed";
                }
                throw work_buffer_alloc_failure(oss.str());
            }

            auto rocret
                = rocfft_execution_info_set_work_buffer(info, wbuffer.data(), workbuffersize);
            if(rocret != rocfft_status_success)
            {
                throw std::runtime_error("rocfft_execution_info_set_work_buffer failed");
            }
        }

        return ret;
    }

    fft_status set_callbacks(void* load_cb_host,
                             void* load_cb_data,
                             void* store_cb_host,
                             void* store_cb_data) override
    {
        if(run_callbacks)
        {
            auto roc_status
                = rocfft_execution_info_set_load_callback(info, &load_cb_host, &load_cb_data, 0);
            if(roc_status != rocfft_status_success)
                return fft_status_from_rocfftparams(roc_status);

            roc_status
                = rocfft_execution_info_set_store_callback(info, &store_cb_host, &store_cb_data, 0);
            if(roc_status != rocfft_status_success)
                return fft_status_from_rocfftparams(roc_status);
        }
        return fft_status_success;
    }

    fft_status execute(void** in, void** out) override
    {
        auto ret = rocfft_execute(plan, in, out, info);
        return fft_status_from_rocfftparams(ret);
    }

    // scatter data to multiple GPUs and adjust I/O buffers to match
    void multi_gpu_prepare(std::vector<gpubuf>& ibuffer,
                           std::vector<void*>&  pibuffer,
                           std::vector<void*>&  pobuffer) override
    {
        auto alloc_fields = [&](const fft_params::fft_field& field,
                                fft_array_type               array_type,
                                std::vector<void*>&          pbuffer,
                                bool                         copy_input) {
            if(field.bricks.empty())
                return;

            // we have a field defined, clear the list of buffers as
            // we'll be allocating new ones for each brick
            pbuffer.clear();

            for(const auto& b : field.bricks)
            {
                // get brick's length - note that this includes batch
                // dimension
                const auto brick_len    = b.length();
                const auto brick_stride = b.stride;

                const size_t brick_size_elems = product(brick_len.begin(), brick_len.end());
                const size_t elem_size_bytes  = var_size<size_t>(precision, array_type);
                const size_t brick_size_bytes = brick_size_elems * elem_size_bytes;

                // set device for the alloc, but we want to return to the
                // default device as the source of a following memcpy
                {
                    rocfft_scoped_device dev(b.device);
                    multi_gpu_data.emplace_back();
                    if(multi_gpu_data.back().alloc(brick_size_bytes) != hipSuccess)
                        throw std::runtime_error("device allocation failure");
                    pbuffer.push_back(multi_gpu_data.back().data());
                }

                if(copy_input)
                {
                    // For now, assume we're only splitting on highest FFT
                    // dimension, lower-dimensional FFT data is all
                    // contiguous, and batches are contiguous in each brick.
                    //
                    // That means we can express this as a 2D memcpy.
                    const size_t unbatched_elems_per_brick
                        = product(brick_len.begin() + 1, brick_len.end());
                    const size_t unbatched_elems_per_fft = product(length.begin(), length.end());

                    // get this brick's starting offset in the field
                    const size_t brick_offset
                        = b.lower_field_offset(istride, idist) * elem_size_bytes;

                    // copy from original input - note that we're
                    // assuming interleaved data so ibuffer has only one
                    // gpubuf
                    if(hipMemcpy2D(pbuffer.back(),
                                   unbatched_elems_per_brick * elem_size_bytes,
                                   ibuffer.front().data_offset(brick_offset),
                                   unbatched_elems_per_fft * elem_size_bytes,
                                   unbatched_elems_per_brick * elem_size_bytes,
                                   brick_len.front(),
                                   hipMemcpyHostToDevice)
                       != hipSuccess)
                        throw std::runtime_error("hipMemcpy failure");
                }
            }

            // if we copied the input to all the other devices, and
            // this is an out-of-place transform, we no longer
            // need the original input
            if(copy_input && placement == fft_placement_notinplace)
                ibuffer.clear();
        };

        // assume one input, one output field for simple cases
        if(!ifields.empty())
            alloc_fields(ifields.front(), itype, pibuffer, true);
        if(!ofields.empty())
        {
            if(!ifields.empty() && placement == fft_placement_inplace)
                pobuffer = pibuffer;
            else
                alloc_fields(ofields.front(), otype, pobuffer, false);
        }
    }

    // when preparing for multi-GPU transform, we need to allocate data
    // on each GPU.  This vector remembers all of those allocations.
    std::vector<gpubuf> multi_gpu_data;

    // gather data after multi-GPU FFT for verification
    void multi_gpu_finalize(std::vector<gpubuf>& obuffer, std::vector<void*>& pobuffer) override
    {
        if(ofields.empty())
            return;

        for(size_t i = 0; i < ofields.front().bricks.size(); ++i)
        {
            const auto& b         = ofields.front().bricks[i];
            const auto& brick_ptr = pobuffer[i];

            const auto brick_len = b.length();

            const size_t elem_size_bytes = var_size<size_t>(precision, otype);

            // get this brick's starting offset in the field
            const size_t brick_offset = b.lower_field_offset(ostride, odist) * elem_size_bytes;

            // switch device to where we're copying from
            rocfft_scoped_device dev(b.device);

            // For now, assume we're only splitting on highest FFT
            // dimension, lower-dimensional FFT data is all
            // contiguous, and batches are contiguous in each brick.
            //
            // That means we can express this as a 2D memcpy.
            const size_t unbatched_elems_per_brick
                = product(brick_len.begin() + 1, brick_len.end());
            const auto   output_length = olength();
            const size_t unbatched_elems_per_fft
                = product(output_length.begin(), output_length.end());

            // copy to original output buffer - note that
            // we're assuming interleaved data so obuffer
            // has only one gpubuf
            if(hipMemcpy2D(obuffer.front().data_offset(brick_offset),
                           unbatched_elems_per_fft * elem_size_bytes,
                           brick_ptr,
                           unbatched_elems_per_brick * elem_size_bytes,
                           unbatched_elems_per_brick * elem_size_bytes,
                           brick_len.front(),
                           hipMemcpyDeviceToDevice)
               != hipSuccess)
                throw std::runtime_error("hipMemcpy failure");

            // device-to-device transfers don't synchronize with the
            // host, add explicit sync
            (void)hipDeviceSynchronize();
        }
        pobuffer.clear();
        pobuffer.push_back(obuffer.front().data());
    }
};

#endif
