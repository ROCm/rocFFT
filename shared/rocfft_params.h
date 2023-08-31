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
};

#endif
