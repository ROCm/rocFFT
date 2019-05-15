/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#pragma once
#if !defined(ROCFFT_TRANSFORM_H)
#define ROCFFT_TRANSFORM_H

#include "../rider/misc.h" // to use LIB_V_THROW and HIP_V_THROW
#include "buffer.h"
#include "rocfft.h"
#include "test_constants.h"
#include <iostream>
#include <vector>

using namespace std;

//    unique_ptr is a smart pointer that retains sole ownership of an object
//    through a pointer and destroys that object
//    when the unique_ptr goes out of scope. No two unique_ptr instances can
//    manage the same object.
//    Custom deleter functions for our unique_ptr smart pointer class
//    In my version 1, I do not use it

template <class T>
rocfft_status rocfft_plan_create_template(rocfft_plan*                  plan,
                                          rocfft_result_placement       placement,
                                          rocfft_transform_type         transform_type,
                                          size_t                        dimensions,
                                          const size_t*                 lengths,
                                          size_t                        number_of_transforms,
                                          const rocfft_plan_description description);

template <class T>
rocfft_status rocfft_set_scale_template(const rocfft_plan_description description, const T scale);

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
template <class T>
class rocfft
{
private:
    rocfft_array_type       _input_layout, _output_layout;
    rocfft_result_placement _placement;

    size_t dim;

    rocfft_plan             plan;
    rocfft_plan_description desc;
    rocfft_execution_info   info;
    rocfft_transform_type   _transformation_direction;

    vector<size_t> lengths;
    size_t         batch_size;

    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    // hipStream_t stream;

    //[0] for REAL, [1] for IMAG part, IMAG is not used if data type is real
    void* input_device_buffers[2];
    void* output_device_buffers[2];

    buffer<T> input;
    buffer<T> output;

    T scale;

    size_t device_workspace_size;
    void*  device_workspace;

public:
    /*****************************************************/
    rocfft(const std::vector<size_t>     lengths_in,
           const size_t                  batch_size_in,
           const std::vector<size_t>     input_strides_in,
           const std::vector<size_t>     output_strides_in,
           const size_t                  input_distance_in,
           const size_t                  output_distance_in,
           const rocfft_array_type       input_layout_in,
           const rocfft_array_type       output_layout_in,
           const rocfft_result_placement placement_in,
           const rocfft_transform_type   transform_type_in,
           const T                       scale_in)
    try : dim(lengths_in.size()),
          lengths(lengths_in),
          batch_size(batch_size_in),
          input_strides(input_strides_in),
          output_strides(output_strides_in),
          _input_layout(input_layout_in),
          _output_layout(output_layout_in),
          _placement(placement_in),
          _transformation_direction(transform_type_in),
          scale(scale_in),
          device_workspace_size(0),
          input(lengths_in.size(),
                lengths_in.data(),
                input_strides_in.data(),
                batch_size_in,
                input_distance_in,
                _input_layout,
                _placement),
          output(lengths_in.size(),
                 lengths_in.data(),
                 output_strides_in.data(),
                 batch_size_in,
                 output_distance_in,
                 _output_layout,
                 _placement)
    {

        argument_check(); // TODO add more FFT argument check

        // verbose_output();// DEBUG

        /********************create plan and perform the FFT transformation
     * *********************************/

        input_device_buffers[0]  = NULL;
        input_device_buffers[1]  = NULL;
        output_device_buffers[0] = NULL;
        output_device_buffers[1] = NULL;
        device_workspace         = NULL;

        initialize_resource(); // allocate

        LIB_V_THROW(rocfft_setup(), "rocfft_setup failed");

        plan = NULL;
        desc = NULL;
        info = NULL;

        initialize_plan();

        /*****************************************************/
    }
    catch(const exception&)
    {
        throw;
    }

    /*****************************************************/
    void initialize_resource()
    {

        // size_in_butes is calculated in Class input buffer
        if(is_planar(_input_layout))
        {
            HIP_V_THROW(hipMalloc(&(input_device_buffers[0]), input.size_in_bytes()),
                        "hipMalloc failed");
            HIP_V_THROW(hipMalloc(&(input_device_buffers[1]), input.size_in_bytes()),
                        "hipMalloc failed");
        }
        else if(is_interleaved(_input_layout))
        {
            HIP_V_THROW(hipMalloc(&(input_device_buffers[0]), input.size_in_bytes()),
                        "hipMalloc failed");
        }
        else if(is_real(_input_layout))
        {
            HIP_V_THROW(hipMalloc(&(input_device_buffers[0]), input.size_in_bytes()),
                        "hipMalloc failed");
        }

        if(_placement != rocfft_placement_inplace)
        {
            if(is_planar(_output_layout))
            {
                HIP_V_THROW(hipMalloc(&(output_device_buffers[0]), output.size_in_bytes()),
                            "hipMalloc failed");
                HIP_V_THROW(hipMalloc(&(output_device_buffers[1]), output.size_in_bytes()),
                            "hipMalloc failed");
            }
            else if(is_interleaved(_output_layout))
            {
                HIP_V_THROW(hipMalloc(&(output_device_buffers[0]), output.size_in_bytes()),
                            "hipMalloc failed");
            }
            else if(is_real(_output_layout))
            {
                HIP_V_THROW(hipMalloc(&(output_device_buffers[0]), output.size_in_bytes()),
                            "hipMalloc failed");
            }
        }
    }

    /*****************************************************/
    void initialize_plan()
    {

        if(input_strides[0] == 1 && output_strides[0] == 1 && scale == 1.0)
        {
            LIB_V_THROW(rocfft_plan_create_template<T>(&plan,
                                                       _placement,
                                                       _transformation_direction,
                                                       dim,
                                                       lengths.data(),
                                                       batch_size,
                                                       NULL),
                        "rocfft_plan_create failed"); // simply case plan create
        }
        else
        {
            set_layouts(); // explicitely set layout and then create plan, TODO
        }
#ifdef DEBUG
        LIB_V_THROW(rocfft_plan_get_print(plan), "rocfft_plan_get_print failed");
#endif

        // get the worksapce_size based on the plan
        LIB_V_THROW(rocfft_plan_get_work_buffer_size(plan, &device_workspace_size),
                    "rocfft_plan_get_work_buffer_size failed");
// allocate the worksapce
#ifdef DEBUG
        printf("Device work buffer size in bytes= %zu\n", device_workspace_size);
#endif
        if(device_workspace_size)
        {
            HIP_V_THROW(hipMalloc(&device_workspace, device_workspace_size),
                        "Creating intmediate Buffer failed");
        }
    }

    /*****************************************************/
    void set_layouts()
    {

        LIB_V_THROW(rocfft_plan_description_create(&desc), "rocfft_plan_description_create failed");
        // TODO offset non-packed data; only works for 1D now

        size_t output_distance = output_strides[0] * lengths[0]; // TODO
        if(is_hermitian(_output_layout)) // if real to hermitian
        {
            output_distance = output_distance / 2 + 1;
        }

        LIB_V_THROW(rocfft_plan_description_set_data_layout(desc,
                                                            _input_layout,
                                                            _output_layout,
                                                            0,
                                                            0,
                                                            input_strides.size(),
                                                            input_strides.data(),
                                                            input_strides[0] * lengths[0],
                                                            output_strides.size(),
                                                            output_strides.data(),
                                                            output_distance),
                    "rocfft_plan_description_data_layout failed");

        // In rocfft, scale must be set before plan create
        LIB_V_THROW(rocfft_set_scale_template<T>(desc, scale),
                    "rocfft_plan_descrption_set_scale failed");

        LIB_V_THROW(rocfft_plan_create_template<T>(&plan,
                                                   _placement,
                                                   _transformation_direction,
                                                   dim,
                                                   lengths.data(),
                                                   batch_size,
                                                   desc),
                    "rocfft_plan_create failed");
    }

    /*****************************************************/
    void transform(bool explicit_intermediate_buffer = use_explicit_intermediate_buffer)
    {

        write_local_input_buffer_to_gpu(); // perform memory copy from cpu to gpu

        LIB_V_THROW(rocfft_execution_info_create(&info), "rocfft_execution_info_create failed");

        if(device_workspace != NULL) // if device_workspace is required
        {
            LIB_V_THROW(rocfft_execution_info_set_work_buffer(
                            info, device_workspace, device_workspace_size),
                        "rocfft_execution_info_set_work_buffer failed");
        }

        // Execute once for basic functional test

        // if inplace transform, NULL; else output_device buffer
        void** BuffersOut
            = (_placement == rocfft_placement_inplace) ? NULL : &output_device_buffers[0];
        LIB_V_THROW(rocfft_execute(plan, input_device_buffers, BuffersOut, info),
                    "rocfft_execute failed");

        HIP_V_THROW(hipDeviceSynchronize(), "hipDeviceSynchronize failed");

        if(_placement == rocfft_placement_inplace)
        {
            read_gpu_result_to_input_buffer();
        }
        else
        {
            read_gpu_result_to_output_buffer();
        }
    }

    /*****************************************************/
    void verbose_output()
    {

        cout << "transform parameters as seen by rocfft:" << endl;

        if(_placement == rocfft_placement_inplace)
            cout << "in-place" << endl;
        else
            cout << "out-of-place" << endl;

        cout << "input buffer byte size " << input.size_in_bytes() << endl;
        cout << "output buffer byte size " << output.size_in_bytes() << endl;
    }

    /*****************************************************/
    bool is_real(const rocfft_array_type layout)
    {
        return layout == rocfft_array_type_real;
    }

    /*****************************************************/
    bool is_planar(const rocfft_array_type layout)
    {
        return (layout == rocfft_array_type_complex_planar
                || layout == rocfft_array_type_hermitian_planar);
    }

    /*****************************************************/
    bool is_interleaved(const rocfft_array_type layout)
    {
        return (layout == rocfft_array_type_complex_interleaved
                || layout == rocfft_array_type_hermitian_interleaved);
    }

    /*****************************************************/
    bool is_complex(const rocfft_array_type layout)
    {
        return (layout == rocfft_array_type_complex_interleaved
                || layout == rocfft_array_type_complex_planar);
    }

    /*****************************************************/
    bool is_hermitian(const rocfft_array_type layout)
    {
        return (layout == rocfft_array_type_hermitian_interleaved
                || layout == rocfft_array_type_hermitian_planar);
    }

    /*****************************************************/
    void set_data_to_value(T real)
    {
        input.set_all_to_value(real);
    }

    /*****************************************************/
    void set_data_to_value(T real, T imag)
    {
        input.set_all_to_value(real, imag);
    }

    /*****************************************************/
    void set_data_to_sawtooth(T max)
    {
        input.set_all_to_sawtooth(max);
    }

    /*****************************************************/
    void set_data_to_impulse()
    {
        input.set_all_to_impulse();
    }

    /*****************************************************/
    void set_data_to_random()
    {
        input.set_all_to_random();
    }

    /*****************************************************/
    void set_data_to_buffer(buffer<T> other_buffer)
    {
        input = other_buffer;
    }

    /*****************************************************/
    buffer<T>& input_buffer()
    {
        return input;
    }

    /*****************************************************/
    buffer<T>& output_buffer()
    {
        return output;
    }

    /*****************************************************/
    // Do not need to check placement valid or not. the checking has been done at
    // the very beginning in the constructor
    buffer<T>& result()
    {
        if(_placement == rocfft_placement_inplace)
            return input;
        else
            return output;
    }

    /*****************************************************/
    ~rocfft()
    {
        for(int i = 0; i < 2; i++)
        {
            if(input_device_buffers[i] != NULL)
            {
                hipFree(input_device_buffers[i]);
            }
            if(output_device_buffers[i] != NULL)
            {
                hipFree(output_device_buffers[i]);
            }
        }

        if(device_workspace != NULL)
            hipFree(device_workspace);

        if(desc != NULL)
        {
            LIB_V_THROW(rocfft_plan_description_destroy(desc),
                        "rocfft_plan_description_destroy failed");
        }

        if(info != NULL)
        {
            LIB_V_THROW(rocfft_execution_info_destroy(info),
                        "rocfft_execution_info_destroy failed");
        }

        LIB_V_THROW(rocfft_plan_destroy(plan), "rocfft_plan_destroy failed");
        LIB_V_THROW(rocfft_cleanup(), "rocfft_cleanup failed");
    }

private:
    /*****************************************************/
    void write_local_input_buffer_to_gpu()
    {

        // size_in_bytes is calculated in input buffer
        if(is_planar(_input_layout))
        {
            HIP_V_THROW(hipMemcpy(input_device_buffers[0],
                                  input.real_ptr(),
                                  input.size_in_bytes(),
                                  hipMemcpyHostToDevice),
                        "hipMemcpy failed");
            HIP_V_THROW(hipMemcpy(input_device_buffers[1],
                                  input.imag_ptr(),
                                  input.size_in_bytes(),
                                  hipMemcpyHostToDevice),
                        "hipMemcpy failed");
        }
        else if(is_interleaved(_input_layout))
        {
            HIP_V_THROW(hipMemcpy(input_device_buffers[0],
                                  input.interleaved_ptr(),
                                  input.size_in_bytes(),
                                  hipMemcpyHostToDevice),
                        "hipMemcpy failed");
        }
        else if(is_real(_input_layout))
        {
            HIP_V_THROW(hipMemcpy(input_device_buffers[0],
                                  input.real_ptr(),
                                  input.size_in_bytes(),
                                  hipMemcpyHostToDevice),
                        "hipMemcpy failed");
        }
    }

    void read_gpu_result_to_output_buffer()
    {

        if(is_planar(_output_layout))
        {
            HIP_V_THROW(hipMemcpy(output.real_ptr(),
                                  output_device_buffers[0],
                                  output.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
            HIP_V_THROW(hipMemcpy(output.imag_ptr(),
                                  output_device_buffers[1],
                                  output.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }
        else if(is_interleaved(_output_layout))
        {
            HIP_V_THROW(hipMemcpy(output.interleaved_ptr(),
                                  output_device_buffers[0],
                                  output.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }
        else if(is_real(_output_layout))
        {
            HIP_V_THROW(hipMemcpy(output.real_ptr(),
                                  output_device_buffers[0],
                                  output.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }
    }

    void read_gpu_result_to_input_buffer()
    {

        if(is_planar(_input_layout))
        {
            HIP_V_THROW(hipMemcpy(input.real_ptr(),
                                  input_device_buffers[0],
                                  input.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
            HIP_V_THROW(hipMemcpy(input.imag_ptr(),
                                  input_device_buffers[1],
                                  input.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }
        else if(is_interleaved(_input_layout))
        {
            HIP_V_THROW(hipMemcpy(input.interleaved_ptr(),
                                  input_device_buffers[0],
                                  input.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }
        else if(is_real(_input_layout))
        {
            HIP_V_THROW(hipMemcpy(input.real_ptr(),
                                  input_device_buffers[0],
                                  input.size_in_bytes(),
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }
    }

    void argument_check()
    {
        switch(_input_layout)
        {
        case rocfft_array_type_complex_interleaved:
        case rocfft_array_type_complex_planar:
        case rocfft_array_type_hermitian_interleaved:
        case rocfft_array_type_hermitian_planar:
        case rocfft_array_type_real:
            break;
        default:
            //    Don't recognize input layout
            throw std::runtime_error("Invalid in_array_type");
        }

        switch(_output_layout)
        {
        case rocfft_array_type_complex_interleaved:
        case rocfft_array_type_complex_planar:
        case rocfft_array_type_hermitian_interleaved:
        case rocfft_array_type_hermitian_planar:
        case rocfft_array_type_real:
            break;
        default:
            //    Don't recognize output layout
            throw std::runtime_error("Invalid out_array_type");
        }

        //_input_layout and _output_layout must compatible
        if((_placement == rocfft_placement_inplace) && (_input_layout != _output_layout))
        {
            switch(_input_layout)
            {
            case rocfft_array_type_complex_interleaved:
            {
                if((_output_layout == rocfft_array_type_complex_planar)
                   || (_output_layout == rocfft_array_type_hermitian_planar))
                {
                    throw std::runtime_error("Cannot use the same buffer for "
                                             "interleaved->planar in-place transforms");
                }
                break;
            }
            case rocfft_array_type_complex_planar:
            {
                if((_output_layout == rocfft_array_type_complex_interleaved)
                   || (_output_layout == rocfft_array_type_hermitian_interleaved))
                {
                    throw std::runtime_error("Cannot use the same buffer for "
                                             "planar->interleaved in-place transforms");
                }
                break;
            }
            case rocfft_array_type_hermitian_interleaved:
            {
                if(_output_layout != rocfft_array_type_real)
                {
                    throw std::runtime_error("Cannot use the same buffer for "
                                             "interleaved->planar in-place transforms");
                }
                break;
            }
            case rocfft_array_type_hermitian_planar:
            {
                throw std::runtime_error("Cannot use the same buffer for "
                                         "planar->interleaved in-place transforms");
                break;
            }
            case rocfft_array_type_real:
            {
                if((_output_layout == rocfft_array_type_complex_planar)
                   || (_output_layout == rocfft_array_type_hermitian_planar))
                {
                    throw std::runtime_error("Cannot use the same buffer for "
                                             "interleaved->planar in-place transforms");
                }
                break;
            }
            } // end switch
        } // end if
    } // end check
};

#endif
