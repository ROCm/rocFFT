/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

/*! @file rocfft.h
 *  rocfft.h defines all the public interfaces and types
 *  */

#ifndef __ROCFFT_H__
#define __ROCFFT_H__

#include "rocfft-export.h"
#include "rocfft-version.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef _WIN32
#include <cstddef>
#endif

/*! @brief Pointer type to plan structure
 *  @details This type is used to declare a plan handle that can be initialized
 * with rocfft_plan_create
 *  */
typedef struct rocfft_plan_t* rocfft_plan;

/*! @brief Pointer type to plan description structure
 *  @details This type is used to declare a plan description handle that can be
 * initialized with rocfft_plan_description_create
 *  */
typedef struct rocfft_plan_description_t* rocfft_plan_description;

/*! @brief Pointer type to execution info structure
 *  @details This type is used to declare an execution info handle that can be
 * initialized with rocfft_execution_info_create
 *  */
typedef struct rocfft_execution_info_t* rocfft_execution_info;

/*! @brief rocfft status/error codes */
typedef enum rocfft_status_e
{
    rocfft_status_success,
    rocfft_status_failure,
    rocfft_status_invalid_arg_value,
    rocfft_status_invalid_dimensions,
    rocfft_status_invalid_array_type,
    rocfft_status_invalid_strides,
    rocfft_status_invalid_distance,
    rocfft_status_invalid_offset,
} rocfft_status;

/*! @brief Type of transform */
typedef enum rocfft_transform_type_e
{
    rocfft_transform_type_complex_forward,
    rocfft_transform_type_complex_inverse,
    rocfft_transform_type_real_forward,
    rocfft_transform_type_real_inverse,
} rocfft_transform_type;

/*! @brief Precision */
typedef enum rocfft_precision_e
{
    rocfft_precision_single,
    rocfft_precision_double,
} rocfft_precision;

/*! @brief Result placement */
typedef enum rocfft_result_placement_e
{
    rocfft_placement_inplace,
    rocfft_placement_notinplace,
} rocfft_result_placement;

/*! @brief Array type */
typedef enum rocfft_array_type_e
{
    rocfft_array_type_complex_interleaved,
    rocfft_array_type_complex_planar,
    rocfft_array_type_real,
    rocfft_array_type_hermitian_interleaved,
    rocfft_array_type_hermitian_planar,
    rocfft_array_type_unset,
} rocfft_array_type;

/*! @brief Execution mode */
typedef enum rocfft_execution_mode_e
{
    rocfft_exec_mode_nonblocking,
    rocfft_exec_mode_nonblocking_with_flush,
    rocfft_exec_mode_blocking,
} rocfft_execution_mode;

/*! @brief Library setup function, called once in program before start of
 * library use */
ROCFFT_EXPORT rocfft_status rocfft_setup();

/*! @brief Library cleanup function, called once in program after end of library
 * use */
ROCFFT_EXPORT rocfft_status rocfft_cleanup();

/*! @brief Create an FFT plan
 *
 *  @details This API creates a plan, which the user can execute subsequently.
 * This function
 *  takes many of the fundamental parameters needed to specify a transform. The
 * parameters are
 *  self explanatory. The dimensions parameter can take a value of 1,2 or 3. The
 * 'lengths' array specifies
 *  size of data in each dimension. Note that lengths[0] is the size of the
 * innermost dimension, lengths[1]
 *  is the next higher dimension and so on. The 'number_of_transforms' parameter
 * specifies how many transforms
 *  (of the same kind) needs to be computed. By specifying a value greater than
 * 1, a batch of transforms can
 *  be computed with a single api call. Additionally, a handle to a plan
 * description can be passed for more
 *  detailed transforms. For simple transforms, this parameter can be set to
 * null ptr.
 *
 *  @param[out] plan plan handle
 *  @param[in] placement placement of result
 *  @param[in] transform_type type of transform
 *  @param[in] precision precision
 *  @param[in] dimensions dimensions
 *  @param[in] lengths dimensions sized array of transform lengths
 *  @param[in] number_of_transforms number of transforms
 *  @param[in] description description handle created by
 * rocfft_plan_description_create; can be
 *  null ptr for simple transforms
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_create(rocfft_plan*                  plan,
                                               rocfft_result_placement       placement,
                                               rocfft_transform_type         transform_type,
                                               rocfft_precision              precision,
                                               size_t                        dimensions,
                                               const size_t*                 lengths,
                                               size_t                        number_of_transforms,
                                               const rocfft_plan_description description);

/*! @brief Execute an FFT plan
 *
 *  @details This API executes an FFT plan on buffers given by the user. If the
 * transform is in-place,
 *  only the input buffer is needed and the output buffer parameter can be set
 * to NULL. For not in-place
 *  transforms, output buffers have to be specified. Note that both input and
 * output buffer are arrays of
 *  pointers, this is to facilitate passing planar buffers where real and
 * imaginary parts are in 2 separate
 *  buffers. For the default interleaved format, just a unit sized array holding
 * the pointer to input/output
 *  buffer need to be passed. The final parameter in this function is an
 * execution_info handle. This parameter
 *  serves as a way for the user to control execution, as well as for the
 * library to pass any execution
 *  related information back to the user.
 *
 *  @param[in] plan plan handle
 *  @param[in,out] in_buffer array (of size 1 for interleaved data, of size 2
 * for planar data) of input buffers
 *  @param[in,out] out_buffer array (of size 1 for interleaved data, of size 2
 * for planar data) of output buffers, can be nullptr for inplace result
 * placement
 *  @param[in] info execution info handle created by
 * rocfft_execution_info_create
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execute(const rocfft_plan     plan,
                                           void*                 in_buffer[],
                                           void*                 out_buffer[],
                                           rocfft_execution_info info);

/*! @brief Destroy an FFT plan
 *  @details This API frees the plan. This function destructs a plan after it is
 * no longer needed.
 *  @param[in] plan plan handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_destroy(rocfft_plan plan);

#if 0
/*! @brief Set scaling factor in single precision
 *  @details This is one of plan description functions to specify optional additional plan properties using the description handle. This API specifies scaling factor.
 *  @param[in] description description handle
 *  @param[in] scale scaling factor
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_set_scale_float( rocfft_plan_description description, const float scale );

/*! @brief Set scaling factor in double precision
 *  @details This is one of plan description functions to specify optional additional plan properties using the description handle. This API specifies scaling factor.
 *  @param[in] description description handle
 *  @param[in] scale scaling factor
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_set_scale_double( rocfft_plan_description description, const double scale );
#endif

/*!
 *  @brief Set data layout
 * 
 *  @details This is one of plan description functions to specify
 *   optional additional plan properties using the description
 *   handle. This API specifies the layout of buffers.
 * 
 *  This function can be used to specify input and output array
 *  types. Not all combinations of array types are supported and error
 *  code will be returned for unsupported cases.  Additionally, input
 *  and output buffer offsets can be specified. The function can be
 *  used to specify custom layout of data, with the ability to specify
 *  stride between consecutive elements in all dimensions. Also,
 *  distance between transform data members can be specified. The
 *  library will choose appropriate defaults if offsets/strides are
 *  set to null ptr and/or distances set to 0.
 * 
 *  @param[in, out] description description handle
 *  @param[in] in_array_type array type of input buffer
 *  @param[in] out_array_type array type of output buffer
 *  @param[in] in_offsets offsets, in element units, to start of data in input buffer
 *  @param[in] out_offsets offsets, in element units, to start of data in output buffer
 *  @param[in] in_strides_size size of in_strides array (must be equal to transform dimensions)
 *  @param[in] in_strides array of strides, in each dimension, of
 *   input buffer; if set to null ptr library chooses defaults
 *  @param[in] in_distance distance between start of each data instance in input buffer
 *  @param[in] out_strides_size size of out_strides array (must be
 *  equal to transform dimensions)
 *  @param[in] out_strides array of strides, in each dimension, of
 *   output buffer; if set to null ptr library chooses defaults
 *  @param[in] out_distance distance between start of each data instance in output buffer
 */
ROCFFT_EXPORT rocfft_status
    rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                            const rocfft_array_type in_array_type,
                                            const rocfft_array_type out_array_type,
                                            const size_t*           in_offsets,
                                            const size_t*           out_offsets,
                                            const size_t            in_strides_size,
                                            const size_t*           in_strides,
                                            const size_t            in_distance,
                                            const size_t            out_strides_size,
                                            const size_t*           out_strides,
                                            const size_t            out_distance);

/*! @brief Get library version string
 *
 * @param[in, out] buf buffer of version string
 * @param[in] len the length of input string buffer, expecting minimum 30
 */
ROCFFT_EXPORT rocfft_status rocfft_get_version_string(char* buf, size_t len);

#if 0
/*! @brief Set devices in plan description
 *  @details This is one of plan description functions to specify optional additional plan properties using the description handle. This API specifies what compute devices to target.
 *  @param[in] description description handle
 *  @param[in] devices array of device identifiers
 *  @param[in] number_of_devices number of devices (size of devices array)
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_set_devices( rocfft_plan_description description, void *devices, size_t number_of_devices );
#endif

/*! @brief Get work buffer size
 *  @details This is one of plan query functions to obtain information regarding
 * a plan. This API gets the work buffer size.
 *  @param[in] plan plan handle
 *  @param[out] size_in_bytes size of needed work buffer in bytes
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan,
                                                             size_t*           size_in_bytes);

/*! @brief Print all plan information
 *  @details This is one of plan query functions to obtain information regarding
 * a plan. This API prints all plan info to stdout to help user verify plan
 * specification.
 *  @param[in] plan plan handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_get_print(const rocfft_plan plan);

/*! @brief Create plan description
 *  @details This API creates a plan description with which the user can set
 * more plan properties
 *  @param[out] description plan description handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_create(rocfft_plan_description* description);

/*! @brief Destroy a plan description
 *  @details This API frees the plan description
 *  @param[in] description plan description handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description);

/*! @brief Create execution info
 *  @details This API creates an execution info with which the user can control
 * plan execution & retrieve execution information
 *  @param[out] info execution info handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_create(rocfft_execution_info* info);

/*! @brief Destroy an execution info
 *  @details This API frees the execution info
 *  @param[in] info execution info handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info);

/*! @brief Set work buffer in execution info
 *
 *  @details This is one of the execution info functions to specify optional
 * additional information to control execution.
 *  This API specifies work buffer needed. It has to be called before the call
 * to rocfft_execute.
 *  When a non-zero value is obtained from rocfft_plan_get_work_buffer_size,
 * that means the library needs a work buffer
 *  to compute the transform. In this case, the user has to allocate the work
 * buffer and pass it to the library via this api.
 *
 *  @param[in] info execution info handle
 *  @param[in] work_buffer work buffer
 *  @param[in] size_in_bytes size of work buffer in bytes
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info,
                                                                  void*                 work_buffer,
                                                                  const size_t size_in_bytes);

#if 0
/*! @brief Set execution mode in execution info
 *  @details This is one of the execution info functions to specify optional additional information to control execution.
 *  This API specifies execution mode. It has to be called before the call to rocfft_execute.
 *  Appropriate enumeration value can be specified to control blocking/non-blocking behavior of the rocfft_execute call.
 *  @param[in] info execution info handle
 *  @param[in] mode execution mode
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_set_mode( rocfft_execution_info info,
                                                            const rocfft_execution_mode mode );
#endif

/*! @brief Set stream in execution info
 *  @details This is one of the execution info functions to specify optional
 * additional information to control execution.
 *  This API specifies compute stream. It has to be called before the call to
 * rocfft_execute.
 *  It is the underlying device queue/stream where the library computations
 * would be inserted. The library assumes user
 *  has created such a stream in the program and merely assigns work to the
 * stream. The stream must be of type hipStream_t. It is an error to pass the
 * address of a hipStream_t object.
 *  @param[in] info execution info handle
 *  @param[in] stream underlying compute stream
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info,
                                                             void*                 stream);

#if 0
/*! @brief Get events from execution info
 *  @details This is one of the execution info functions to retrieve information from execution.
 *  This API obtains event information. It has to be called after the call to rocfft_execute.
 *  This gets handles to events that the library created around one or more kernel launches during execution.
 *  @param[in] info execution info handle
 *  @param[out] events array of events
 *  @param[out] number_of_events number of events (size of events array)
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_get_events( const rocfft_execution_info info, void **events, size_t *number_of_events );
#endif

/*! \brief Indicates if layer is active with bitmask*/
typedef enum rocfft_layer_mode_
{
    rocfft_layer_mode_none        = 0b0000000000,
    rocfft_layer_mode_log_trace   = 0b0000000001,
    rocfft_layer_mode_log_bench   = 0b0000000010,
    rocfft_layer_mode_log_profile = 0b0000000100,
} rocfft_layer_mode;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __ROCFFT_H__ */
