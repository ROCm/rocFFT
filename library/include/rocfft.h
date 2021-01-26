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

#ifdef rocfft_EXPORTS
#include "rocfft-export.h"
#else
#define ROCFFT_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef _WIN32
#include <cstddef>
#endif

/*! @brief Pointer type to plan structure
 *  @details This type is used to declare a plan handle that can be initialized
 * with ::rocfft_plan_create.
 *  */
typedef struct rocfft_plan_t* rocfft_plan;

/*! @brief Pointer type to plan description structure
 *  @details This type is used to declare a plan description handle that can be
 * initialized with ::rocfft_plan_description_create.
 *  */
typedef struct rocfft_plan_description_t* rocfft_plan_description;

/*! @brief Pointer type to execution info structure
 *  @details This type is used to declare an execution info handle that can be
 * initialized with ::rocfft_execution_info_create.
 *  */
typedef struct rocfft_execution_info_t* rocfft_execution_info;

/*! @brief rocFFT status/error codes */
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
    rocfft_status_invalid_work_buffer,
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

/*! @brief Result placement
 *  @details Declares where the output of the transform should be
 *  placed.  Note that input buffers may still be overwritten
 *  during execution of a transform, even if the transform is not
 *  in-place.
*/
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

#if 0
/*! @brief Execution mode */
typedef enum rocfft_execution_mode_e
{
    rocfft_exec_mode_nonblocking,
    rocfft_exec_mode_nonblocking_with_flush,
    rocfft_exec_mode_blocking,
} rocfft_execution_mode;
#endif

/*! @brief Library setup function, called once in program before start of
 * library use */
ROCFFT_EXPORT rocfft_status rocfft_setup();

/*! @brief Library cleanup function, called once in program after end of library
 * use */
ROCFFT_EXPORT rocfft_status rocfft_cleanup();

/*! @brief Create an FFT plan
 *
 *  @details This API creates a plan, which the user can execute
 *  subsequently.  This function takes many of the fundamental
 *  parameters needed to specify a transform.
 *
 *  The dimensions parameter can take a value of 1, 2, or 3. The
 *  'lengths' array specifies the size of data in each dimension. Note
 *  that lengths[0] is the size of the innermost dimension, lengths[1]
 *  is the next higher dimension and so on (column-major ordering).
 *
 *  The 'number_of_transforms' parameter specifies how many
 *  transforms (of the same kind) needs to be computed. By specifying
 *  a value greater than 1, a batch of transforms can be computed
 *  with a single API call.
 *
 *  Additionally, a handle to a plan description can be passed for
 *  more detailed transforms. For simple transforms, this parameter
 *  can be set to NULL.
 *
 *  The plan must be destroyed with a call to ::rocfft_plan_destroy.
 *
 *  @param[out] plan plan handle
 *  @param[in] placement placement of result
 *  @param[in] transform_type type of transform
 *  @param[in] precision precision
 *  @param[in] dimensions dimensions
 *  @param[in] lengths dimensions-sized array of transform lengths
 *  @param[in] number_of_transforms number of transforms
 *  @param[in] description description handle created by
 * rocfft_plan_description_create; can be
 *  NULL for simple transforms
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
 *  @details This API executes an FFT plan on buffers given by the user.
 *
 *  If the transform is in-place, only the input buffer is needed and
 *  the output buffer parameter can be set to NULL. For not in-place
 *  transforms, output buffers have to be specified.
 *
 *  Input and output buffer are arrays of pointers.  Interleaved
 *  array formats are the default, and require just one pointer per
 *  input or output buffer.  Planar array formats require two
 *  pointers per input or output buffer - real and imaginary
 *  pointers, in that order.
 *
 *  Note that input buffers may still be overwritten during execution
 *  of a transform, even if the transform is not in-place.
 *
 *  The final parameter in this function is a rocfft_execution_info
 *  handle. This optional parameter serves as a way for the user to control
 *  execution streams and work buffers.
 *
 *  @param[in] plan plan handle
 *  @param[in,out] in_buffer array (of size 1 for interleaved data, of size 2
 * for planar data) of input buffers
 *  @param[in,out] out_buffer array (of size 1 for interleaved data, of size 2
 * for planar data) of output buffers, ignored for in-place transforms
 *  @param[in] info execution info handle created by
 * rocfft_execution_info_create
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execute(const rocfft_plan     plan,
                                           void*                 in_buffer[],
                                           void*                 out_buffer[],
                                           rocfft_execution_info info);

/*! @brief Destroy an FFT plan
 *  @details This API frees the plan after it is no longer needed.
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
 *  @brief Set advanced data layout parameters on a plan description
 * 
 *  @details This API specifies advanced layout of input/output
 *  buffers for a plan description.
 * 
 *  The following parameters are supported for inputs and outputs:
 *
 *  * Array type (real, hermitian, or complex data, in either
 *    interleaved or planar format).
 *      * Real forward transforms require real input and hermitian output.
 *      * Real inverse transforms require hermitian input and real output.
 *      * Complex transforms require complex input and output.
 *      * Hermitian and complex data defaults to interleaved if a specific
          format is not specified.
 *  * Offset of first data element in the data buffer.  Defaults to 0 if unspecified.
 *  * Stride between consecutive elements in each dimension.  Defaults
      to contiguous data in all dimensions if unspecified.
 *  * Distance between consecutive batches.  Defaults to contiguous
      batches if unspecified.
 *
 *  Not all combinations of array types are supported and error codes
 *  will be returned for unsupported cases.
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
 * @param[in, out] buf buffer that receives the version string
 * @param[in] len length of buf, minimum 30 characters
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
 *  @details Get the work buffer size required for a plan.
 *  @param[in] plan plan handle
 *  @param[out] size_in_bytes size of needed work buffer in bytes
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan,
                                                             size_t*           size_in_bytes);

/*! @brief Print all plan information
 *  @details Prints plan details to stdout, to aid debugging
 *  @param[in] plan plan handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_get_print(const rocfft_plan plan);

/*! @brief Create plan description
 *  @details This API creates a plan description with which the user
 * can set extra plan properties.  The plan description must be freed
 * with a call to ::rocfft_plan_description_destroy.
 *  @param[out] description plan description handle */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_create(rocfft_plan_description* description);

/*! @brief Destroy a plan description
 *  @details This API frees the plan description.  A plan description
 *  can be freed any time after it is passed to ::rocfft_plan_create.
 *  @param[in] description plan description handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description);

/*! @brief Create execution info
 *  @details This API creates an execution info with which the user
 * can control plan execution and work buffers.  The execution info must be freed
 * with a call to ::rocfft_execution_info_destroy.
 *  @param[out] info execution info handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_create(rocfft_execution_info* info);

/*! @brief Destroy an execution info
 *  @details This API frees the execution info.  An execution info
 *  object can be freed any time after it is passed to
 *  ::rocfft_execute.
 *  @param[in] info execution info handle
 *  */
ROCFFT_EXPORT rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info);

/*! @brief Set work buffer in execution info
 *
 *  @details This is one of the execution info functions to specify
 *  optional additional information to control execution.  This API
 *  provides a work buffer for the transform. It must be called
 *  before ::rocfft_execute.
 *
 *  When a non-zero value is obtained from
 *  ::rocfft_plan_get_work_buffer_size, that means the library needs a
 *  work buffer to compute the transform. In this case, the user
 *  should allocate the work buffer and pass it to the library via
 *  this API.
 *
 *  If a work buffer is required for the transform but is not
 *  specified using this function, ::rocfft_execute will automatically
 *  allocate the required buffer and free it when execution is
 *  finished.
 *
 *  Users should allocate their own work buffers if they need precise
 *  control over the lifetimes of those buffers, or if multiple plans
 *  need to share the same buffer.
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
 *  @details Associates an existing compute stream to a plan.  This
 * must be called before the call to ::rocfft_execute.
 *
 *  Once the association is made, execution of the FFT will run the
 *  computation through the specified stream.
 *
 *  The stream must be of type hipStream_t. It is an error to pass
 *  the address of a hipStream_t object.
 *
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

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __ROCFFT_H__ */
