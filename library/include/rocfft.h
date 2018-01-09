/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

/*! @file rocfft.h
 *  rocfft.h defines all the public interfaces and types
 *  */

#ifndef __ROCFFT_H__
#define __ROCFFT_H__

#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


/*! @brief Opaque pointer type to plan structure */
typedef struct rocfft_plan_t *rocfft_plan;
/*! @brief Opaque pointer type to plan description structure */
typedef struct rocfft_plan_description_t *rocfft_plan_description;
/*! @brief Opaque pointer type to execution info structure */
typedef struct rocfft_execution_info_t *rocfft_execution_info;

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
} rocfft_array_type;

/*! @brief Execution mode */
typedef enum rocfft_execution_mode_e
{
    rocfft_exec_mode_nonblocking,
    rocfft_exec_mode_nonblocking_with_flush,
    rocfft_exec_mode_blocking,
} rocfft_execution_mode;



/*! @brief Library setup function, called once in program before start of library use */
DLL_PUBLIC rocfft_status rocfft_setup();

/*! @brief Library cleanup function, called once in program after end of library use */
DLL_PUBLIC rocfft_status rocfft_cleanup();


/*! @brief Create an FFT plan
 *  @details This API creates a plan, which the user can execute subsequently
 *  @param[out] plan plan handle
 *  @param[in] placement placement of result
 *  @param[in] transform_type type of transform
 *  @param[in] precision precision
 *  @param[in] dimensions dimensions
 *  @param[in] lengths dimensions sized array of transform lengths
 *  @param[in] number_of_transforms number of transforms
 *  @param[in] description description handle created by rocfft_plan_description_create
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_create(    rocfft_plan *plan,
                            					rocfft_result_placement placement,
                                                rocfft_transform_type transform_type, rocfft_precision precision,
                                                size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                                const rocfft_plan_description description );


/*! @brief Execute an FFT plan
 *  @details This API executes an FFT plan on buffers given by the user
 *  @param[in] plan plan handle
 *  @param[in,out] in_buffer array (of size 1 for interleaved data, of size 2 for planar data) of input buffers 
 *  @param[in,out] out_buffer array (of size 1 for interleaved data, of size 2 for planar data) of output buffers, can be nullptr for inplace result placement
 *  @param[in] info execution info handle created by rocfft_execution_info_create
 *  */
DLL_PUBLIC rocfft_status rocfft_execute(    const rocfft_plan plan,
                                            void *in_buffer[],
                                            void *out_buffer[],
                                            rocfft_execution_info info );

/*! @brief Destroy an FFT plan
 *  @details This API frees the plan
 *  @param[in] plan plan handle
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_destroy( rocfft_plan plan );


/*! @brief Set scaling factor in single precision
 *  @details This is one of plan description funtions to specify optional additional plan properties using the description handle. This API specifies scaling factor.
 *  @param[in] description description handle
 *  @param[in] scale scaling factor
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_description_set_scale_float( rocfft_plan_description description, float scale );

/*! @brief Set scaling factor in double precision
 *  @details This is one of plan description funtions to specify optional additional plan properties using the description handle. This API specifies scaling factor.
 *  @param[in] description description handle
 *  @param[in] scale scaling factor
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_description_set_scale_double( rocfft_plan_description description, double scale );

/*! @brief Set data layout 
 *  @details This is one of plan description funtions to specify optional additional plan properties using the description handle. This API specifies the layout of buffers.
 *  @param[in] description description handle
 *  @param[in] in_array_type array type of input buffer 
 *  @param[in] out_array_type array type of output buffer 
 *  @param[in] in_offsets offsets, in element units, to start of data in input buffer 
 *  @param[in] out_offsets offsets, in element units, to start of data in output buffer 
 *  @param[in] in_strides_size size of in_strides array (must be equal to transform dimensions)
 *  @param[in] in_strides array of strides, in each dimension, of input buffer 
 *  @param[in] in_distance distance between start of each data instance in input buffer
 *  @param[in] out_strides_size size of out_strides array (must be equal to transform dimensions)
 *  @param[in] out_strides array of strides, in each dimension, of output buffer 
 *  @param[in] out_distance distance between start of each data instance in output buffer
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_description_set_data_layout(   rocfft_plan_description description,
                                                                    rocfft_array_type in_array_type, rocfft_array_type out_array_type,
                                                                    const size_t *in_offsets, const size_t *out_offsets,
                                                                    size_t in_strides_size, const size_t *in_strides, size_t in_distance,
                                                                    size_t out_strides_size, const size_t *out_strides, size_t out_distance );

/*! @brief Set devices in plan description
 *  @details This is one of plan description funtions to specify optional additional plan properties using the description handle. This API specifies what compute devices to target.
 *  @param[in] description description handle
 *  @param[in] devices array of device identifiers
 *  @param[in] number_of_devices number of devices (size of devices array)
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_description_set_devices( rocfft_plan_description description, void *devices, size_t number_of_devices );


/*! @brief Get work buffer size
 *  @details This is one of plan query functions to obtain information regarding a plan. This API gets the work buffer size.
 *  @param[in] plan plan handle
 *  @param[out] size_in_bytes size of needed work buffer in bytes 
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_get_work_buffer_size( const rocfft_plan plan, size_t *size_in_bytes );

/*! @brief Print all plan information 
 *  @details This is one of plan query functions to obtain information regarding a plan. This API prints all plan info to stdout to help user verify plan properties.
 *  @param[in] plan plan handle
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_get_print( const rocfft_plan plan );


/*! @brief Create plan description 
 *  @details This API creates a plan description with which the user can set more plan properties
 *  @param[out] description plan description handle
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_description_create( rocfft_plan_description *description );

/*! @brief Destroy a plan description
 *  @details This API frees the plan description
 *  @param[in] description plan description handle
 *  */
DLL_PUBLIC rocfft_status rocfft_plan_description_destroy( rocfft_plan_description description );

/*! @brief Create execution info 
 *  @details This API creates an execution info with which the user can control plan execution & retrieve execution information
 *  @param[out] info execution info handle
 *  */
DLL_PUBLIC rocfft_status rocfft_execution_info_create( rocfft_execution_info *info );

/*! @brief Destroy an execution info 
 *  @details This API frees the execution info 
 *  @param[in] info execution info handle
 *  */
DLL_PUBLIC rocfft_status rocfft_execution_info_destroy( rocfft_execution_info info );


/*! @brief Set work buffer in execution info
 *  @details This is one of the execution info funtions to specify optional additional information to control execution. This API specifies work buffer needed.
 *  @param[in] info execution info handle
 *  @param[in] work_buffer work buffer
 *  @param[in] size_in_bytes size of work buffer in bytes
 *  */
DLL_PUBLIC rocfft_status rocfft_execution_info_set_work_buffer( rocfft_execution_info info, void *work_buffer, size_t size_in_bytes );

/*! @brief Set execution mode in execution info
 *  @details This is one of the execution info funtions to specify optional additional information to control execution. This API specifies execution mode.
 *  @param[in] info execution info handle
 *  @param[in] mode execution mode
 *  */
DLL_PUBLIC rocfft_status rocfft_execution_info_set_mode( rocfft_execution_info info, rocfft_execution_mode mode );

/*! @brief Set stream in execution info
 *  @details This is one of the execution info funtions to specify optional additional information to control execution. This API specifies underlying compute stream.
 *  @param[in] info execution info handle
 *  @param[in] stream underlying compute stream
 *  */
DLL_PUBLIC rocfft_status rocfft_execution_info_set_stream( rocfft_execution_info info, void *stream );


/*! @brief Get events from execution info
 *  @details This is one of the execution info funtions to retrieve information from execution. This API obtains event information.
 *  @param[in] info execution info handle
 *  @param[out] events array of events 
 *  @param[out] number_of_events number of events (size of events array) 
 *  */
DLL_PUBLIC rocfft_status rocfft_execution_info_get_events( const rocfft_execution_info info, void **events, size_t *number_of_events );



#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __ROCFFT_H__

